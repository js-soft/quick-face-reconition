import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
import torchinfo
import numpy as np
import plotnine as p9
import pandas as pd
from src import util


class VGG16DualHead(nn.Module):
    def __init__(self, freeze_vgg16_weights=True, device=None):
        super().__init__()
        # Pre-trained VGG16 with an additional convlayer.
        self.vgg16_base = models.vgg16(weights="DEFAULT").features
        if freeze_vgg16_weights:
            for p in self.vgg16_base.parameters():
                p.requires_grad = False
        self.cn2 = nn.Conv2d(512, 32, kernel_size=(3, 3))

        # Head 1: Bounding box
        self.bbox_dense1 = nn.Linear(800, 512)
        self.bbox_dense2 = nn.Linear(512, 4)

        # Head 2: Conviction of seeing a face
        self.conviction_dense1 = nn.Linear(800, 512)
        self.conviction_dense2 = nn.Linear(512, 1)

        if device is not None:
            self.to(device)

    # Expects input shape of either (batch_size, 3, 244, 244)
    def forward(self, x):
        """x: Image data as tensors of shape (batch_size, 3, 244, 244) or (3, 244, 244), for a single sample.

        Returns tensor of shape (n, 5) or (n,)."""
        x = self.vgg16_base(x)  # (n, 512, 7, 7)
        x = self.cn2(x)  # (n, 32, 5, 5)
        x = F.relu(x)  # (n, 32, 5, 5)
        x = torch.flatten(x, -3)  # (n, 800)

        # Head 1: Bounding box
        bbox = self.bbox_dense1(x)  # (n, 512)
        bbox = F.relu(bbox)  # (n, 512)
        bbox = self.bbox_dense2(bbox)  # (n, 4)
        bbox = torch.sigmoid(bbox)  # (n, 4)

        # Head 2: Conviction, that there is a face
        conviction = self.conviction_dense1(x)  # (n, 512)
        conviction = F.relu(conviction)  # (n, 512)
        conviction = self.conviction_dense2(conviction)  # (n, 1)
        conviction = torch.sigmoid(conviction)  # (n, 1)

        result = torch.cat((bbox, conviction), dim=-1)  # (n, 5)
        return result

    def summary(self, input_size=(32, 3, 224, 224)):
        """Helper for determining the expected output shapes given an input"""
        # Pytorch uses dynamic computational graphs, whose tensor dimensions depend on
        # the shape of the input data. In order to get a listing of concrete layer
        # dimensions, a specific input shape has to be specified. See
        # https://ai.stackexchange.com/questions/3801
        torchinfo.summary(self, input_size=input_size)

    @staticmethod
    def from_state_dict_pth(fpath, device=None):
        """Helper function to initialize a model from a saved state dict."""
        state_dict = torch.load(fpath)
        model = VGG16DualHead(device=device)
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        return model


def loss(y_hat, y_true):
    """Returns per-sample loss"""
    # Destructure tensors into bboxes and convictions.
    bbox_hat, bbox_true = y_hat[:, :4], y_true[:, :4]
    conviction_hat, conviction_true = y_hat[:, 4:], y_true[:, 4:]

    # Compute individual contributions to the loss.
    bbox_loss = torch.sum(torch.square(bbox_hat - bbox_true), dim=1)
    conviction_loss = torch.sum(torch.square(conviction_hat - conviction_true), dim=1)

    # Scale conviction and return sum of losses.
    importance = 16.0
    result = bbox_loss + importance * conviction_loss

    return result


def cost(y_hat, y_true):
    """Returns mean per-sample loss"""
    lss = loss(y_hat, y_true)
    batch_size = y_hat.size()[0]
    result = torch.sum(lss, dim=0) / batch_size
    return result


def train_epoch(model: nn.Module, device, train_dataloader, optim, cb=None):
    # There exist layers whose output is designed to differ between training
    # and inference, e.g. dropout layers. We explicitly set the model to
    # training mode. See https://stackoverflow.com/questions/51433378.
    model.train()

    for batch_ii, (x, y) in enumerate(train_dataloader):
        # Create a copy of the input data on the device.
        # ???: Wouldn't it be faster to move all the training data to GPU in one step?
        # ???: Can't the device be inferred from the model's device?
        x, y = x.to(device), y.to(device)

        # Reset the gradients, which, by default, are accumulated by a call to
        # backward(). See https://stackoverflow.com/questions/48001598
        optim.zero_grad()

        # Compute the model's inference and the resulting cost.
        y_hat = model(x)
        cost_val = cost(y_hat, y)

        # Compute gradients via backprop and adjust params. On why there's
        # apparently no coupling between the loss function and the optimizer,
        # see https://stackoverflow.com/questions/53975717.
        cost_val.backward()
        optim.step()

        if cb is not None:
            cb(batch_ii, cost_val.item())


def eval_performance(model, dataloader, compute_std=False):
    """Evaluates performance of a model on a dataset"""
    result = {}

    model.eval()
    device = next(model.parameters()).device.type  # get model device

    # Pass 1: mean
    num_samples = 0
    loss_sum = 0.0
    for x, y in dataloader:
        batch_size = x.size()[0]
        num_samples += batch_size

        x, y = x.to(device), y.to(device)

        # Compute the model's inference and the resulting loss.
        y_hat = model(x)
        lss = loss(y_hat, y)
        loss_sum += torch.sum(lss, dim=0).item()

    result["loss_mean"] = loss_sum / num_samples

    # Pass 2: standard deviation
    if compute_std:
        squared_error_acc = 0.0
        num_samples = 0
        for x, y in dataloader:
            batch_size = x.size()[0]
            num_samples += batch_size
            x, y = x.to(device), y.to(device)

            # Compute the model's inference and the resulting loss.
            y_hat = model(x)
            lss = loss(y_hat, y)
            squared_error_tensor = torch.square(torch.sub(lss, result["loss_mean"]))
            squared_error_sum = torch.sum(squared_error_tensor)
            squared_error_acc += squared_error_sum

        result["loss_std"] = np.sqrt(squared_error_acc.item()) / num_samples

    return result


class DataFromDisk(Dataset):
    """Torch.Dataset returning pairs of (image, label) loaded from an input directory.

    Images are returned in numpy uint8 format, labels are (5,) dimensional
    numpy.ndarrays, where the first 4 values encode the bounding box in
    Albumentations format, and the last one indicates the likelihood of a face
    in view.

    Pairs are returned in lexicographical order of the respective filenames.

    directory: Path to input directory containing images in PNG format and one
        eponymous csv file per image, which contains its label.
    transform_x: Callable to transform the output images when using the []-operator.
    transform_y: Callable to transform the label when using the []-operator.
    """

    def __init__(self, directory, transform_x=None, transform_y=None):
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.image_paths = sorted([f"{directory}/{path}" for path in os.listdir(directory) if path.endswith(".png")])
        self.label_paths = sorted([f"{directory}/{path}" for path in os.listdir(directory) if path.endswith(".csv")])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]

        image = util.load_image_from_file(self.image_paths[idx])
        label = util.load_label_from_csv(self.label_paths[idx])

        if self.transform_x is not None:
            image = self.transform_x(image)
        if self.transform_y is not None:
            label = self.transform_y(label)

        return image, label


def plot_training_progress(training_loss, test_loss):
    """Produces a plot of training and test loss curves.

    training_loss: list of training loss values, one per epoch
    test_loss: list of test loss values, one per epoch

    Returns a plotnine.ggplot object."""
    # Transform data to long form.
    data = {
        "epoch": list(range(len(test_loss))),
        "test_loss": test_loss,
        "train_loss": training_loss,
    }
    data = pd.DataFrame.from_dict(data)
    data = pd.melt(data, id_vars=["epoch"], value_vars=["test_loss", "train_loss"])

    # fmt: off
    num_epochs = int(data["epoch"].max())
    plot = p9.ggplot(data=data) + \
        p9.geom_line(p9.aes(x="epoch", y="value", color="variable")) + \
        p9.scale_x_continuous(breaks=range(0, num_epochs, 10), minor_breaks=range(0, num_epochs)) + \
        p9.scale_y_log10()
    # fmt: on
    return plot
