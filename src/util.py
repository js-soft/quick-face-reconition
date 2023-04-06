import numpy as np
import pylab as plt
import PIL as pil
from plotnine import *
import cv2
import torch
import torchvision
import json
import pandas as pd
import plotnine as p9
from enum import Enum
from copy import deepcopy


class ImgType(Enum):
    """Enum for the different image data types that we encounter here.

    Provides some helper functionality to convert between the different image
    formats. Implemented as needed."""

    np_u8 = 1
    np_f32 = 2
    pil = 3
    torch = 4

    @staticmethod
    def typeof(obj):
        if type(obj) == np.ndarray:
            if len(obj.shape) == 3 and obj.shape[2] == 3:
                if obj.dtype == np.float32:
                    return ImgType.np_f32
                if obj.dtype == np.uint8:
                    return ImgType.np_u8
        if type(obj) == pil.Image.Image:
            return ImgType.pil
        if type(obj) == torch.Tensor:
            shape = obj.size()
            if len(shape) == 3 and shape[0] == 3:
                return ImgType.torch

        return None

    @staticmethod
    def convert(obj, to_type):
        input_type = ImgType.typeof(obj)
        if to_type is ImgType.pil:
            if input_type is ImgType.np_u8:
                return pil.Image.fromarray(obj)
            if input_type is ImgType.torch:
                return torchvision.transforms.ToPILImage()(obj)

        raise NotImplementedError()


def read_cam(resolution=(800, 600), device_id=0):
    """Returns a generator yielding images from a video capture device.

    Images are returned in np_u8 format."""
    cam = cv2.VideoCapture(device_id)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    while True:
        try:
            cam.read()  # throw away buffered image
            _, img_cam = cam.read()
            img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)  # convert to RGB
            yield img_cam
        except (GeneratorExit, Exception):
            break

    cam.release()


def load_label_from_csv(fpath: str):
    """Loads image label from csv file, returning a numpy tensor

    Labels are expected to be stored in a single-row, comma-separated array of 5 values"""
    return np.genfromtxt(fpath, delimiter=",")


def load_image_from_file(filepath: str):
    """Loads image from file and returns it in numpy uint8 format"""
    img = pil.Image.open(filepath)
    img = np.asarray(img)
    return img


def save_image(img_np, fpath):
    """Saves an image in numpy uint8 format to file"""
    img_pil = pil.Image.fromarray(img_np, "RGB")
    img_pil.save(fpath)


def overlay_bboxes(img, bboxes):
    """Returns an image with bboxes drawn on top.

    img: image in numpy uint8 format
    bboxes: iterable of bboxes in albumentations format

    Returns an image in numpy uint8 format."""
    img_pil = ImgType.convert(img, ImgType.pil)
    draw = pil.ImageDraw.Draw(img_pil)
    for bbox in bboxes:
        h, w = img_pil.height, img_pil.width
        bbox_abs = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
        draw.rectangle(((bbox_abs[0], bbox_abs[1]), (bbox_abs[2], bbox_abs[3])), outline="red")
    return np.array(img_pil)


def load_labelme_bbox(filepath: str, dtype: np.floating = np.float_):
    """Loads the bounding box from a labelme json file, returning a (4,) numpy
    floating-point tensor which encodes the bounding box in albumentations
    format"""
    with open(filepath) as f:
        labelme_dict = json.load(f)
        h, w = labelme_dict["imageHeight"], labelme_dict["imageWidth"]
        if len(labelme_dict["shapes"]) != 0:
            bbox = np.array(
                [
                    labelme_dict["shapes"][0]["points"][0][0] / w,
                    labelme_dict["shapes"][0]["points"][0][1] / h,
                    labelme_dict["shapes"][0]["points"][1][0] / w,
                    labelme_dict["shapes"][0]["points"][1][1] / h,
                ],
                dtype=dtype,
            )
        else:
            bbox = None
        return bbox


def bbox_alb_to_abs(bbox_alb, w, h):
    """Converts bounding boxes from albumentations format to absolute coordinates"""
    res = [bbox_alb[0] * w, bbox_alb[1] * h, bbox_alb[2] * w, bbox_alb[3] * h]
    return res
