import os
import albumentations as alb
import numpy as np

from src import util


def generate_samples(pipeline, num_samples, input_dir, output_dir):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    transform_fn = alb.Compose(pipeline, bbox_params={"format": "albumentations", "label_fields": ["class_labels"]})

    input_fnames_noext = sorted([fname[:-4] for fname in os.listdir(input_dir) if fname.endswith(".png")])
    for ii, input_fname_noext in enumerate(input_fnames_noext):
        input_fpath_noext = f"{input_dir}/{input_fname_noext}"

        img_cam = util.load_image_from_file(f"{input_fpath_noext}.png")
        bbox_alb = util.load_labelme_bbox(f"{input_fpath_noext}.json", np.float32)

        for jj in range(num_samples):
            output_fname_noext = f"{input_fname_noext}-{jj:04d}"

            # Generate sample from augmentation pipeline.
            if bbox_alb is not None:
                sample = transform_fn(image=img_cam, bboxes=[bbox_alb], class_labels=["face"])

                # Some transformations lead to an out-of-view bounding box. Beware.
                if len(sample["bboxes"]) > 0:
                    label = np.asarray([*sample["bboxes"][0], 1.0])
                else:
                    label = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                sample = transform_fn(image=img_cam, bboxes=[], class_labels=[])
                label = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])

            # Write image and label to file. We reshape the numpy array to produce
            # a single row output.
            np.savetxt(f"{output_dir}/{output_fname_noext}.csv", label.reshape((1, 5)), delimiter=",")
            util.save_image(f"{output_dir}/{output_fname_noext}.png", sample["image"])

        # Print progress.
        print("\r" * 100 + f"{ii+1:03d}/{len(input_fnames_noext)}", end="")
