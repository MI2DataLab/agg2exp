import sys

sys.path.append("../model")

import random
import shutil
import tempfile
from pathlib import Path

import monai.transforms as transforms
import numpy as np
import torch
from models.swin_unetr import SwinUNETR
from monai.inferers import sliding_window_inference

MODEL_PATH = "path/to/model_checkpoint/SwinUNETR_full_v2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B50_FOLDER = Path("B50_folder/")
B50_PATHS = sorted([str(x) for x in B50_FOLDER.glob("*/*/*")])


class Wrapper:
    def __init__(self, model, out_max):
        self.model = model
        self.out_max = out_max

    def wrapper_classes(self, img):
        # simply summing
        with torch.no_grad():
            output = inference(self.model, img)
        return output.sum(axis=(2, 3, 4))

    def wrapper_classes_intersection(self, img):
        # Creates binary matrix with 1 for original argmax class for each voxel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        with torch.no_grad():
            output = inference(self.model, img)
            selected_inds = torch.zeros_like(output[0:1]).scatter_(1, self.out_max, 1)
        return (output * selected_inds).sum(axis=(2, 3, 4))


def inference(model, loaded_img):
    with torch.no_grad():
        output = sliding_window_inference(
            loaded_img, (96, 96, 96), 1, model, sw_device=DEVICE
        )  # , device=torch.device("cpu"))
        return output


def output_segmentation(model, loaded_img):
    with torch.no_grad():
        output = inference(model, loaded_img)
        output = torch.argmax(output, dim=1, keepdim=True)
        return output


def load_model():
    model = SwinUNETR(
        in_channels=1,
        out_channels=17,
        img_size=(96, 96, 96),
        feature_size=48,
        use_checkpoint=False,
        use_v2=True,
    )
    model = model.to(DEVICE)
    loaded_cpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(loaded_cpt)
    model.eval()

    test_transform = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            transforms.ToTensor(),
        ]
    )
    return model, test_transform


def read_path(path):
    tmp_scan_path = None
    if not str(path).endswith("nii.gz"):
        # Generate a unique temporary file name to make it parallel-safe
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
            tmp_scan_path = tmp_file.name

        # This is needed so that nibabel recognizes the file by extension
        shutil.copyfile(path, tmp_scan_path)
        path = tmp_scan_path

    return path


def create_cubical_mask(image: torch.Tensor, num_components: int) -> torch.Tensor:
    if len(image.shape) != 5 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError("Input tensor must have shape (1, 1, x, y, z)")

    _, _, x, y, z = image.shape

    # Calculate the size of the cube side based on the smallest dimension and num_components
    cube_size = min(x, y, z) // int(pow(num_components, 1 / 3))

    if cube_size == 0:
        raise ValueError("Too many components for the given image dimensions")

    mask = torch.zeros((1, 1, x, y, z), dtype=torch.long)

    unique_id = 0
    for i in range(0, x, cube_size):
        for j in range(0, y, cube_size):
            for k in range(0, z, cube_size):
                mask[0, 0, i : i + cube_size, j : j + cube_size, k : k + cube_size] = (
                    unique_id % num_components
                )
                unique_id += 1

    return mask


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
