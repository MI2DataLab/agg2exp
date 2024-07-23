import sys

sys.path.append("../model")

import shutil
from pathlib import Path

import monai.transforms as transforms
import torch
from models.swin_unetr import SwinUNETR

DEFAULT_MODEL_PATH = "TotalSegmentator_SwinUNETR_full_v2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B50_FOLDER = Path("b50/original_image/ct_1.0/nifti_1.0/")
B50_PATHS = sorted([str(x) for x in B50_FOLDER.glob("*/*/*")])
TSV2_PATIENTS_WITH_ALL_CLASSES = [
    "s0029",
    "s0040",
    "s0230",
    "s0423",
    "s0440",
    "s0459",
    "s0470",
    "s0482",
    "s0499",
    "s0543",
    "s0561",
    "s0667",
    "s0687",
    "s0735",
    "s0753",
    "s0802",
    "s0829",
    "s0923",
    "s0933",
    "s0994",
    "s1094",
    "s1119",
    "s1152",
    "s1174",
    "s1176",
    "s1240",
    "s1248",
    "s1249",
    "s1276",
    "s1323",
    "s1377",
    "s1420",
]
TSV2_FOLDER = Path("data/Totalsegmentator_dataset/test/")
TSV2_PATHS = [
    str(TSV2_FOLDER / f"{x}/{x}_ct.nii.gz") for x in TSV2_PATIENTS_WITH_ALL_CLASSES
]


def get_transform():
    return transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="RAS"),
            transforms.ScaleIntensityRange(
                a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            transforms.SpatialPad(spatial_size=(96, 96, 96)),
            transforms.ToTensor(),
        ]
    )


def load_model(
    model_path: str = DEFAULT_MODEL_PATH, num_classes: int = 15, use_v2: bool = False
):
    model = SwinUNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=48,
        use_checkpoint=False,
        use_v2=use_v2,
    )
    model = model.to(DEVICE)
    loaded_cpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(loaded_cpt)
    model.eval()

    test_transform = get_transform()
    return model, test_transform


def read_path(path, tmp_file_name="tmp_scan.nii.gz"):
    if not path.endswith("nii.gz"):
        tmp_scan_path = f"images/{tmp_file_name}"
        # this is needed so that nibabel recognizes the file by extension
        shutil.copyfile(path, tmp_scan_path)
        path = tmp_scan_path
    return path
