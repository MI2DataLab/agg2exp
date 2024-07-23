import glob
import os

import torch
import torch.nn as nn
from models.swin_unetr import SwinUNETR
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms import (
    ActivationsD,
    AsDiscreteD,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImageD,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.cuda.amp import autocast

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

DATA_PATH = ""
MODEL_PATH = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    device_name = torch.cuda.get_device_name(0)

inference_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
        ),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
    ]
)

post_transform = Compose(
    [
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=inference_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        ActivationsD(keys="pred", softmax=True),
        AsDiscreteD(keys="pred", argmax=True),
        SaveImageD(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir="inference_STOIC",
            output_postfix="_seg",
            resample=False,
        ),
    ]
)

model = SwinUNETR(
    in_channels=1,
    out_channels=17,
    img_size=(96, 96, 96),
    feature_size=48,
    use_checkpoint=True,
    use_v2=False,
)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

inferer = SlidingWindowInferer(
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    sw_device="cuda",
    device="cuda",
    overlap=0.25,
    mode="gaussian",
    padding_mode="replicate",
)

data_dir = DATA_PATH
test_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
test_data = [{"image": image} for image in test_images]
test_dataset = Dataset(data=test_data, transform=inference_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

with torch.no_grad():
    for test_data in test_loader:
        print(test_data)
        test_inputs = test_data["image"].to(device)
        with autocast():
            test_data["pred"] = sliding_window_inference(
                test_inputs, (96, 96, 96), 1, model, sw_device="cuda", device="cuda"
            )
        test_data = [post_transform(i) for i in decollate_batch(test_data)]
        torch.cuda.empty_cache()
