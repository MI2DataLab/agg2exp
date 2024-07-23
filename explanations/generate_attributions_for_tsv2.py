from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from attribution_functions import (
    get_ig_attribution,
    get_pred_and_grad,
    get_smoothgrad_attribution,
)
from generation_utils import (
    DEFAULT_MODEL_PATH,
    DEVICE,
    TSV2_PATHS,
    load_model,
)
from tqdm import tqdm

ATTR_FUNCTIONS = {
    "vg": get_pred_and_grad,
    "ig": get_ig_attribution,
    "sg": get_smoothgrad_attribution,
}

BASELINES = {
    "zeros": torch.zeros_like,
    "ones": torch.ones_like,
    "mean": lambda x: 0.2302 * torch.ones_like(x),
    "random": torch.rand_like,
    "gaussian": lambda x: 0.2377 * torch.randn_like(x) + 0.2302,
}


def process_b50(
    model,
    test_transform,
    save_folder: Path,
    save_pred: bool = False,
    attr_function=get_pred_and_grad,
    skip_already_processed=False,
    baseline: Callable[[torch.Tensor], torch.Tensor] = torch.zeros_like,
    sw_batch_size: int = 1,
    use_not_perturbed_image_pred: bool = False,
    num_classes: int = 17,
    images_paths=TSV2_PATHS,
):
    for path in tqdm(images_paths):
        patient_path = path.split("/")[-2]
        save_path = save_folder / patient_path
        save_path.mkdir(parents=True, exist_ok=True)
        if skip_already_processed and (save_path / "grad.npy").exists():
            continue
        loaded_img = test_transform(path).to(DEVICE).unsqueeze(0)
        pred, grad = attr_function(
            loaded_img,
            model,
            class_for_saliency=list(range(num_classes)),
            sw_batch_size=sw_batch_size,
            baseline_function=baseline,
            use_not_perturbed_image_pred=use_not_perturbed_image_pred,
        )
        if pred is not None:
            pred = pred.detach().cpu().numpy()
        grad = grad.detach().cpu().numpy()
        with open(save_path / "grad.npy", "wb") as f:
            np.save(f, grad)
        if save_pred:
            with open(save_path / "pred.npy", "wb") as f:
                np.save(f, pred)


def main(args):
    model, test_transform = load_model(
        model_path=args.model_path, num_classes=args.num_classes, use_v2=args.use_v2
    )
    if args.images_file is not None:
        with open(args.images_file) as f:
            images_paths = [x.strip() for x in f.readlines()][
                args.start_idx : args.end_idx
            ]
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    process_b50(
        model,
        test_transform,
        save_folder,
        save_pred=args.save_pred,
        attr_function=ATTR_FUNCTIONS[args.attribution_type],
        skip_already_processed=args.skip_already_processed,
        baseline=BASELINES[args.baseline],
        sw_batch_size=args.sw_batch_size,
        use_not_perturbed_image_pred=args.use_not_perturbed_image_pred,
        num_classes=args.num_classes,
        images_paths=images_paths,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-folder", type=str, required=True)
    parser.add_argument(
        "-a",
        "--attribution-type",
        type=str,
        default="vg",
        choices=ATTR_FUNCTIONS.keys(),
    )
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--skip-already-processed", action="store_true", default=False)
    parser.add_argument(
        "--baseline", type=str, default="zeros", choices=BASELINES.keys()
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--use-not-perturbed-image-pred", action="store_true")
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--use-v2", action="store_true")
    parser.add_argument("--images-file", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()
    main(args)
