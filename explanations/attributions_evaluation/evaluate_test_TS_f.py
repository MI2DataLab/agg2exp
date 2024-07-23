import os
import sys

sys.path.append("..")
import csv
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from captum.attr import KernelShap
from evaluation_utils import (
    ID2LABELS,
    evaluate_instance_batch_methods,
    general_preprocess_batch_methods,
    get_instance_iterator_batch_methods,
    inference_transform,
    predict_sliding_window,
    read_path,
)
from kernelshap_utils import (
    DEVICE,
    Wrapper,
    create_cubical_mask,
    output_segmentation,
    set_seeds,
)
from models.swin_unetr import SwinUNETR

print(DEVICE, torch.cuda.is_available())

import quantus
from quantus.helpers.model.pytorch_model import PyTorchModel

PyTorchModel.predict = predict_sliding_window

from quantus import FaithfulnessCorrelation  # , AvgSensitivity, EffectiveComplexity

FaithfulnessCorrelation.evaluate_instance = evaluate_instance_batch_methods

from quantus.metrics.base import Metric

Metric.general_preprocess = general_preprocess_batch_methods
Metric.get_instance_iterator = get_instance_iterator_batch_methods

from attribution_functions import (
    get_ig_attribution,
    get_pred_and_grad,
    get_smoothgrad_attribution,
)

with open("samples_test_TS_all_classes.txt", "r") as file:
    names_all_classes = [line.strip() for line in file.readlines()]

from data_loader import TotalChestSegmentatorDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="TSv2", choices=["TSv2", "b50"])
    parser.add_argument("--partition", type=int)
    parser.add_argument("--num_partitions", type=int)
    parser.add_argument("--save_folder", type=Path, default="data/b50")
    parser.add_argument("--attr_names", nargs="+", type=str)
    parser.add_argument("--calc_time", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    attr_names = args.attr_names
    assert all(
        [
            attr_name
            in [
                "vg_grad",
                "ig_grad",
                "sg_grad",
                "ks_attr_segmentations",
                "ks_attr_cubes",
            ]
            for attr_name in attr_names
        ]
    )

    partition = args.partition
    save_folder = Path(args.save_folder)

    if partition is None:
        partition = os.environ.get("SLURM_ARRAY_TASK_ID")
        if partition is not None:
            partition = int(partition)

    if partition is not None and args.num_partitions is None:
        raise ValueError("Need to specify num_partitions when specifying partition")

    set_seeds()

    if args.dataset == "TSv2":
        data_path = "data/TotalChestSegmentator/test/**"
        lung_artery_dataset = TotalChestSegmentatorDataset(
            data_path=data_path, mode="valid"
        )

        data = []
        for i, instance in enumerate(lung_artery_dataset):
            print(f"Instance {i}")
            path = lung_artery_dataset.labels[i]
            assert (
                path.split("/")[-2] == instance[1].split("/")[-2]
            ), f"{path} != {instance[1]}"
            inst = instance[0]
            for name in names_all_classes:
                if name in path:
                    inst["path"] = path
                    data.append(inst)
                    break
    elif args.dataset == "b50":
        data_path = "b50_paths.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Number of test data: {len(data)}")  # {TSv2: 32, b50: 55}

    model_path = Path("data/TotalSegmentator_SwinUNETR_full_v2.pth")
    model = SwinUNETR((96, 96, 96), 1, 17, feature_size=48, use_v2=True).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    targets_names = [
        "background",
        "aorta",
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "trachea",
        "heart",
        "ribs",
    ]
    targets = [key for key, value in ID2LABELS.items() if value in targets_names]

    faithfulness_estimator = FaithfulnessCorrelation(
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        subset_size=int(224**2),
        return_aggregate=False,
        normalise=True,
    )

    if partition is not None:
        data = data[partition : partition + 1]

    for inst in data:
        if args.dataset == "TSv2":
            data_name = inst["path"].split("/")[-2]
        elif args.dataset == "b50":
            data_name = inst.split("/")[-3]
        print(f"Processing {data_name}")

        attr_str = "__".join(attr_names)
        csv_file_path = (
            save_folder / f"faithfulness_estimates_{data_name}_{attr_str}.csv"
        )

        if args.calc_time:
            csv_time = save_folder / f"time_{data_name}_{attr_str}.csv"

        if not args.resume:
            with open(csv_file_path, "w") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")
                csv_writer.writerow(["path", "target", *attr_names])
            if args.calc_time:
                with open(csv_time, "w") as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=",")
                    csv_writer.writerow(["path", "target", *attr_names])

        if args.dataset == "TSv2":
            loaded_img = inst["image"].unsqueeze(0).to(DEVICE)
        elif args.dataset == "b50":
            image_path = read_path(inst)
            loaded_img = inference_transform(image_path).unsqueeze(0).to(DEVICE)
            os.remove(image_path)

        print(f"Loaded image shape: {loaded_img.shape}")

        if "ks_attr_segmentations" in attr_names or "ks_attr_cubes" in attr_names:
            cube_mask = create_cubical_mask(loaded_img, num_components=512).to(DEVICE)
            out_max = output_segmentation(model, loaded_img)
            wrapper = Wrapper(model, out_max)
            ks_wrapper = KernelShap(wrapper.wrapper_classes_intersection)

        for target in targets:
            target_calc = False
            if args.resume:
                with open(csv_file_path, "r") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    for row in csv_reader:
                        if row[1] == str(target):
                            target_calc = True
                            print(f"Skipping {data_name} {target}")
                            break
            if target_calc:
                continue

            print(f"Processing {data_name} {target}")

            attr_list = []
            if args.calc_time:
                time_list = []

            for attr_name in attr_names:
                path = save_folder / attr_name / f"{data_name}"
                path.mkdir(parents=True, exist_ok=True)
                if not (path / f"{target}.npy").exists():
                    print(f"Computing {attr_name}")
                    if args.calc_time:
                        start = time.time()

                    loaded_img_c = loaded_img.clone().detach()  # needed
                    if attr_name == "vg_grad":
                        _, attr = get_pred_and_grad(
                            loaded_img_c, model, class_for_saliency=[target]
                        )
                    elif attr_name == "ig_grad":
                        _, attr = get_ig_attribution(
                            loaded_img_c,
                            model,
                            class_for_saliency=[target],
                            use_not_perturbed_image_pred=True,
                        )
                    elif attr_name == "sg_grad":
                        _, attr = get_smoothgrad_attribution(
                            loaded_img_c,
                            model,
                            class_for_saliency=[target],
                            use_not_perturbed_image_pred=True,
                        )
                    elif attr_name == "ks_attr_segmentations":
                        attr = ks_wrapper.attribute(
                            loaded_img_c,
                            feature_mask=out_max,
                            perturbations_per_eval=1,
                            target=target,
                            n_samples=200,
                            show_progress=True,
                        )
                    elif attr_name == "ks_attr_cubes":
                        attr = ks_wrapper.attribute(
                            loaded_img_c,
                            feature_mask=cube_mask,
                            perturbations_per_eval=1,
                            target=target,
                            n_samples=1000,
                            show_progress=True,
                        )
                    else:
                        raise ValueError(f"Unknown attr_name: {attr_name}")

                    if args.calc_time:
                        end = time.time()
                        time_list.append(end - start)

                    attr = attr.squeeze(0)
                    np.save(path / f"{target}.npy", attr.cpu().numpy())

                else:
                    print(f"Loading {attr_name}")
                    attr = np.load(path / f"{target}.npy")
                    attr = torch.tensor(attr, device=DEVICE)

                if attr_name in ["ks_attr_segmentations", "ks_attr_cubes"]:
                    attr = attr.unsqueeze(0)
                attr_list.append(attr)

            if args.calc_time:
                with open(csv_time, "a") as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=",")
                    csv_writer.writerow([data_name, target, *time_list])

            faithfulness_estimate = faithfulness_estimator(
                model=model,
                x_batch=loaded_img.squeeze(0).cpu().numpy(),
                y_batch=np.array([target]),
                a_batch=[attr.squeeze(0).cpu().numpy() for attr in attr_list],
                channel_first=True,  # don't change the order of dimensions
                device=DEVICE,
            )
            print(
                f"Faithfulness estimate: {faithfulness_estimate}"
            )  # return something like [[0, 1]]

            with open(csv_file_path, "a") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")
                csv_writer.writerow([data_name, target, *faithfulness_estimate[0]])


if __name__ == "__main__":
    main()
