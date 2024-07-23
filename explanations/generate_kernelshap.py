import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from captum.attr import KernelShap
from kernelshap_utils import (
    B50_PATHS,
    DEVICE,
    Wrapper,
    create_cubical_mask,
    load_model,
    output_segmentation,
    read_path,
    set_seeds,
)
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["b50"], default="b50")

    parser.add_argument("--partition", type=int)
    parser.add_argument("--num_partitions", type=int)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--predictions_save_folder", type=Path, default=None)
    parser.add_argument("--skip_already_processed", action="store_true")

    parser.add_argument("--num_components", type=int, default=512)
    parser.add_argument("--targets", nargs="+", type=int, required=True)

    arguments = parser.parse_args()
    partition = arguments.partition
    save_folder = Path(arguments.save_folder)

    if partition is None:
        partition = os.environ.get("SLURM_ARRAY_TASK_ID")
        if partition is not None:
            partition = int(partition)

    if partition is not None and arguments.num_partitions is None:
        raise ValueError("Need to specify num_partitions when specifying partition")

    if arguments.dataset == "b50":
        dataset_scans = B50_PATHS
    else:
        raise ValueError(f"dataset {arguments.dataset} not supported")

    if partition is not None:
        dataset_scans = dataset_scans[partition : partition + 1]

    print(f"Device: {DEVICE}")

    set_seeds()

    model, test_transform = load_model()

    for path in tqdm(dataset_scans):
        patient_path = "/".join(path.split("/")[-3:])
        save_path = save_folder / patient_path
        save_path.mkdir(parents=True, exist_ok=True)

        image_path = read_path(path)
        loaded_img = test_transform(image_path).to(DEVICE).unsqueeze(0)
        os.remove(image_path)
        cube_mask = create_cubical_mask(
            loaded_img, num_components=arguments.num_components
        ).to(DEVICE)
        out_max = output_segmentation(model, loaded_img).to(DEVICE)

        wrapper = Wrapper(model, out_max)
        ks_wrapper = KernelShap(wrapper.wrapper_classes_intersection)

        if arguments.predictions_save_folder is not None:
            predictions_save_path = arguments.predictions_save_folder / patient_path
            if not (predictions_save_path / "output_segmentation.npy").exists():
                predictions_save_path.mkdir(parents=True, exist_ok=True)
                np.save(
                    predictions_save_path / "output_segmentation.npy",
                    out_max.cpu().numpy(),
                )

        for target in arguments.targets:
            if (
                not arguments.skip_already_processed
                or not (save_path / f"kernelshap_cubes_{target}.npy").exists()
            ):
                ks_attr_cubes = ks_wrapper.attribute(
                    loaded_img,
                    feature_mask=cube_mask,
                    perturbations_per_eval=1,
                    target=target,
                    n_samples=1000,
                    show_progress=True,
                )
                np.save(
                    save_path / f"kernelshap_cubes_{target}.npy",
                    ks_attr_cubes.cpu().numpy(),
                )

            if (
                not arguments.skip_already_processed
                or not (save_path / f"kernelshap_segmentations_{target}.npy").exists()
            ):
                ks_attr_segmentations = ks_wrapper.attribute(
                    loaded_img,
                    feature_mask=out_max,
                    perturbations_per_eval=1,
                    target=target,
                    n_samples=200,
                    show_progress=True,
                )
                np.save(
                    save_path / f"kernelshap_segmentations_{target}.npy",
                    ks_attr_segmentations.cpu().numpy(),
                )


if __name__ == "__main__":
    main()
