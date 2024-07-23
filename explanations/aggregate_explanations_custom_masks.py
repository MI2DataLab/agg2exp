from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from jsonargparse import CLI
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    Spacing,
    ToTensor,
)
from pqdm.processes import pqdm
from tqdm import tqdm

ATTRIBUTION_PROCESSING_FUNCTION = np.abs
ID2LABELS = {
    0: "background",
    1: "aorta",
    2: "lung_upper_lobe_left",
    3: "lung_lower_lobe_left",
    4: "lung_upper_lobe_right",
    5: "lung_middle_lobe_right",
    6: "lung_lower_lobe_right",
    7: "trachea",
    8: "heart",
    9: "pulmonary_vein",
    10: "thyroid_gland",
    11: "ribs",
    12: "vertebraes",
    13: "autochthon_left",
    14: "autochthon_right",
    15: "sternum",
    16: "costal_cartilages",
}

ADDITIONALY_CALCULATE_JOINED = False


def pos_processing_function(explanation):
    return np.clip(explanation, 0, None)


def neg_processing_function(explanation):
    return pos_processing_function(-explanation)


def get_explanation_mass_inside_segmentation_mask(
    explanation, segmentation_mask, class_id, attribution_processing_function
):
    explanation = attribution_processing_function(explanation)
    return np.sum(explanation * (segmentation_mask == class_id)) / np.sum(explanation)


def get_explanation_mean_inside_segmentation_mask(
    explanation, segmentation_mask, class_id, attribution_processing_function
):
    explanation = attribution_processing_function(explanation)
    segmentation_mask = segmentation_mask == class_id
    return np.sum(explanation * segmentation_mask) / np.sum(segmentation_mask)


def get_joined_explanation_mass_inside_segmentation_mask(
    explanation, segmentation_mask, class_ids, attribution_processing_function
):
    explanation = attribution_processing_function(explanation)
    return np.sum(
        explanation * np.isin(segmentation_mask, class_ids).astype(float)
    ) / np.sum(explanation)


def get_joined_explanation_mean_inside_segmentation_mask(
    explanation, segmentation_mask, class_ids, attribution_processing_function
):
    explanation = attribution_processing_function(explanation)
    segmentation_mask = np.isin(segmentation_mask, class_ids).astype(float)
    return np.sum(explanation * segmentation_mask).astype(float) / np.sum(
        segmentation_mask
    )


def get_iou_and_dice(pred, label, dice_metric, iou_metric, y_post, post_pred):
    dice_metric.reset()
    iou_metric.reset()
    y = y_post(label).unsqueeze(0)
    y_pred = post_pred(pred[0]).unsqueeze(0)
    return dice_metric(y_pred, y), iou_metric(y_pred, y)


def process_explanation(
    explanation_folder,
    patient_id,
    mask_path,
    attribution_processing_function=ATTRIBUTION_PROCESSING_FUNCTION,
    num_classes=17,
    custom_masks_labels: Optional[List[str]] = None,
    ids_of_interest: Optional[List[int]] = None,
    is_mask_logits: bool = False,
):
    load_transform = Compose(
        [
            LoadImage(reader="NibabelReader"),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(
                pixdim=(1.5, 1.5, 1.5), mode="bilinear" if is_mask_logits else "nearest"
            ),
            ToTensor(),
        ]
    )
    pred = load_transform(mask_path)
    if pred.shape[0] == 1:
        segmentation_mask = pred.numpy()
    else:
        segmentation_mask = pred.argmax(axis=0, keepdim=True).numpy()
    if ids_of_interest is not None:
        mask_indices = np.isin(segmentation_mask, ids_of_interest)
        segmentation_mask = np.where(mask_indices, segmentation_mask, 0)
        for i, id_of_interest in enumerate(ids_of_interest, start=1):
            segmentation_mask = np.where(
                segmentation_mask == id_of_interest, i, segmentation_mask
            )

    num_mask_classes = len(np.unique(segmentation_mask))
    explanation = np.load((explanation_folder / patient_id) / "grad.npy")
    if custom_masks_labels is None:
        if ids_of_interest is None:
            custom_masks_labels = [
                f"custom_mask_{i}" for i in range(1, num_mask_classes)
            ]
        else:
            custom_masks_labels = [f"custom_mask_{i}" for i in ids_of_interest]
    explanation_metrics = {}
    for class_id in range(num_classes):
        cur_explanation = explanation[class_id]
        if ADDITIONALY_CALCULATE_JOINED:
            explanation_mass = get_joined_explanation_mass_inside_segmentation_mask(
                cur_explanation,
                segmentation_mask,
                list(range(1, len(custom_masks_labels) + 1)),
                attribution_processing_function,
            )
            explanation_mean = get_joined_explanation_mean_inside_segmentation_mask(
                cur_explanation,
                segmentation_mask,
                list(range(1, len(custom_masks_labels) + 1)),
                attribution_processing_function,
            )
            explanation_metrics[
                f"{ID2LABELS[class_id]}_explanation_in_joined_custom_mask_mean"
            ] = explanation_mean
            explanation_metrics[
                f"{ID2LABELS[class_id]}_explanation_in_joined_custom_mask_mass"
            ] = explanation_mass

        for i, class_name in enumerate(custom_masks_labels, start=1):
            explanation_mass = get_explanation_mass_inside_segmentation_mask(
                cur_explanation,
                segmentation_mask,
                i,
                attribution_processing_function,
            )
            explanation_metrics[
                f"{ID2LABELS[class_id]}_explanation_in_{class_name}_mass"
            ] = explanation_mass
            explanation_mean = get_explanation_mean_inside_segmentation_mask(
                cur_explanation,
                segmentation_mask,
                i,
                attribution_processing_function,
            )
            explanation_metrics[
                f"{ID2LABELS[class_id]}_explanation_in_{class_name}_mean"
            ] = explanation_mean
    output_data = {
        **{
            "patient": patient_id,
        },
        **explanation_metrics,
    }
    return output_data


def main(
    explanations_folder: Path = Path("data/ig_explanations_and_pred"),
    custom_masks_folder: Path = Path("data/custom_mask_folder"),
    custom_masks_labels: Optional[List[str]] = None,
    save_path: Path = Path("data/explanations_aggregations_ig_custom_mask.csv"),
    glob_str: str = "**/**/*.nii.gz",
    num_workers: int = 4,
    seperate_neg_and_pos: bool = False,
    ids_of_interest: Optional[List[int]] = None,
    are_masks_logits: bool = False,
    calculate_joined: bool = False,
    test_run: bool = False,
):
    """
    Aggregate explanations for predictions and compute metrics.

    Args:
        explanations_folder: Path to folder with explanations.
        custom_masks_folder: Path to folder with custom masks.
        custom_masks_labels: List of custom masks labels names.
        save_path: Path to save aggregated explanations.
        glob_str: Glob string to find custom masks.
        num_workers: Number of workers for multiprocessing.
        seperate_neg_and_pos: Whether to compute metrics for negative and positive explanations separately.
        ids_of_interest: List of ids of interest.
        are_masks_logits: Whether masks are logits.
        test_run: Whether to run in test mode (only one patient).

    Returns:
        None
    """
    global ADDITIONALY_CALCULATE_JOINED
    ADDITIONALY_CALCULATE_JOINED = calculate_joined
    custom_masks_paths = sorted(custom_masks_folder.glob(glob_str))
    patient_id_depth = len(glob_str.split("/"))
    patient_ids = [
        "/".join(str(custom_mask_path).split("/")[-patient_id_depth:]).replace(
            ".nii.gz", ""
        )
        for custom_mask_path in custom_masks_paths
    ]
    if seperate_neg_and_pos:
        kwargs_neg = [
            {
                "explanation_folder": explanations_folder,
                "patient_id": patient_id,
                "mask_path": mask_path,
                "attribution_processing_function": neg_processing_function,
                "custom_masks_labels": custom_masks_labels,
                "ids_of_interest": ids_of_interest,
                "is_mask_logits": are_masks_logits,
            }
            for mask_path, patient_id in zip(
                custom_masks_paths,
                patient_ids,
            )
        ]
        kwargs_pos = [
            {
                "explanation_folder": explanations_folder,
                "patient_id": patient_id,
                "mask_path": mask_path,
                "attribution_processing_function": pos_processing_function,
                "custom_masks_labels": custom_masks_labels,
                "ids_of_interest": ids_of_interest,
                "is_mask_logits": are_masks_logits,
            }
            for mask_path, patient_id in zip(
                custom_masks_paths,
                patient_ids,
            )
        ]
        if test_run:
            output_data_neg = []
            output_data_pos = []
            for kwargs in tqdm(kwargs_neg, desc="Processing negative explanations"):
                output_data_neg.append(process_explanation(**kwargs))
            for kwargs in tqdm(kwargs_pos, desc="Processing positive explanations"):
                output_data_pos.append(process_explanation(**kwargs))
        else:
            output_data_neg = pqdm(
                kwargs_neg,
                process_explanation,
                n_jobs=num_workers,
                argument_type="kwargs",
                desc="Processing negative explanations",
            )
            output_data_pos = pqdm(
                kwargs_pos,
                process_explanation,
                n_jobs=num_workers,
                argument_type="kwargs",
                desc="Processing positive explanations",
            )
        output_data_neg = pd.DataFrame(output_data_neg)
        output_data_pos = pd.DataFrame(output_data_pos)
        output_data = output_data_neg.merge(
            output_data_pos, on="patient", suffixes=("_neg", "_pos")
        )

    else:
        kwargs = [
            {
                "explanation_folder": explanations_folder,
                "patient_id": patient_id,
                "mask_path": mask_path,
                "attribution_processing_function": ATTRIBUTION_PROCESSING_FUNCTION,
                "custom_masks_labels": custom_masks_labels,
                "ids_of_interest": ids_of_interest,
                "is_mask_logits": are_masks_logits,
            }
            for mask_path, patient_id in zip(
                custom_masks_paths,
                patient_ids,
            )
        ]
        if test_run:
            output_data = []
            for kwargs in tqdm(kwargs, desc="Processing explanations"):
                output_data.append(process_explanation(**kwargs))
        else:
            output_data = pqdm(
                kwargs,
                process_explanation,
                n_jobs=num_workers,
                argument_type="kwargs",
                desc="Processing explanations",
            )
        output_data = pd.DataFrame(output_data)
    output_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    CLI(main)
