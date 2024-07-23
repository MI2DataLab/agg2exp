from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import CLI
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    SpatialPad,
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

JOINED_CLASSES = {
    "background": [0],
    "aorta": [1],
    "lung": [2, 3, 4, 5, 6],
    "trachea": [7],
    "heart": [8],
    "pulmonary_vein": [9],
    "thyroid_gland": [10],
    "ribs": [11],
    "vertebraes": [12],
    "autochthon": [13, 14],
    "sternum": [15],
    "costal_cartilages": [16],
}

USE_JOINED_CLASSES = False


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
    explanation_path,
    patient_id,
    preds_folder,
    segmentation_path: Optional[Path] = None,
    attribution_processing_function=ATTRIBUTION_PROCESSING_FUNCTION,
    num_classes=17,
):
    if segmentation_path is not None:
        load_transform = Compose(
            [
                LoadImage(reader="NibabelReader"),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                SpatialPad(spatial_size=(96, 96, 96)),
                ToTensor(),
            ]
        )

        y_post = AsDiscrete(to_onehot=num_classes)
        post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)

        dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        iou_metric = MeanIoU(
            include_background=True, reduction="mean", get_not_nans=False
        )

        mask = load_transform(segmentation_path)
    pred = np.load((preds_folder / patient_id) / "pred.npy")
    explanation = np.load(explanation_path)
    segmentation_mask = pred.argmax(axis=1)
    explanation_metrics = {}
    if segmentation_path is not None:
        dice, iou = get_iou_and_dice(
            pred, mask, dice_metric, iou_metric, y_post, post_pred
        )
    for class_id in range(num_classes):
        cur_explanation = explanation[class_id]
        if USE_JOINED_CLASSES:
            for joined_class_name, joined_class_ids in JOINED_CLASSES.items():
                explanation_mass = get_joined_explanation_mass_inside_segmentation_mask(
                    cur_explanation,
                    segmentation_mask,
                    joined_class_ids,
                    attribution_processing_function,
                )
                explanation_mean = get_joined_explanation_mean_inside_segmentation_mask(
                    cur_explanation,
                    segmentation_mask,
                    joined_class_ids,
                    attribution_processing_function,
                )
                explanation_metrics[
                    f"{ID2LABELS[class_id]}_explanation_in_{joined_class_name}_mean"
                ] = explanation_mean
                explanation_metrics[
                    f"{ID2LABELS[class_id]}_explanation_in_{joined_class_name}_mass"
                ] = explanation_mass
        else:
            for i in range(num_classes):
                explanation_mass = get_explanation_mass_inside_segmentation_mask(
                    cur_explanation,
                    segmentation_mask,
                    i,
                    attribution_processing_function,
                )
                explanation_metrics[
                    f"{ID2LABELS[class_id]}_explanation_in_{ID2LABELS[i]}_mass"
                ] = explanation_mass
                explanation_mean = get_explanation_mean_inside_segmentation_mask(
                    cur_explanation,
                    segmentation_mask,
                    i,
                    attribution_processing_function,
                )
                explanation_metrics[
                    f"{ID2LABELS[class_id]}_explanation_in_{ID2LABELS[i]}_mean"
                ] = explanation_mean
        explanation_var = np.var(cur_explanation)
        explanation_mean = np.mean(cur_explanation)
        explanation_metrics.update(
            {
                f"{ID2LABELS[class_id]}_explanation_var": explanation_var,
                f"{ID2LABELS[class_id]}_explanation_mean": explanation_mean,
            }
        )
    output_data = {
        **{
            "patient": patient_id,
        },
        **explanation_metrics,
    }
    if segmentation_path is not None:
        output_data.update(
            **{
                f"{ID2LABELS[class_id]}_iou": iou[0, class_id].item()
                for class_id in range(num_classes)
            },
            **{
                f"{ID2LABELS[class_id]}_dice": dice[0, class_id].item()
                for class_id in range(num_classes)
            },
        )
    return output_data


def main(
    explanations_folder: Path = Path("data/ig_explanations_and_pred"),
    preds_folder: Path = Path("data/ig_explanations_and_pred"),
    save_path: Path = Path("explanations_aggregations_ig.csv"),
    glob_str: str = "**/**/**/grad.npy",
    num_workers: int = 8,
    seperate_neg_and_pos: bool = False,
    ground_truth_folder: Optional[Path] = None,
    test_run: bool = False,
    is_train_gt: bool = False,
):
    """
    Aggregate explanations for predictions and compute metrics.

    Args:
        explanations_folder (Path): Path to the folder containing the explanations.
        preds_folder (Path): Path to the folder containing the predictions.
        save_path (Path): Path to save the resulting metrics CSV file.
        glob_str (str): Glob string to match the explanation files.
        num_workers (int): Number of worker processes to use for parallel processing.
        seperate_neg_and_pos (bool): Whether to compute metrics for negative and positive attributions seperately.
        ground_truth_folder (Path): Path to the folder containing the ground truth segmentations.
        test_run (bool): Flag to run in a loop instead of using pqdm.

    Returns:
        None
    """
    explanations = sorted(explanations_folder.glob(glob_str))
    patient_id_depth = len(glob_str.split("/")) - 1
    patient_ids = [
        "/".join(str(explanation_path.parent).split("/")[-patient_id_depth:])
        for explanation_path in explanations
    ]
    if ground_truth_folder is not None:
        segmentations_paths = [
            (ground_truth_folder / patient_id / "segmentations.nii.gz")
            if is_train_gt
            else (
                ground_truth_folder / patient_id / f"{patient_id}_segmentations.nii.gz"
            )
            for patient_id in patient_ids
        ]
    else:
        segmentations_paths = [None] * len(explanations)
    if seperate_neg_and_pos:
        kwargs_neg = [
            {
                "explanation_path": explanation,
                "patient_id": patient_id,
                "preds_folder": preds_folder,
                "attribution_processing_function": neg_processing_function,
                "segmentation_path": segmentation_path,
            }
            for explanation, patient_id, segmentation_path in zip(
                explanations, patient_ids, segmentations_paths
            )
        ]
        kwargs_pos = [
            {
                "explanation_path": explanation,
                "patient_id": patient_id,
                "preds_folder": preds_folder,
                "attribution_processing_function": pos_processing_function,
                "segmentation_path": segmentation_path,
            }
            for explanation, patient_id, segmentation_path in zip(
                explanations, patient_ids, segmentations_paths
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
                "explanation_path": explanation,
                "patient_id": patient_id,
                "preds_folder": preds_folder,
                "attribution_processing_function": ATTRIBUTION_PROCESSING_FUNCTION,
                "segmentation_path": segmentation_path,
            }
            for explanation, patient_id, segmentation_path in zip(
                explanations, patient_ids, segmentations_paths
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
        try:
            output_data = pd.DataFrame(output_data)
        except Exception as e:
            with open(save_path.parent / "error.pkl", "wb") as f:
                import pickle

                pickle.dump(output_data, f)
            raise e
    output_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    CLI(main)
