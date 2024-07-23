from pathlib import Path

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
from generation_utils import B50_PATHS, get_transform, read_path
from matplotlib.colors import ListedColormap

EXPLANATIONS_FOLER = Path("data/")
EXPLANATION_PATHS = {
    "VG": EXPLANATIONS_FOLER / "b50_vg_explanations_and_pred",
    "IG": EXPLANATIONS_FOLER / "b50_ig_explanations_and_pred",
    "SG": EXPLANATIONS_FOLER / "b50_sg_explanations_and_pred",
    "kernelSHAP cubes": EXPLANATIONS_FOLER / "b50_kernelshap_explanations",
    "kernelSHAP seg": EXPLANATIONS_FOLER / "b50_kernelshap_explanations",
}

LABELS = {
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


def load_explanation(
    explanation,
    selected_label=1,
    binary_mask: bool = False,
    min_val: float = 0.01,
    max_val: float = 0.95,
    use_abs: bool = False,
):
    explanation = explanation[selected_label].squeeze()

    if use_abs:
        explanation = np.abs(explanation)
    grad_min = np.min(explanation) * max_val
    grad_max = np.max(explanation) * max_val
    max_x, max_y, max_z = np.unravel_index(
        np.argmax(explanation, axis=None), explanation.shape
    )
    if binary_mask:
        _explanation = np.where(
            explanation <= 0,
            0,
            1,
        )
        explanation = np.where(
            (np.abs(explanation) <= (min_val * np.max(np.abs(explanation))))
            | (np.abs(explanation) >= (max_val * np.max(np.abs(explanation)))),
            0.5,
            _explanation,
        )
    else:
        explanation = np.where(
            np.abs(explanation) <= (min_val * np.max(np.abs(explanation))),
            -1,
            (explanation - grad_min) / (grad_max - grad_min),
        )
    return explanation, (max_x, max_y, max_z)


def load_image(path, _transform):
    return _transform(read_path(path)).squeeze().numpy()


def load_prediction(path):
    return np.load(path)[0]


def get_cmap(binary=False, use_abs=False):
    if binary:
        cmap = ListedColormap(["blue", "#FFFFFF00", "red"])
    elif use_abs:
        cmap = plt.cm.Reds
        cmap.set_under("#FFFFFF00")
    else:
        cmap = plt.cm.seismic
        cmap.set_under("#FFFFFF00")
    return cmap


def get_loading_function():
    return get_transform()


def generate_example_explanations(
    current_patient_id,
    current_explanations,
    label,
    current_prediction,
    min_val,
    max_val,
    output_path,
    explanation_type="grad",
    select_slices_with="explanation",
    zero_background=False,
):
    cmap = get_cmap(False, use_abs=False)

    image_path = B50_PATHS[current_patient_id]
    patient_path = "/".join(image_path.split("/")[-3:])

    raw_prediction = load_prediction(current_prediction / patient_path / "pred.npy")
    raw_prediction = np.argmax(raw_prediction, axis=0)
    loaded_img = load_image(image_path, get_loading_function())
    explanation_type_list = current_explanations.split(",")

    fig, ax = plt.subplots(
        len(explanation_type_list) // 2,
        6,
        figsize=(30, 5 * (len(explanation_type_list) // 2)),
    )
    fig.tight_layout()
    ax = ax.flatten()
    i = 0

    for explanation_type in explanation_type_list:
        if explanation_type == "kernelSHAP cubes":
            raw_explanation = np.concatenate(
                [np.zeros((1, 1) + loaded_img.shape)]
                + [
                    np.load(
                        EXPLANATION_PATHS[explanation_type]
                        / patient_path
                        / f"kernelshap_cubes_{i}.npy"
                    )
                    for i in range(1, 17)
                ],
                axis=0,
            )
        elif explanation_type == "kernelSHAP seg":
            raw_explanation = np.concatenate(
                [np.zeros((1, 1) + loaded_img.shape)]
                + [
                    np.load(
                        EXPLANATION_PATHS[explanation_type]
                        / patient_path
                        / f"kernelshap_segmentations_{i}.npy"
                    )
                    for i in range(1, 17)
                ],
                axis=0,
            )
        else:
            raw_explanation = explanation = np.load(
                (EXPLANATION_PATHS[explanation_type] / patient_path / "grad.npy")
            )
        prediction_cmap = ListedColormap(["#00000000", "yellow"])

        explanation, (max_x, max_y, max_z) = load_explanation(
            raw_explanation,
            label,
            binary_mask=False,
            min_val=min_val,
            max_val=max_val,
        )
        pred = raw_prediction == label
        if select_slices_with == "prediction":
            max_x = np.argmax(pred.sum(axis=(1, 2)))
            max_y = np.argmax(pred.sum(axis=(0, 2)))
            max_z = np.argmax(pred.sum(axis=(0, 1)))
        if zero_background:
            explanation = np.where(raw_prediction == 0, -1, explanation)
        # Y-Z projection with rotation
        ax[i].imshow(np.rot90(loaded_img[max_x], k=1), cmap="gray")
        ax[i].imshow(
            np.rot90(explanation[max_x], k=1),
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        ax[i].imshow(
            np.rot90(pred[max_x], k=1),
            cmap=prediction_cmap,
            alpha=0.3,
        )
        ax[i].set_title(
            f"Y-Z slice {explanation_type}",
            fontdict={"fontsize": 23},
        )
        ax[i].axis("off")
        ax[i].set_aspect("auto")
        i += 1

        # X-Z projection with rotation
        ax[i].imshow(np.rot90(loaded_img[:, max_y], k=1), cmap="gray")
        ax[i].imshow(
            np.rot90(explanation[:, max_y], k=1),
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        ax[i].imshow(
            np.rot90(pred[:, max_y], k=1),
            cmap=prediction_cmap,
            alpha=0.3,
        )
        ax[i].set_title(
            f"X-Z slice {explanation_type}",
            fontdict={"fontsize": 23},
        )
        ax[i].axis("off")
        ax[i].set_aspect("auto")
        i += 1

        # X-Y projection with rotation
        ax[i].imshow(np.rot90(loaded_img[:, :, max_z], k=1), cmap="gray")
        ax[i].imshow(
            np.rot90(explanation[:, :, max_z], k=1),
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        ax[i].imshow(
            np.rot90(pred[:, :, max_z], k=1),
            cmap=prediction_cmap,
            alpha=0.3,
        )
        ax[i].set_title(
            f"X-Y slice {explanation_type}",
            fontdict={"fontsize": 23},
        )
        ax[i].axis("off")
        ax[i].set_aspect("auto")
        i += 1

    plt.savefig(output_path, bbox_inches="tight", dpi=50)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--current_patient_id", type=int, required=True)
    parser.add_argument("--current_explanations", type=str, required=True)
    parser.add_argument("--selected_label", type=int, default=1)
    parser.add_argument("--current_prediction", type=Path, required=True)
    parser.add_argument("--min_val", type=float, default=0.01)
    parser.add_argument("--max_val", type=float, default=1.0)
    parser.add_argument("--output_path", type=str, default="example_explanations.png")
    parser.add_argument(
        "--explanation_type",
        type=str,
        default="grad",
        choices=["grad", "kernelshap_cubes", "kernelshap_segmentations"],
    )
    parser.add_argument(
        "--select_slices_with",
        type=str,
        default="explanation",
        choices=["prediction", "explanation"],
    )
    parser.add_argument("--zero_background", type=bool, default=False)
    args = parser.parse_args()

    generate_example_explanations(
        args.current_patient_id,
        args.current_explanations,
        args.selected_label,
        args.current_prediction,
        args.min_val,
        args.max_val,
        args.output_path,
        args.explanation_type,
        args.select_slices_with,
        args.zero_background,
    )
