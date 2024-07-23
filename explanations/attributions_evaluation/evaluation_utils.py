import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from quantus.helpers import asserts, utils, warn
from quantus.helpers.model.model_interface import ModelInterface
from tqdm import tqdm

B50_FOLDER = Path("data/b50")
B50_PATHS = sorted([str(x) for x in B50_FOLDER.glob("*/*/*")])

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


def predict_sliding_window(
    self, x: np.ndarray, grad: bool = False, **kwargs
) -> np.array:
    """
    Predict on the given input.

    Parameters
    ----------
    x: np.ndarray
        A given input that the wrapped model predicts on.
    grad: boolean
        Indicates if gradient-calculation is disabled or not.
    kwargs: optional
        Keyword arguments.

    Returns
    --------
    np.ndarray
        predictions of the same dimension and shape as the input, values in the range [0, 1].
    """

    # Use kwargs of predict call if specified, but don't overwrite object attribute
    model_predict_kwargs = {**self.model_predict_kwargs, **kwargs}

    if self.model.training:
        raise AttributeError("Torch model needs to be in the evaluation mode.")

    grad_context = torch.no_grad() if not grad else suppress()

    with grad_context:
        pred_model = self.get_softmax_arg_model()
        # pred = pred_model(torch.Tensor(x).to(self.device), **model_predict_kwargs)
        pred = sliding_window_inference(
            inputs=torch.Tensor(x).unsqueeze(0).to(self.device),
            roi_size=(96, 96, 96),
            sw_batch_size=8,
            predictor=pred_model,
            sw_device=self.device,
            device=torch.device("cpu"),
            **model_predict_kwargs,
        )  # .sum(axis=(2, 3, 4))
        if pred.requires_grad:
            return pred.detach().cpu().numpy()
        return pred.cpu().numpy()


def evaluate_instance_batch_methods(
    self,
    model: ModelInterface,
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    s: np.ndarray,
) -> float:
    """
    Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

    Parameters
    ----------
    model: ModelInterface
        A ModelInteface that is subject to explanation.
    x: np.ndarray
        The input to be evaluated on an instance-basis.
    y: np.ndarray
        The output to be evaluated on an instance-basis.
    a: np.ndarray (or list of np.ndarray)
        The explanation to be evaluated on an instance-basis.
    s: np.ndarray
        The segmentation to be evaluated on an instance-basis.

    Returns
    -------
    float
        The evaluation results.
    """
    # Flatten the attributions.
    a = [a_i.flatten() for a_i in a]
    # Predict on input.
    x_input = model.shape_input(x, x.shape, channel_first=True)
    y_pred_seg = model.predict(x_input)
    y_pred_binary = np.argmax(y_pred_seg, axis=1, keepdims=True) == y
    if y_pred_binary.sum() == 0:
        print(f"No prediction for class {y}!")

    y_pred = float((y_pred_seg[:, y] * y_pred_binary).sum(axis=(2, 3, 4)))

    pred_deltas = []
    att_sums = [[] for _ in range(len(a))]

    # For each test data point, execute a couple of runs.
    for i_ix in range(self.nr_runs):
        # Randomly mask by subset size.
        a_ix = np.random.choice(a[0].shape[0], self.subset_size, replace=False)

        x_perturbed = self.perturb_func(
            arr=x,
            indices=a_ix,
            indexed_axes=self.a_axes,
            **self.perturb_func_kwargs,
        )
        warn.warn_perturbation_caused_no_change(x=x, x_perturbed=x_perturbed)

        # Predict on perturbed input x.
        x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
        y_pred_perturb_seg = model.predict(x_input)
        y_pred_perturb_binary = (
            np.argmax(y_pred_perturb_seg, axis=1, keepdims=True) == y
        )
        y_pred_perturb = float(
            (y_pred_perturb_seg[:, y] * y_pred_perturb_binary).sum(axis=(2, 3, 4))
        )
        pred_deltas.append(float(y_pred - y_pred_perturb))

        # Sum attributions of the random subset.
        for i in range(len(att_sums)):
            att_sums[i].append(np.sum(a[i][a_ix]))

    similarity = [
        self.similarity_func(a=att_sums[i], b=pred_deltas) for i in range(len(att_sums))
    ]

    return similarity


def general_preprocess_batch_methods(
    self,
    model,
    x_batch: np.ndarray,
    y_batch: Optional[np.ndarray],
    a_batch: Optional[np.ndarray],
    s_batch: Optional[np.ndarray],
    channel_first: Optional[bool],
    explain_func: Callable,
    explain_func_kwargs: Optional[Dict[str, Any]],
    model_predict_kwargs: Optional[Dict],
    softmax: bool,
    device: Optional[str],
    custom_batch: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Prepares all necessary variables for evaluation.

        - Reshapes data to channel first layout.
        - Wraps model into ModelInterface.
        - Creates attributions if necessary.
        - Expands attributions to data shape (adds channel dimension).
        - Calls custom_preprocess().
        - Normalises attributions if desired.
        - Takes absolute of attributions if desired.
        - If no segmentation s_batch given, creates list of Nones with as many
            elements as there are data instances.
        - If no custom_batch given, creates list of Nones with as many
            elements as there are data instances.

    Parameters
    ----------

    model: torch.nn.Module, tf.keras.Model
        A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    y_batch: np.ndarray
        A np.ndarray which contains the output labels that are explained.
    a_batch: np.ndarray (or list of np.ndarray)
        A np.ndarray which contains pre-computed attributions i.e., explanations.
    s_batch: np.ndarray, optional
        A np.ndarray which contains segmentation masks that matches the input.
    channel_first: boolean, optional
        Indicates of the image dimensions are channel first, or channel last.
        Inferred from the input shape if None.
    explain_func: callable
        Callable generating attributions.
    explain_func_kwargs: dict, optional
        Keyword arguments to be passed to explain_func on call.
    model_predict_kwargs: dict, optional
        Keyword arguments to be passed to the model's predict method.
    softmax: boolean
        Indicates whether to use softmax probabilities or logits in model prediction.
        This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    custom_batch: any
        Gives flexibility ot the user to use for evaluation, can hold any variable.

    Returns
    -------
    tuple
        A general preprocess.

    """

    # Reshape input batch to channel first order:
    if not isinstance(channel_first, bool):  # None is not a boolean instance.
        channel_first = utils.infer_channel_first(x_batch)
    x_batch = utils.make_channel_first(x_batch, channel_first)

    # Wrap the model into an interface.
    if model:
        # Use attribute value if not passed explicitly.
        model = utils.get_wrapped_model(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
        )

    # Save as attribute, some metrics need it during processing.
    self.explain_func = explain_func
    if explain_func_kwargs is None:
        explain_func_kwargs = {}
    self.explain_func_kwargs = explain_func_kwargs

    # Include device in explain_func_kwargs.
    if device is not None and "device" not in self.explain_func_kwargs:
        self.explain_func_kwargs["device"] = device

    if a_batch is None:
        # Asserts.
        asserts.assert_explain_func(explain_func=self.explain_func)

        # Generate explanations.
        a_batch = self.explain_func(
            model=model.get_model(),
            inputs=x_batch,
            targets=y_batch,
            **self.explain_func_kwargs,
        )

    # Expand attributions to input dimensionality.
    # a_batch = utils.expand_attribution_channel(a_batch, x_batch)

    # Asserts.
    # asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)

    # Infer attribution axes for perturbation function.
    self.a_axes = utils.infer_attribution_axes(a_batch[0], x_batch)

    # Initialize data dictionary.
    data = {
        "model": model,
        "x_batch": x_batch,
        "y_batch": y_batch,
        "a_batch": a_batch,
        "s_batch": s_batch,
        "custom_batch": custom_batch,
    }
    # Call custom pre-processing from inheriting class.
    # custom_preprocess_dict = self.custom_preprocess(**data)
    custom_preprocess_dict = None

    # Save data coming from custom preprocess to data dict.
    if custom_preprocess_dict:
        for key, value in custom_preprocess_dict.items():
            data[key] = value
    # Remove custom_batch if not used.
    if data["custom_batch"] is None:
        del data["custom_batch"]

    # Normalise with specified keyword arguments if requested.
    if self.normalise:
        data["a_batch"] = self.normalise_func(
            a=data["a_batch"],
            normalise_axes=list(range(np.ndim(data["a_batch"])))[1:],
            **self.normalise_func_kwargs,
        )

    # Take absolute if requested.
    if self.abs:
        data["a_batch"] = np.abs(data["a_batch"])

    return data


def get_instance_iterator_batch_methods(self, data: Dict[str, Any]):
    """
    Creates iterator to iterate over all instances in data dictionary.
    Each iterator output element is a keyword argument dictionary with
    string keys.

    Each item key in the input data dictionary has to be of type string.
    - If the item value is not a sequence, the respective item key/value pair
        will be written to each iterator output dictionary.
    - If the item value is a sequence and the item key ends with '_batch',
        a check will be made to make sure length matches number of instances.
        The value of each instance in the sequence will be added to the respective
        iterator output dictionary with the '_batch' suffix removed.
    - If the item value is a sequence but doesn't end with '_batch', it will be treated
        as a simple value and the respective item key/value pair will be
        written to each iterator output dictionary.

    Parameters
    ----------
    data: dict[str, any]
        The data input dictionary.

    Returns
    -------
    iterator
        Each iterator output element is a keyword argument dictionary (string keys).

    """
    n_instances = len(data["x_batch"])

    for key, value in data.items():
        # If data-value is not a Sequence or a string, create list of repeated values with length of n_instances.
        if not isinstance(value, (Sequence, np.ndarray)) or isinstance(value, str):
            data[key] = [value for _ in range(n_instances)]

        elif key.endswith("a_batch"):
            if len(value[0]) != n_instances:
                # Sequence has to have correct length.
                raise ValueError(
                    f"'{key}' has incorrect length (expected: {n_instances}, is: {len(value)})"
                )
        # If data-value is a sequence and ends with '_batch', only check for correct length.
        elif key.endswith("_batch"):
            if len(value) != n_instances:
                # Sequence has to have correct length.
                raise ValueError(
                    f"'{key}' has incorrect length (expected: {n_instances}, is: {len(value)})"
                )

        # If data-value is a sequence and doesn't end with '_batch', create
        # list of repeated sequences with length of n_instances.
        else:
            data[key] = [value for _ in range(n_instances)]

    # We create a list of dictionaries where each dictionary holds all data for a single instance.
    # We remove the '_batch' suffix if present.
    data_instances = [
        {
            re.sub("_batch", "", key): value[id_instance]
            if "a_batch" not in key
            else value
            for key, value in data.items()
        }
        for id_instance in range(n_instances)
    ]

    iterator = tqdm(
        enumerate(data_instances),
        total=n_instances,
        disable=not self.display_progressbar,  # Create progress bar if desired.
        desc=f"Evaluating {self.__class__.__name__}",
    )

    return iterator


totalsegmentator_paths_dir = Path("data/Totalsegmentator_dataset")


def get_totalsegmentator_input_paths():
    totalsegmentator_paths = []
    for id_path in tqdm(list(totalsegmentator_paths_dir.glob("s*"))):
        scan_reconstruction = id_path / "ct.nii.gz"
        if scan_reconstruction.exists():
            totalsegmentator_paths.append(
                dict(
                    scan_reconstruction_path=scan_reconstruction,
                    patient_id=id_path.name,
                    study_id="",
                    reconstruction_id="",
                    seg_path=(id_path / "segmentations"),
                )
            )
    return totalsegmentator_paths


def get_totalsegmentator_test_input_paths():
    totalsegmentator_paths = []

    metadata = pd.read_csv(
        "data/Totalsegmentator_dataset/meta.csv",
        delimiter=";",
    )
    image_ids = metadata.loc[metadata.split == "test"]["image_id"].values

    for id_path in tqdm(list(totalsegmentator_paths_dir.glob("s*"))):
        scan_reconstruction = id_path / "ct.nii.gz"
        if scan_reconstruction.exists() and id_path.name in image_ids:
            totalsegmentator_paths.append(
                dict(
                    scan_reconstruction_path=scan_reconstruction,
                    patient_id=id_path.name,
                    study_id="",
                    reconstruction_id="",
                    seg_path=(id_path / "segmentations"),
                )
            )
    return totalsegmentator_paths


from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientation,
    Orientationd,
    ScaleIntensityRange,
    ScaleIntensityRanged,
    Spacing,
    ToTensor,
    ToTensord,
)

test_transform = Compose(
    [
        LoadImaged(keys="image", reader="NibabelReader"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
        ),
        Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ToTensord(keys="image"),
    ]
)

inference_transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
        Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ToTensor(),
    ]
)


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
