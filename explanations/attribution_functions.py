from typing import Callable, Literal, Optional, Sequence

import torch
from sliding_window_gradient_inference import sliding_window_inference, sum_aggregation
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pred_and_grad(
    loaded_image,
    model,
    class_for_saliency,
    sw_batch_size=1,
    roi_size: Sequence[int] = (96, 96, 96),
    preds_for_saliency: Optional[torch.Tensor] = None,
    mask_for_aggregation: Optional[torch.Tensor] = None,
    progress=False,
    aggregation_fn=sum_aggregation,
    **kwargs,
):
    pred, grad = sliding_window_inference(
        loaded_image,
        roi_size,
        sw_batch_size,
        model,
        sw_device=DEVICE,
        device=torch.device("cpu"),
        class_for_saliency=class_for_saliency,
        preds_for_saliency=preds_for_saliency,
        mask_for_aggregation=mask_for_aggregation,
        aggregation_fn=aggregation_fn,
        progress=progress,
    )
    return pred, grad


def get_grad_quantile_attributions(
    loaded_image,
    model,
    class_for_saliency,
    sw_batch_size=1,
    roi_size: Sequence[int] = (96, 96, 96),
    preds_for_saliency: Optional[torch.Tensor] = None,
    mask_for_aggregation: Optional[torch.Tensor] = None,
    quantile=0.9,
    num_samples=100,
    gradient_sampling_mode: Literal["split_mask", "sample_mask"] = "sample_mask",
    progress=False,
    **kwargs,
):
    pred, grad = sliding_window_inference(
        loaded_image,
        roi_size,
        sw_batch_size,
        model,
        sw_device=DEVICE,
        device=torch.device("cpu"),
        class_for_saliency=class_for_saliency,
        preds_for_saliency=preds_for_saliency,
        mask_for_aggregation=mask_for_aggregation,
        num_gradient_samples=num_samples,
        gradient_sampling_mode=gradient_sampling_mode,
        gradient_sample_aggregation_fn=torch.quantile,
        gradient_sample_aggregation_fn_kwargs={"q": quantile},
        progress=progress,
    )
    return pred, grad


def get_ig_attribution(
    loaded_image,
    model,
    class_for_saliency,
    baseline_function: Callable[[torch.Tensor], torch.Tensor] = torch.zeros_like,
    steps=20,
    preds_for_saliency: Optional[torch.Tensor] = None,
    sw_batch_size=1,
    roi_size: Sequence[int] = (96, 96, 96),
    progress=True,
    mask_for_aggregation: Optional[torch.Tensor] = None,
    use_not_perturbed_image_pred: bool = False,
    **kwargs,
):
    baseline = baseline_function(loaded_image)
    alphas = torch.linspace(0, 1, steps=steps)
    if (preds_for_saliency is None) and use_not_perturbed_image_pred:
        with torch.inference_mode():
            preds_for_saliency = sliding_window_inference(
                loaded_image,
                roi_size,
                sw_batch_size,
                model,
                sw_device=DEVICE,
                device=torch.device("cpu"),
            )

    ig = torch.repeat_interleave(
        torch.zeros_like(loaded_image).unsqueeze(0),
        len(class_for_saliency),
        dim=0,
    )
    for alpha in tqdm(alphas, leave=False) if progress else alphas:
        cur_image = baseline + alpha * (loaded_image - baseline)
        cur_image.requires_grad = True
        _, cur_grad = get_pred_and_grad(
            cur_image,
            model,
            class_for_saliency,
            sw_batch_size,
            roi_size,
            preds_for_saliency=preds_for_saliency,
            mask_for_aggregation=mask_for_aggregation,
        )
        ig += cur_grad * (loaded_image - baseline) / steps
    return preds_for_saliency, ig


def get_smoothgrad_attribution(
    loaded_image,
    model,
    class_for_saliency,
    sigma_level=0.1,
    steps=20,
    preds_for_saliency: Optional[torch.Tensor] = None,
    sw_batch_size=1,
    roi_size: Sequence[int] = (96, 96, 96),
    progress=True,
    mask_for_aggregation: Optional[torch.Tensor] = None,
    use_not_perturbed_image_pred: bool = False,
    **kwargs,
):
    x_min = loaded_image.min()
    x_max = loaded_image.max()
    sigma = sigma_level * (x_max - x_min)

    if (preds_for_saliency is None) and use_not_perturbed_image_pred:
        with torch.inference_mode():
            preds_for_saliency = sliding_window_inference(
                loaded_image,
                roi_size,
                sw_batch_size,
                model,
                sw_device=DEVICE,
                device=torch.device("cpu"),
            )

    smoothgrad = torch.repeat_interleave(
        torch.zeros_like(loaded_image).unsqueeze(0),
        len(class_for_saliency),
        dim=0,
    )

    for _ in tqdm(range(steps), leave=False) if progress else range(steps):
        cur_image = loaded_image + torch.randn_like(loaded_image) * sigma
        _, cur_grad = get_pred_and_grad(
            cur_image,
            model,
            class_for_saliency,
            sw_batch_size,
            roi_size,
            preds_for_saliency=preds_for_saliency,
            mask_for_aggregation=mask_for_aggregation,
        )
        smoothgrad += cur_grad / steps

    return preds_for_saliency, smoothgrad
