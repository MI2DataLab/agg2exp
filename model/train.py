import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from data_loader import TotalChestSegmentatorDataset
from models.swin_unetr import SwinUNETR
from models.unetr import UNETR
from monai.apps import download_url
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default="../../Totalsegmentator_dataset",
    type=str,
    help="Path to the training data.",
)
parser.add_argument(
    "--data_train",
    default="Totalsegmentator_dataset_full_v3/train",
    type=str,
    help="Path to the training data.",
)
parser.add_argument(
    "--data_val",
    default="Totalsegmentator_dataset_full_v3/val",
    type=str,
    help="Path to the training data.",
)
parser.add_argument("--batch_size", default=2, type=int, help="Number of batch size.")
parser.add_argument(
    "--skip_val", default=3, type=int, help="Skip validation step by N epochs."
)
parser.add_argument("--classes", default=17, type=int, help="Number of classes.")
parser.add_argument("--epochs", default=250, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate value.")
parser.add_argument(
    "--weight_decay", default=1e-5, type=float, help="Weight decay value."
)
parser.add_argument("--optimizer", default="AdamW", type=str, help="Type of optimizer.")
parser.add_argument(
    "--scheduler", default="CALR", type=str, help="Type of learning rate scheduler."
)
parser.add_argument("--k_fold", default=5, type=int, help="Number of K-Fold splits.")
parser.add_argument(
    "--patch_size", default=(96, 96, 96), type=list, help="Patch size value."
)
parser.add_argument(
    "--feature_size", default=48, type=int, help="Feature size of Transformer."
)
parser.add_argument(
    "--use_checkpoint",
    default=True,
    type=bool,
    help="Use checkpoint in training model.",
)
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers.")
parser.add_argument("--pin_memory", default=True, type=bool, help="Pin memory.")
parser.add_argument(
    "--use_pretrained", default=False, type=bool, help="Use pre-trained weights."
)
parser.add_argument("--model", default="SwinUNETR", type=str, help="Type of model.")
parser.add_argument("--parallel", default=True, type=bool, help="Use multi-GPU.")
parser.add_argument(
    "--model_name", default="Fullbody", type=str, help="File model name."
)

args = parser.parse_args()

if args.use_pretrained:
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
    dst = "./ssl_pretrained_weights.pth"
    download_url(resource, dst)
    pretrained_path = os.path.normpath(dst)


def train(global_step, train_loader, valid_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, len(train_loader) * args.epochs, loss)
        )
        if (
            global_step % (args.skip_val * len(train_loader)) == 0 and global_step != 0
        ) or global_step == len(train_loader) * args.epochs:
            epoch_iterator_val = tqdm(
                valid_loader,
                desc="Validate (X / X Steps) (dice=X.X)",
                dynamic_ncols=True,
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.data_train, f"TotalSegmentator_{args.model}_clean_v3.pth"
                    ),
                )
                print(
                    "Model was saved...Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model was not saved...Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            scheduler.step()
        global_step += 1

    return global_step, dice_val_best, global_step_best


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (
                batch["image"].to("cpu"),
                batch["label"].to("cpu"),
            )
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    val_inputs,
                    args.patch_size,
                    1,
                    model,
                    sw_device="cuda",
                    device="cpu",
                    buffer_steps=8,
                    buffer_dim=-1,
                )
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric1(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val1 = dice_metric1.aggregate().item()
        dice_metric1.reset()
    return mean_dice_val1


train_ds = TotalChestSegmentatorDataset(
    args.data_train, mode="train", patch_size=args.patch_size
)
valid_ds = TotalChestSegmentatorDataset(args.data_val, mode="valid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()

post_label = AsDiscrete(to_onehot=args.classes)
post_pred = AsDiscrete(argmax=True, to_onehot=args.classes)
dice_metric1 = DiceMetric(
    include_background=False, reduction="mean", get_not_nans=False
)
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
)

valid_loader = DataLoader(valid_ds, batch_size=1, num_workers=0, pin_memory=False)

if args.model == "SwinUNETR":
    model = SwinUNETR(
        img_size=args.patch_size,
        in_channels=1,
        out_channels=args.classes,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        use_v2=True,
    )
elif args.model == "UNETR":
    model = UNETR(
        img_size=args.patch_size,
        in_channels=1,
        out_channels=args.classes,
        feature_size=args.feature_size,
    )
else:
    raise NotImplementedError("This model not exists!")

if args.use_pretrained:
    print("Loading Weights from the Path {}".format(pretrained_path))
    ssl_dict = torch.load(pretrained_path)
    ssl_weights = ssl_dict["model"]
    print(ssl_weights)

    # Generate new state dict so it can be loaded to MONAI SwinUNETR Model
    monai_loadable_state_dict = OrderedDict()
    model_prior_dict = model.state_dict()
    model_update_dict = model_prior_dict

    del ssl_weights["encoder.mask_token"]
    del ssl_weights["encoder.norm.weight"]
    del ssl_weights["encoder.norm.bias"]
    del ssl_weights["out.conv.conv.weight"]
    del ssl_weights["out.conv.conv.bias"]

    for key, value in ssl_weights.items():
        if key[:8] == "encoder.":
            if key[8:19] == "patch_embed":
                new_key = "swinViT." + key[8:]
            else:
                new_key = "swinViT." + key[8:18] + key[20:]
            monai_loadable_state_dict[new_key] = value
        else:
            monai_loadable_state_dict[key] = value

    model_update_dict.update(monai_loadable_state_dict)
    model.load_state_dict(model_update_dict, strict=True)
    model_final_loaded_dict = model.state_dict()

    # Safeguard test to ensure that weights got loaded successfully
    layer_counter = 0
    for k, _v in model_final_loaded_dict.items():
        if k in model_prior_dict:
            layer_counter = layer_counter + 1

            old_wts = model_prior_dict[k]
            new_wts = model_final_loaded_dict[k]

            old_wts = old_wts.to("cpu").numpy()
            new_wts = new_wts.to("cpu").numpy()
            diff = np.mean(np.abs(old_wts, new_wts))
            print("Layer {}, the update difference is: {}".format(k, diff))
            if diff == 0.0:
                print("Warning: No difference found for layer {}".format(k))
    print("Total updated layers {} / {}".format(layer_counter, len(model_prior_dict)))
    print("Pretrained Weights Succesfully Loaded !")


elif args.use_pretrained is False:
    print("No weights were loaded, all weights being used are randomly initialized!")

if args.parallel:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)

if args.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
elif args.optimizer == "Adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
elif args.optimizer == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
else:
    raise NotImplementedError(
        "Optimizer not found! Please select one of [Adam, AdamW or SGD]"
    )

if args.scheduler == "CALR":
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epochs // args.skip_val, verbose=True
    )
else:
    raise NotImplementedError(
        "Learning rate scheduler not found! Please select one of [CALR]"
    )

dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
global_step = 0

while global_step < len(train_loader) * args.epochs:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, valid_loader, dice_val_best, global_step_best
    )
