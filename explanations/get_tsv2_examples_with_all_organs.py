from pathlib import Path

import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, ToTensor
from pqdm.processes import pqdm

TSV2_TRAIN_FOLDER = Path("data/Totalsegmentator_dataset/train/")
NUM_CLASSES = 17
patients = list(TSV2_TRAIN_FOLDER.glob("*/segmentations.nii.gz"))

transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
    ]
)


def does_patient_have_all_organs(patient):
    image = transform(patient)
    return len(image.unique()) == NUM_CLASSES


def main(args):
    df = pd.DataFrame(
        {
            "all_organs": pqdm(
                patients,
                does_patient_have_all_organs,
                n_jobs=args.n_jobs,
                desc="Checking patients",
            ),
            "patient_path": [str(x.parent / "ct.nii.gz") for x in patients],
        }
    )
    if args.save_type == "txt":
        with open(args.output_file + ".txt", "w") as f:
            f.write("\n".join(df[df.all_organs].patient_path))
    elif args.save_type == "csv":
        df.to_csv(args.output_file + ".csv", index=False)
    else:
        raise ValueError(f"Unknown save type: {args.save_type}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument(
        "--output_file",
        type=str,
        default="tsv2_patients_with_all_organs",
        help="Output file path",
    )
    parser.add_argument(
        "--save-type",
        type=str,
        default="txt",
        choices=["txt", "csv"],
        help="Save type",
    )
    parser.add_argument(
        "--config", action=ActionConfigFile, help="Path to JSON or YAML config file"
    )
    args = parser.parse_args()
    main(args)
