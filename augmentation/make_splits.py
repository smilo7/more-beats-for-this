"""
Create splits based on the original splits, i.e. augmented tracks stay in the
same fold or keep the same train/validation value
"""

import argparse
from pathlib import Path


def make_splits(original_split_folder, output_split_folder, augmentations=["24", "34"]):
    split_files = ("8-folds.split", "single.split")
    for split in split_files:
        split_path = Path(original_split_folder) / split
        entries = split_path.read_text().strip().split("\n")
        entries = [tuple(entry.split("\t")) for entry in entries]
        output = []
        for entry in entries:
            for aug in augmentations:
                output.append((f"{entry[0]}_{aug}", entry[1]))
        breakpoint()
        # with open(Path(output_split_folder) / split, "w") as file:
        #     for entry in output:
        #         file.write(f"{entry[0]}\t{entry[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_split_folder",
        type=str,
        required=True,
        help="path for original split",
    )
    parser.add_argument(
        "--output_split_folder", type=str, required=True, help="path for output split"
    )
    args = parser.parse_args()

    make_splits(args.original_split_folder, args.output_split_folder)
