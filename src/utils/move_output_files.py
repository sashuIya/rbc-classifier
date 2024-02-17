import re
import shutil
from pathlib import Path

from src.common.filepath_util import get_rel_filepaths_from_subfolders


def remove_suffix_with_number(input_string):
    return re.sub(r"_result_label_counts$", "", input_string)


filepaths = get_rel_filepaths_from_subfolders("dataset/raw/", "tif")
# print(filepaths)

TESTING = True

for filepath in filepaths:
    f = Path(filepath)

    if not f.name.endswith("_result.tif"):
        continue

    move_to = (
        "dataset/processed"
        / f.relative_to("dataset/raw").parent
        / remove_suffix_with_number(f.stem)
        / "labeled_image.tif"
    )

    print(f"moving {f}\nto\n{move_to}")

    if TESTING:
        break

    move_to.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(f, move_to)
