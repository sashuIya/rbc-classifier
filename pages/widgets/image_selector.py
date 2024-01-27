import os
from filepath_util import get_rel_filepaths_from_subfolders, read_images_metadata
from dash import html, dcc

_IMAGES_PATH = os.path.normpath("./dataset/")
TIF_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=_IMAGES_PATH, extension="tif"
)


def get_image_filepath_options():
    df = read_images_metadata()
    options = []
    for filepath in TIF_FILEPATHS:
        completed = (
            filepath in df["filepath"].values
            and df.loc[df["filepath"] == filepath, "completed"].iloc[0]
        )

        options.append(
            {
                "label": html.Span(
                    [filepath],
                    style={
                        "background-color": "palegreen" if completed else "moccasin"
                    },
                ),
                "value": filepath,
            }
        )
    return options


def image_selection_dropdown(id: str) -> dcc.Dropdown:
    return dcc.Dropdown(get_image_filepath_options(), TIF_FILEPATHS[0], id=id)
