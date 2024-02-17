import os

from dash import dcc, html

from src.common.filepath_util import (
    get_rel_filepaths_from_subfolders,
    read_images_metadata,
)

_IMAGES_PATH = os.path.normpath("./dataset/")
TIF_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=_IMAGES_PATH, extension="tif", exclude="result"
)


def is_completed(df, filepath):
    return (
        filepath in df["filepath"].values
        and df.loc[df["filepath"] == filepath, "completed"].iloc[0]
    )


def is_measured(df, filepath):
    return (
        filepath in df["filepath"].values
        and df.loc[df["filepath"] == filepath, "measured"].iloc[0]
    )


def get_image_filepath_options(predicate_fn):
    df = read_images_metadata()
    options = []
    for filepath in TIF_FILEPATHS:
        predicate_value = predicate_fn(df, filepath)
        options.append(
            {
                "label": html.Span(
                    [filepath],
                    style={
                        "background-color": (
                            "palegreen" if predicate_value else "moccasin"
                        )
                    },
                ),
                "value": filepath,
            }
        )
    return options


def image_selection_dropdown(id: str, predicate_fn) -> dcc.Dropdown:
    return dcc.Dropdown(
        get_image_filepath_options(predicate_fn), TIF_FILEPATHS[0], id=id
    )
