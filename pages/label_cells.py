from dash import (
    Dash,
    html,
    ALL,
    dcc,
    callback,
    Output,
    Input,
    State,
    ctx,
    register_page,
    MATCH,
)
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_util import id_factory
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

import glob
import base64
import os

import pandas as pd
import numpy as np
import cv2

from draw_util import get_masks_img, get_masked_crop
from filepath_util import (
    read_masks_for_image,
    read_image,
    get_rel_filepaths_from_subfolders,
    get_masks_features_filepath,
    read_masks_features,
    write_masks_features,
    get_classifier_model_filepaths,
)
from mask_util import is_point_in_mask
from train_classifier import (
    train_pipeline,
    classify,
)

from consts import (
    # Features metadata
    Y_COLUMN,
    MASK_ID_COLUMN,
    # Labeling modes
    LABELING_MODE_COLUMN,
    LABELING_MANUAL,
    LABELING_APPROVED,
    LABELING_AUTO,
    # Labels (options of Y_COLUMN)
    LABEL_UNLABELED,
    LABEL_WRONG,
)

IMAGES_PATH = os.path.normpath("./dataset/")

DISPLAY_IMAGE = "Image"
DISPLAY_MASKS = "Masks"
DISPLAY_LABELED_DATA = "Labeled data"
DISPLAY_UNLABELED_DATA = "Unlabeled data"
DISPLAY_OPTIONS = [
    DISPLAY_IMAGE,
    DISPLAY_MASKS,
    DISPLAY_LABELED_DATA,
    DISPLAY_UNLABELED_DATA,
]

LABELS = {
    LABEL_UNLABELED: {"color": [255, 255, 255]},
    LABEL_WRONG: {"color": [0, 0, 0]},
    "0": {"color": [255, 0, 0]},
    "1": {"color": [0, 255, 0]},
    "2": {"color": [0, 0, 255]},
}

id = id_factory("label-cells")
register_page(__name__, order=2)

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

TIF_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=IMAGES_PATH, extension="tif"
)

CLASSIFIER_MODEL_FILEPATHS = get_classifier_model_filepaths()
if not CLASSIFIER_MODEL_FILEPATHS:
    CLASSIFIER_MODEL_FILEPATHS = ["none"]

# Create dictionaries to store processed images and image traces
image_cache = {}
image_trace_cache = {}

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(children="Labeling tool", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        TIF_FILEPATHS, TIF_FILEPATHS[0], id=id("image-filepath")
                    ),
                    dcc.Dropdown(
                        CLASSIFIER_MODEL_FILEPATHS,
                        CLASSIFIER_MODEL_FILEPATHS[0],
                        id=id("classifier-model"),
                    ),
                    html.Button(
                        "Train classifier on labeled data",
                        id=id("train-classifier-button"),
                        n_clicks=0,
                    ),
                    html.Button(
                        "Run classifier",
                        id=id("run-classifier-button"),
                        n_clicks=0,
                    ),
                    html.Button(
                        "Save masks",
                        id=id("save-labels-button"),
                        n_clicks=0,
                    ),
                    dcc.RadioItems(
                        DISPLAY_OPTIONS, DISPLAY_LABELED_DATA, id=id("display-options")
                    ),
                    html.Div(id=id("clicked-pixel-coords")),
                    html.Div(style={"padding": "20px"}),
                    dcc.RadioItems(list(LABELS.keys()), "0", id=id("active-label")),
                ],
                # width=22,
            )
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id=id("canvas"))),
                dbc.Col(id=id("selected-masks")),
            ],
            justify="between",
        ),
        dcc.Store(id=id("labeled-masks")),
        dcc.Store(id=id("figure-store")),
    ],
    fluid=True,
)


# Creates Plotly image trace function with caching (Optimized).
def create_image_trace(image, image_filepath):
    if image_filepath in image_trace_cache:
        return image_trace_cache[image_filepath]

    img_trace = px.imshow(image).data[0]
    image_trace_cache[image_filepath] = img_trace
    return img_trace


def image_to_base64(image_array):
    # Convert the image array to base64-encoded string
    image_base64 = base64.b64encode(image_array).decode("utf-8")
    return image_base64


def ndarray_to_b64(ndarray):
    """
    converts a np ndarray to a b64 string readable by html-img tags
    """
    img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def crop_html(crop):
    _, crop_png = cv2.imencode(".png", crop)
    crop_base64 = base64.b64encode(crop_png).decode("utf-8")
    crop_html = html.Img(
        src=f"data:image/png;base64,{crop_base64}",
        style={"display": "block", "margin-bottom": "10px"},
        className="img-item",
    )

    return crop_html


def generate_crop_with_radio(
    crop_with_highlighting, original_crop, mask_id, labels, label
):
    return dbc.Form(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            "mask_id: {}".format(mask_id),
                            id={"type": id("mask-id-div"), "index": mask_id},
                        ),
                    ),
                    dbc.Col(
                        crop_html(crop_with_highlighting),
                        style={"width": "{}px".format(crop_with_highlighting.shape[1])},
                    ),
                    dbc.Col(
                        crop_html(original_crop),
                        style={"width": "{}px".format(original_crop.shape[1])},
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            options=labels,
                            value=label,
                            id={"type": id("radio-item"), "index": mask_id},
                            inline=False,
                            className="ml-3",
                        )
                    ),
                ],
                className="img-container",
                style={"margin-bottom": "5px"},
            ),
            dbc.Row(
                dbc.Col(html.Hr(style={"margin-top": "10px", "margin-bottom": "20px"}))
            ),
        ],
    )


@callback(
    Output(id("labeled-masks"), "data", allow_duplicate=True),
    Input({"type": id("radio-item"), "index": ALL}, "value"),
    State({"type": id("radio-item"), "index": ALL}, "id"),
    State(id("labeled-masks"), "data"),
    prevent_initial_call=True,
    suppress_callback_exceptions=True,
)
def update_label(labels, ids, labeled_masks: dict):
    labeled_masks = pd.DataFrame(labeled_masks)
    for label, id in zip(labels, ids):
        mask_id = id["index"]

        print("changing mask_id {} to {}".format(mask_id, label))

        labeled_masks.loc[
            labeled_masks[MASK_ID_COLUMN] == mask_id,
            [Y_COLUMN, LABELING_MODE_COLUMN],
        ] = (label, LABELING_MANUAL)

    return labeled_masks.to_dict()


@callback(
    Output(id("canvas"), "figure", allow_duplicate=True),
    Input(id("labeled-masks"), "data"),
    State(id("display-options"), "value"),
    State(id("image-filepath"), "value"),
    prevent_initial_call=True,
)
def handle_labels_change(labeled_masks, display_option, image_filepath):
    if display_option != DISPLAY_LABELED_DATA:
        raise PreventUpdate

    if not labeled_masks:
        raise PreventUpdate

    print("Masks or display-option has changed")

    labeled_masks = pd.DataFrame(labeled_masks)

    masks = read_masks_for_image(image_filepath)
    masks_to_display = []
    for mask in masks:
        mask_id = mask["id"]
        label = labeled_masks.loc[
            labeled_masks[MASK_ID_COLUMN] == mask_id, Y_COLUMN
        ].values[0]
        if label in [LABEL_UNLABELED, LABEL_WRONG]:
            continue
        mask["color"] = LABELS[label]["color"]
        masks_to_display.append(mask)

    image = read_image(image_filepath, with_alpha=False)
    masks_image = get_masks_img(masks_to_display, image)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]

    fig = go.Figure()
    fig.update_layout(autosize=False, width=1024, height=1024)
    fig.add_trace(masks_trace)

    return fig


@callback(
    Output(id("clicked-pixel-coords"), "children"),
    Output(id("selected-masks"), "children"),
    Output(id("labeled-masks"), "data", allow_duplicate=True),
    Input(id("canvas"), "clickData"),
    State(id("active-label"), "value"),
    State(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
    prevent_initial_call=True,
)
def handle_canvas_click(
    click_data, active_label: str, labeled_masks: dict, image_filepath: str
):
    if not click_data:
        raise PreventUpdate

    labeled_masks = pd.DataFrame(labeled_masks)

    point = click_data["points"][0]
    x, y = point["x"], point["y"]

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath)

    assert len(masks) == labeled_masks.shape[0]

    crops = []
    for mask in masks:
        mask_id = mask["id"]
        if is_point_in_mask(x, y, mask):
            label = active_label if len(crops) == 0 else LABEL_WRONG
            crops.append(
                (
                    get_masked_crop(
                        image, mask, xy_threshold=20, with_highlighting=True
                    ),
                    get_masked_crop(
                        image, mask, xy_threshold=20, with_highlighting=False
                    ),
                    label,
                    mask_id,
                )
            )
            labeled_masks.loc[
                labeled_masks[MASK_ID_COLUMN] == mask_id,
                [Y_COLUMN, LABELING_MODE_COLUMN],
            ] = (label, LABELING_MANUAL)

    labels = list(LABELS.keys())

    image_radio_items = []
    for crop_with_highlighting, original_crop, label, mask_id in crops:
        image_radio_item = generate_crop_with_radio(
            crop_with_highlighting, original_crop, mask_id, labels, label
        )
        image_radio_items.append(image_radio_item)

    return (
        html.H3("x: {}, y: {}".format(x, y)),
        image_radio_items,
        labeled_masks.to_dict(),
    )


@callback(
    Output(id("save-labels-button"), "style"),
    Input(id("save-labels-button"), "n_clicks"),
    State(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
)
def handle_save_labels_button_click(n_clicks, labeled_masks, image_filepath):
    if n_clicks == 0 or not labeled_masks:
        raise PreventUpdate

    labeled_masks = pd.DataFrame(labeled_masks)

    labeled_masks.loc[
        (labeled_masks[Y_COLUMN] != LABEL_UNLABELED)
        # & (labeled_masks[Y_COLUMN] != LABEL_WRONG)
        & (labeled_masks[LABELING_MODE_COLUMN] == LABELING_AUTO),
        LABELING_MODE_COLUMN,
    ] = LABELING_APPROVED

    masks_features = read_masks_features(image_filepath)
    labeled_masks.index = masks_features.index
    masks_features[labeled_masks.columns] = labeled_masks

    write_masks_features(masks_features, image_filepath)

    return {}


@callback(
    Output(id("canvas"), "figure", allow_duplicate=True),
    Input(id("display-options"), "value"),
    Input(id("canvas"), "figure"),
    prevent_initial_call=True,
)
def handle_display_option_change(display_option, figure):
    if not figure:
        raise PreventUpdate

    # TODO: Implement this.

    return figure


@callback(
    Output(id("canvas"), "figure"),
    Output(id("labeled-masks"), "data"),
    Input(id("image-filepath"), "value"),
)
def handle_image_filepath_selection(image_filepath):
    if not image_filepath:
        return {}, {}, {}

    print("Image filepath changed")

    image = read_image(image_filepath, with_alpha=False)
    masks = read_masks_for_image(image_filepath)
    masks_image = get_masks_img(masks, image)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]

    image_fig = go.Figure()
    image_fig.update_layout(autosize=False, width=1024, height=1024)
    image_fig.add_trace(masks_trace)

    labeled_masks = read_masks_features(image_filepath)
    print("read labeled_masks, shape", labeled_masks.shape)
    print(labeled_masks[[MASK_ID_COLUMN, Y_COLUMN, LABELING_MODE_COLUMN]].head())

    assert len(masks) == labeled_masks.shape[0], (
        "Labels do not correspond to the masks."
        "Probably you updated the masks."
        "Consider removing {}".format(get_masks_features_filepath(image_filepath))
    )

    labeled_masks = labeled_masks[[MASK_ID_COLUMN, Y_COLUMN, LABELING_MODE_COLUMN]]

    return image_fig, labeled_masks.to_dict()


@callback(
    Output(id("classifier-model"), "options"),
    Output(id("classifier-model"), "value"),
    Input(id("train-classifier-button"), "n_clicks"),
)
def handle_train_classifier_button(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    train_pipeline()
    model_filepaths = get_classifier_model_filepaths()
    if not model_filepaths:
        return ["none"], "none"

    return model_filepaths, model_filepaths[0]


@callback(
    Output(id("labeled-masks"), "data", allow_duplicate=True),
    Input(id("run-classifier-button"), "n_clicks"),
    State(id("classifier-model"), "value"),
    State(id("image-filepath"), "value"),
    prevent_initial_call=True,
)
def handle_run_classifier_button(n_clicks, classifier_model_filepath, image_filepath):
    if not n_clicks or not image_filepath or not classifier_model_filepath:
        raise PreventUpdate

    labeled_masks = read_masks_features(image_filepath)
    predictions = classify(
        labeled_masks[labeled_masks[Y_COLUMN] == LABEL_UNLABELED],
        classifier_model_filepath,
    )

    labeled_masks.loc[
        labeled_masks[Y_COLUMN] == LABEL_UNLABELED, Y_COLUMN
    ] = predictions

    return labeled_masks.to_dict()
