from dash import Dash, html, dcc, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

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
    write_labels,
    read_labels_for_image,
    get_labels_filepath,
)
from mask_util import is_point_in_mask

IMAGES_PATH = os.path.normpath("./dataset/")
SAM_CHECKPOINTS_FOLDER = os.path.normpath("./model/sam/")

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

LABEL_UNLABELED = "unlabeled"
LABEL_WRONG = "wrong"
LABELS = {
    LABEL_UNLABELED: {"color": [255, 255, 255]},
    LABEL_WRONG: {"color": [0, 0, 0]},
    "0": {"color": [255, 0, 0, 0.35]},
    "1": {"color": [0, 255, 0, 0.35]},
    "2": {"color": [0, 0, 255, 0.35]},
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


TIF_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=IMAGES_PATH, extension="tif"
)
SAM_CHECKPOINT_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=SAM_CHECKPOINTS_FOLDER, extension="pth"
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        children="Title of Dash App", style={"textAlign": "center"}
                    ),
                    dcc.Dropdown(TIF_FILEPATHS, TIF_FILEPATHS[0], id="image-filepath"),
                    dcc.Dropdown(
                        SAM_CHECKPOINT_FILEPATHS,
                        SAM_CHECKPOINT_FILEPATHS[0],
                        id="sam-checkpoint-filepath",
                    ),
                    dcc.RadioItems(
                        DISPLAY_OPTIONS, DISPLAY_LABELED_DATA, id="display-options"
                    ),
                    html.Button("Run SAM", id="run-sam-button", n_clicks=0),
                    html.Div(id="clicked-pixel-coords"),
                    html.Div(style={"padding": "20px"}),
                    dcc.RadioItems(list(LABELS.keys()), "0", id="active-label"),
                ],
                # width=22,
            )
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="canvas")),
                dbc.Col(id="selected-masks"),
            ],
            justify="between",
        ),
        dcc.Store(id="labeled-masks"),
        dcc.Store(id="figure-store"),
    ],
    fluid=True,
)


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


def generate_crop_with_radio(crop, labels, label, index):
    _, crop_png = cv2.imencode(".png", crop)
    crop_base64 = base64.b64encode(crop_png).decode("utf-8")
    crop_html = html.Img(
        src=f"data:image/png;base64,{crop_base64}",
        style={"display": "block", "margin-bottom": "10px"},
        className="img-item",
    )
    return dbc.Form(
        [
            dbc.Row(
                [
                    dbc.Col(crop_html),
                    dbc.Col(
                        dbc.RadioItems(
                            options=labels,
                            value=label,
                            id=f"radio-item-{index}",
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
    Output("canvas", "figure", allow_duplicate=True),
    Input("labeled-masks", "data"),
    Input("display-options", "value"),
    State("image-filepath", "value"),
    prevent_initial_call=True,
)
def handle_labels_change(labeled_masks, display_option, image_filepath):
    if display_option != DISPLAY_LABELED_DATA:
        raise PreventUpdate

    if ctx.triggered_id != "labeled-masks" and ctx.triggered_id != "display-options":
        raise PreventUpdate

    labeled_masks = pd.DataFrame(labeled_masks)
    fig = go.FigureWidget()
    fig.update_layout(autosize=False, width=1024, height=1024)

    image = read_image(image_filepath)
    img_trace = px.imshow(image).data[0]
    fig.add_trace(img_trace)

    masks = read_masks_for_image(image_filepath)
    masks_to_display = []
    for mask in masks:
        mask_id = mask["id"]
        label = labeled_masks.loc[labeled_masks["id"] == mask_id, "label"].values[0]
        mask["color"] = LABELS[label]["color"]
        if label in [LABEL_UNLABELED, LABEL_WRONG]:
            continue
        masks_to_display.append(mask)

    masks_image = get_masks_img(masks_to_display, image)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]
    fig.add_trace(masks_trace)
    fig.data[1].update(opacity=0.3)

    return fig


@callback(
    Output("clicked-pixel-coords", "children"),
    Output("selected-masks", "children"),
    Output("labeled-masks", "data", allow_duplicate=True),
    Input("canvas", "clickData"),
    State("active-label", "value"),
    State("labeled-masks", "data"),
    State("image-filepath", "value"),
    prevent_initial_call=True,
)
def handle_canvas_click(
    click_data, active_label: str, labeled_masks: dict, image_filepath: str
):
    if ctx.triggered_id == "image-filepath":
        return [], [], labeled_masks

    if ctx.triggered_id == "active-label":
        raise PreventUpdate
    if not click_data:
        raise PreventUpdate

    labeled_masks = pd.DataFrame(labeled_masks)

    point = click_data["points"][0]
    x, y = point["x"], point["y"]

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath)
    crops = []
    for mask in masks:
        mask_id = mask["id"]
        if is_point_in_mask(x, y, mask):
            label = active_label if len(crops) == 0 else LABEL_WRONG
            crops.append((get_masked_crop(image, mask), label))
            labeled_masks.loc[
                labeled_masks["id"] == mask_id, ["label", "manually_labeled"]
            ] = (label, True)

    write_labels(image_filepath, labeled_masks)

    labels = list(LABELS.keys())

    image_radio_items = []
    for i, (crop, label) in enumerate(crops):
        image_radio_item = generate_crop_with_radio(crop, labels, label, i)
        image_radio_items.append(image_radio_item)

    return (
        html.H3("x: {}, y: {}".format(x, y)),
        image_radio_items,
        labeled_masks.to_dict(),
    )


# @callback(
#     Output('canvas', 'figure', allow_duplicate=True),
#     Input('display-options', 'value'),
#     Input('canvas', 'figure'),
#     prevent_initial_call=True
# )
# def handle_display_option_change(display_option, figure):
#     if not figure:
#         raise PreventUpdate

#     # Assuming that data[0] is image layer and data[1] is masks layer (see
#     # `update_graph`).
#     if display_option == DISPLAY_IMAGE:
#         figure['data'][1].update(opacity=0.0)
#     if display_option == DISPLAY_MASKS:
#         figure['data'][1].update(opacity=0.3)

#     return figure


@callback(
    Output("canvas", "figure"),
    Output("labeled-masks", "data"),
    Input("image-filepath", "value"),
)
def handle_image_filepath_selection(image_filepath):
    if not image_filepath:
        return {}, {}, {}

    image_fig = go.FigureWidget()
    image_fig.update_layout(autosize=False, width=1024, height=1024)

    image = read_image(image_filepath)
    img_trace = px.imshow(image).data[0]
    image_fig.add_trace(img_trace)

    masks = read_masks_for_image(image_filepath)
    masks_image = get_masks_img(masks, image)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]
    image_fig.add_trace(masks_trace)
    image_fig.data[1].update(opacity=0.0)

    image_fig.add_trace(px.imshow(image).data[0])
    image_fig.data[2].update(opacity=0.0)

    labeled_masks = read_labels_for_image(image_filepath)
    print(labeled_masks.head())

    assert len(masks) == len(labeled_masks), (
        "Labels do not correspond to the masks."
        "Probably you updated the masks."
        "Consider removing {}".format(get_labels_filepath(image_filepath))
    )

    print("read masks", len(labeled_masks))
    return image_fig, labeled_masks.to_dict()


if __name__ == "__main__":
    app.run(debug=True)
