from dash import Dash, html, dcc, callback, Output, Input, State, ctx, register_page
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_util import id_factory

import plotly.express as px
import plotly.graph_objects as go

import glob
import base64
import os

import pandas as pd
import numpy as np
import cv2

from sklearn.cluster import KMeans

from draw_util import get_masks_img, get_masked_crop, draw_height_and_scale
from filepath_util import (
    read_masks_for_image,
    read_image,
    get_rel_filepaths_from_subfolders,
    read_images_metadata,
    write_masks_features,
)
from mask_util import (
    is_point_in_mask,
    run_sam,
    save_masks,
    compute_resnet_features,
    compute_measure_features,
    construct_features_dataframe,
)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

METADATA_DF = read_images_metadata()
SAM_CHECKPOINTS_FOLDER = os.path.normpath("./model/sam/")


TIF_FILEPATHS = list(METADATA_DF["filepath"].unique())
SAM_CHECKPOINT_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=SAM_CHECKPOINTS_FOLDER, extension="pth"
)

HEIGHT_FULL = "full"
HEIGHT_VALUES = [HEIGHT_FULL, 1024, 2048]

USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION = "Use mesh size for masks file suffix"

id = id_factory("execute_sam")
register_page(__name__, order=1)

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H2(
                        children="Segment-Anything executor",
                        style={"textAlign": "center"},
                    ),
                    html.Div("Image filepath"),
                    dcc.Dropdown(
                        TIF_FILEPATHS, TIF_FILEPATHS[0], id=id("image-filepath")
                    ),
                    html.Div(style={"padding": "20px"}),
                    html.Div("Segment-Anything checkpoint filepath"),
                    dcc.Dropdown(
                        SAM_CHECKPOINT_FILEPATHS,
                        SAM_CHECKPOINT_FILEPATHS[0],
                        id=id("sam-checkpoint-filepath"),
                    ),
                    html.Div(style={"padding": "20px"}),
                    html.Div("Image height without scaling:", id=id("image-height")),
                    html.Div(style={"padding": "20px"}),
                    html.Button("Run SAM", id=id("run-sam-button"), n_clicks=0),
                    dcc.Checklist(
                        [USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION],
                        value=[USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION],
                        id=id("grid-size-as-suffix"),
                    ),
                    html.Button(
                        "Compute and save features",
                        id=id("compute-features-button"),
                        n_clicks=0,
                    ),
                    dcc.Loading(
                        id=id("loading-sam"),
                        type="default",
                        children=html.Div(id=id("loading-sam-output")),
                    ),
                    html.Div(style={"padding": "20px"}),
                    dcc.Slider(5, 300, 5, id=id("grid-size"), value=5),
                    html.Div(style={"padding": "20px"}),
                    html.Div("crop-n-layers"),
                    dcc.RadioItems([0, 1, 2, 3], 0, id=id("crop-n-layers")),
                    dcc.Graph(id=id("canvas")),
                    html.H4("Masks preview"),
                    dcc.Graph(id=id("masks-preview")),
                ],
                # width=22,
            )
        ),
    ],
    fluid=True,
)


def suffix_for_masks_file(points_per_side):
    return "__{}".format(points_per_side)


@callback(
    Output(id("image-height"), "children"),
    Output(id("canvas"), "figure"),
    Input(id("image-filepath"), "value"),
    Input(id("grid-size"), "value"),
)
def handle_image_filepath_selection(image_filepath, grid_size):
    if not image_filepath:
        return {}, {}

    image_fig = go.FigureWidget()
    image_fig.update_layout(autosize=False, width=1024, height=1024)

    image = read_image(image_filepath)
    img_trace = px.imshow(image).data[0]
    image_fig.add_trace(img_trace)

    for i in range(grid_size + 1):
        y = [i * (image.shape[0] / grid_size)] * 2
        x = [0, image.shape[1]]
        image_fig.add_scatter(x=x, y=y, line=dict(color="#ffe476"))

    for i in range(grid_size + 1):
        x = [i * (image.shape[1] / grid_size)] * 2
        y = [0, image.shape[0]]
        image_fig.add_scatter(x=x, y=y, line=dict(color="#ffe476"))

    metadata = read_images_metadata()
    height, scale_x0, scale_x1, micrometers = metadata.loc[
        metadata["filepath"] == image_filepath,
        ["height", "scale_x0", "scale_x1", "micrometers"],
    ].values[0]

    draw_height_and_scale(
        image_fig, image.shape, height, scale_x0, scale_x1, micrometers
    )

    return ["Image height without scaling: {}".format(height)], image_fig


def figure_widget_for_image_and_masks(image, masks):
    image_and_masks_fig = go.FigureWidget()
    image_and_masks_fig.update_layout(autosize=False, width=1024, height=1024)

    img_trace = px.imshow(image).data[0]
    image_and_masks_fig.add_trace(img_trace)

    masks_image = get_masks_img(masks, image)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]
    image_and_masks_fig.add_trace(masks_trace)
    image_and_masks_fig.data[1].update(opacity=0.35)

    return image_and_masks_fig


@callback(
    Output(id("masks-preview"), "figure"),
    Output(id("loading-sam-output"), "children"),
    Input(id("run-sam-button"), "n_clicks"),
    State(id("image-filepath"), "value"),
    State(id("sam-checkpoint-filepath"), "value"),
    State(id("grid-size"), "value"),
    State(id("crop-n-layers"), "value"),
    State(id("grid-size-as-suffix"), "value"),
)
def handle_run_sam_button_click(
    n_clicks,
    image_filepath,
    sam_checkpoint_filepath,
    points_per_side,
    crop_n_layers,
    points_per_side_as_suffix: list,
):
    if ctx.triggered_id != id("run-sam-button"):
        raise PreventUpdate

    metadata = read_images_metadata()
    image_height_adjustment = metadata.loc[
        metadata["filepath"] == image_filepath, "height"
    ].values[0]
    image = read_image(image_filepath)
    masks = run_sam(
        image[:image_height_adjustment, :, :],
        sam_checkpoint_filepath,
        crop_n_layers=crop_n_layers,
        points_per_side=points_per_side,
    )

    suffix = ""
    if USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION in points_per_side_as_suffix:
        suffix = suffix_for_masks_file(points_per_side)
    save_masks(masks, image_filepath, suffix=suffix)

    return figure_widget_for_image_and_masks(image, masks), {}


@callback(
    Output(id("compute-features-button"), "style"),
    Input(id("compute-features-button"), "n_clicks"),
    State(id("image-filepath"), "value"),
    State(id("grid-size"), "value"),
    State(id("grid-size-as-suffix"), "value"),
    prevent_initial_call=True,
)
def handle_compute_features_button_click(
    n_clicks, image_filepath, points_per_side, points_per_side_as_suffix: list
):
    if not n_clicks or not image_filepath:
        raise PreventUpdate

    suffix = ""
    if USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION in points_per_side_as_suffix:
        suffix = suffix_for_masks_file(points_per_side)

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath, suffix=suffix)

    resnet_features, resnet_columns = compute_resnet_features(masks, image)
    measure_features, measure_columns = compute_measure_features(masks, image_filepath)

    all_features = np.concatenate([resnet_features, measure_features], axis=1)
    all_columns = resnet_columns + measure_columns
    features_df = construct_features_dataframe(
        image_filepath, masks, all_features, all_columns
    )
    write_masks_features(features_df, image_filepath, suffix=suffix)

    return {}
