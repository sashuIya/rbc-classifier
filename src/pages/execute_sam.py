import json
import os
import warnings
from datetime import datetime

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, register_page
from dash.exceptions import PreventUpdate
from skimage import measure
from tqdm import tqdm

from src.common.consts import DEFAULT_SAM_CONFIG, SAM_CHECKPOINTS_FOLDER
from src.common.filepath_util import (
    ImageDataWriter,
    get_rel_filepaths_from_subfolders,
    load_lastest_sam_config,
    read_image,
    read_images_metadata,
    save_sam_config,
)
from src.pages.widgets.image_selector import (
    image_selection_dropdown,
    is_completed,
    is_measured,
)
from src.utils.dash_util import id_factory
from src.utils.draw_util import MasksColorOptions, draw_height_and_scale, get_masks_img
from src.utils.mask_util import (
    compute_measure_features,
    compute_resnet_features,
    construct_features_dataframe,
    run_sam,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

SAM_CHECKPOINT_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=SAM_CHECKPOINTS_FOLDER, extension="pth"
)
SAM_CHECKPOINT_FILEPATHS = [str(f) for f in SAM_CHECKPOINT_FILEPATHS]

HEIGHT_FULL = "full"
HEIGHT_VALUES = [HEIGHT_FULL, 1024, 2048]

USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION = "Use mesh size for masks file suffix"

id = id_factory("execute_sam")
register_page(__name__, order=1)


def is_measured_and_not_completed(df, filepath):
    return (
        filepath in df["filepath"].values
        and df.loc[df["filepath"] == filepath, "measured"].iloc[0]
        and not df.loc[df["filepath"] == filepath, "completed"].iloc[0]
    )


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
                    image_selection_dropdown(
                        id=id("image-filepath"),
                        predicate_fn=is_completed,
                    ),
                    html.Div(style={"padding": "10px"}),
                    html.Div("Segment-Anything checkpoint filepath"),
                    dcc.Dropdown(
                        SAM_CHECKPOINT_FILEPATHS,
                        SAM_CHECKPOINT_FILEPATHS[0],
                        id=id("sam-checkpoint-filepath"),
                    ),
                    html.Div(style={"padding": "10px"}),
                    html.Div("Image height without scaling:", id=id("image-height")),
                    html.Div(style={"padding": "10px"}),
                ]
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Textarea(
                        id=id("sam-config"),
                        value=json.dumps(load_lastest_sam_config(), indent=4),
                        style={"width": "100%", "height": "250px"},
                    ),
                    width=2,
                ),
                dbc.Col(
                    dcc.Markdown(
                        id="default-config",
                        children=[
                            f"**Default Configuration:**\n```json\n{json.dumps(DEFAULT_SAM_CONFIG, indent=4)}\n```"
                        ],
                    ),
                    width=2,
                ),
            ],
            justify="start",
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Div(style={"padding": "10px"}),
                    dbc.Button(
                        "Run SAM",
                        color="primary",
                        className="me-1",
                        id=id("run-sam-button"),
                        n_clicks=0,
                    ),
                    html.Div(style={"padding": "10px"}),
                    dbc.Alert(
                        "Cannot execute for already completed image!",
                        id=id("execute-alert"),
                        color="danger",
                        is_open=False,
                    ),
                    dbc.Button(
                        "Run SAM and compute features for all not-completed",
                        color="primary",
                        className="me-1",
                        id=id("run-for-all-button"),
                        n_clicks=0,
                    ),
                    dcc.Loading(
                        id=id("loading-sam"),
                        type="default",
                        children=html.Div(id=id("loading-sam-output")),
                    ),
                    html.Div(style={"padding": "10px"}),
                    dcc.Slider(1, 10, 1, id=id("crops-per-side"), value=1),
                    dcc.Slider(5, 300, 5, id=id("grid-size"), value=5),
                    html.Div(style={"padding": "10px"}),
                    dbc.Label("crop-n-layers"),
                    dbc.RadioItems([0, 1, 2, 3], 0, id=id("crop-n-layers")),
                ]
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id=id("canvas")),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(id=id("masks-preview")),
                    width=6,
                ),
            ]
        ),
    ],
    fluid=True,
)


def add_grid(fig: go.FigureWidget, width: int, height: int, grid_size: int, color: int):
    for i in range(grid_size + 1):
        y = [i * (height / grid_size)] * 2
        x = [0, width]
        fig.add_scatter(x=x, y=y, line=dict(color=color))

    for i in range(grid_size + 1):
        x = [i * (width / grid_size)] * 2
        y = [0, height]
        fig.add_scatter(x=x, y=y, line=dict(color=color))


@callback(
    Output(id("image-height"), "children"),
    Output(id("canvas"), "figure"),
    Output(id("execute-alert"), "is_open"),
    Input(id("image-filepath"), "value"),
    Input(id("crops-per-side"), "value"),
    Input(id("grid-size"), "value"),
)
def handle_image_filepath_selection(image_filepath, crops_size, grid_size):
    if not image_filepath:
        return {}, {}, False

    image_fig = go.FigureWidget()
    image_fig.update_layout(autosize=False, width=1024, height=1024)

    image = read_image(image_filepath)
    img_trace = px.imshow(image).data[0]
    image_fig.add_trace(img_trace)

    metadata = read_images_metadata()
    height, scale_x0, scale_x1, micrometers = metadata.loc[
        metadata["filepath"] == image_filepath,
        ["height", "scale_x0", "scale_x1", "micrometers"],
    ].values[0]

    add_grid(image_fig, image.shape[1], height, grid_size * crops_size, "#ffe476")
    add_grid(image_fig, image.shape[1], height, crops_size, "#03cea4")
    draw_height_and_scale(
        image_fig, image.shape, height, scale_x0, scale_x1, micrometers
    )

    return ["Image height without scaling: {}".format(height)], image_fig, False


def figure_widget_for_image_and_masks(image, masks):
    masks_image = get_masks_img(masks, image, MasksColorOptions.RANDOM)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]

    image_and_masks_fig = go.FigureWidget()
    image_and_masks_fig.update_layout(autosize=False, width=1024, height=1024)
    image_and_masks_fig.add_trace(masks_trace)

    return image_and_masks_fig


def image_width(image):
    return image.shape[1]


def run_sam_for_image(
    metadata,
    image_filepath,
    sam_checkpoint_filepath,
    crop_n_layers,
    crops_per_side,
    points_per_crop,
    sam_config: dict,
):
    height, scale_x0, scale_x1, micrometers = metadata.loc[
        metadata["filepath"] == image_filepath,
        ["height", "scale_x0", "scale_x1", "micrometers"],
    ].values[0]
    height = int(height)
    image = read_image(image_filepath, with_alpha=False)
    width = image_width(image)
    masks = run_sam(
        image[: height // crops_per_side, : width // crops_per_side, :],
        sam_checkpoint_filepath,
        crop_n_layers=crop_n_layers,
        points_per_crop=points_per_crop,
        sam_config=sam_config,
    )

    def area_predicate(mask):
        HOLE_SIZE_MICROMETERS_SQ = 4.0  # μm^2
        pixels_sq_to_microm_sq_coeff = (micrometers / (scale_x1 - scale_x0)) ** 2
        return pixels_sq_to_microm_sq_coeff * mask["area"] > HOLE_SIZE_MICROMETERS_SQ

    masks = [mask for mask in masks if area_predicate(mask)]
    print("Number of masks after filtering by area: {}".format(len(masks)))

    def connected_components_predicate(mask):
        _, labels_num = measure.label(
            mask["segmentation"], return_num=True, connectivity=2
        )
        return labels_num == 1

    masks = [mask for mask in masks if connected_components_predicate(mask)]
    print("Number of masks after filtering by connectivity: {}".format(len(masks)))

    sorted_masks = sorted(masks, key=(lambda x: x["area"]))
    for i, mask in enumerate(sorted_masks):
        mask["id"] = i

    return image, sorted_masks


def compute_masks_features(image_filepath, image, masks) -> pd.DataFrame:
    resnet_features, resnet_columns = compute_resnet_features(masks, image)
    measure_features, measure_columns = compute_measure_features(masks, image_filepath)

    all_features = np.concatenate([resnet_features, measure_features], axis=1)
    all_columns = resnet_columns + measure_columns
    features_df = construct_features_dataframe(
        image_filepath, masks, all_features, all_columns
    )

    return features_df


@callback(
    Output(id("masks-preview"), "figure"),
    Output(id("loading-sam-output"), "children"),
    Output(id("execute-alert"), "is_open", allow_duplicate=True),
    Input(id("run-sam-button"), "n_clicks"),
    State(id("image-filepath"), "value"),
    State(id("sam-checkpoint-filepath"), "value"),
    State(id("crops-per-side"), "value"),
    State(id("grid-size"), "value"),
    State(id("crop-n-layers"), "value"),
    State(id("sam-config"), "value"),
    prevent_initial_call=True,
)
def handle_run_sam_button_click(
    n_clicks,
    image_filepath,
    sam_checkpoint_filepath,
    crops_per_side,
    points_per_crop,
    crop_n_layers,
    sam_config,
):
    if not n_clicks or not image_filepath:
        raise PreventUpdate

    sam_config = {**json.loads(sam_config)}
    save_sam_config(sam_config)

    metadata = read_images_metadata()
    if is_completed(metadata, image_filepath):
        return {}, {}, True

    timestamp = datetime.now()
    filename_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    image, masks = run_sam_for_image(
        metadata=metadata,
        image_filepath=image_filepath,
        sam_checkpoint_filepath=sam_checkpoint_filepath,
        crop_n_layers=crop_n_layers,
        crops_per_side=crops_per_side,
        points_per_crop=points_per_crop,
        sam_config=sam_config,
    )
    features_df = compute_masks_features(image_filepath, image, masks)

    image_data_writer = ImageDataWriter(image_filepath)
    image_data_writer.write_masks(masks, filename_timestamp)
    image_data_writer.write_masks_features(features_df, filename_timestamp)

    return figure_widget_for_image_and_masks(image, masks), {}, False


@callback(
    Output(id("run-for-all-button"), "style"),
    Input(id("run-for-all-button"), "n_clicks"),
    State(id("sam-checkpoint-filepath"), "value"),
    State(id("crops-per-side"), "value"),
    State(id("grid-size"), "value"),
    State(id("crop-n-layers"), "value"),
    State(id("sam-config"), "value"),
)
def handle_run_for_all_button(
    n_clicks,
    sam_checkpoint_filepath,
    crops_per_side,
    points_per_crop,
    crop_n_layers,
    sam_config,
):
    if not n_clicks:
        raise PreventUpdate

    sam_config = {**json.loads(sam_config)}
    save_sam_config(sam_config)

    metadata = read_images_metadata()
    image_filepaths = []
    skipping = []
    for filepath in metadata["filepath"]:
        if is_measured(metadata, filepath) and not is_completed(metadata, filepath):
            image_filepaths.append(filepath)
        else:
            skipping.append(filepath)

    print("skipping", skipping)
    print("running for", image_filepaths)

    if n_clicks == 1:
        return {}

    timestamp = datetime.now()
    filename_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    for image_filepath in tqdm(image_filepaths, desc="Running pipeline"):
        print("Executing for", image_filepath)

        image, masks = run_sam_for_image(
            metadata=metadata,
            image_filepath=image_filepath,
            sam_checkpoint_filepath=sam_checkpoint_filepath,
            crop_n_layers=crop_n_layers,
            crops_per_side=crops_per_side,
            points_per_crop=points_per_crop,
            sam_config=sam_config,
        )
        features_df = compute_masks_features(image_filepath, image, masks)

        image_data_writer = ImageDataWriter(image_filepath)
        image_data_writer.write_masks(masks, filename_timestamp)
        image_data_writer.write_masks_features(features_df, filename_timestamp)

    return {}
