import os
import warnings

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, register_page
from dash.exceptions import PreventUpdate
from skimage import measure
from tqdm import tqdm

from dash_util import id_factory
from draw_util import MasksColorOptions, draw_height_and_scale, get_masks_img
from filepath_util import (
    get_rel_filepaths_from_subfolders,
    read_image,
    read_images_metadata,
    read_masks_for_image,
    write_masks_features,
)
from mask_util import (
    compute_measure_features,
    compute_resnet_features,
    construct_features_dataframe,
    run_sam,
    save_masks,
)
from pages.widgets.image_selector import (
    image_selection_dropdown,
    is_completed,
    is_measured,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

SAM_CHECKPOINTS_FOLDER = os.path.normpath("./model/sam/")
SAM_CHECKPOINT_FILEPATHS = get_rel_filepaths_from_subfolders(
    folder_path=SAM_CHECKPOINTS_FOLDER, extension="pth"
)

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
                    dbc.Button(
                        "Run SAM",
                        color="primary",
                        className="me-1",
                        id=id("run-sam-button"),
                        n_clicks=0,
                    ),
                    dcc.Checklist(
                        [USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION],
                        value=[USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION],
                        id=id("grid-size-as-suffix"),
                    ),
                    dbc.Button(
                        "Compute and save features",
                        color="primary",
                        className="me-1",
                        id=id("compute-features-button"),
                        n_clicks=0,
                    ),
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
                    html.Div(style={"padding": "20px"}),
                    dcc.Slider(5, 300, 5, id=id("grid-size"), value=5),
                    html.Div(style={"padding": "20px"}),
                    html.Div("crop-n-layers"),
                    dcc.RadioItems([0, 1, 2, 3], 0, id=id("crop-n-layers")),
                    dcc.Graph(id=id("canvas")),
                    html.H4("Masks preview"),
                    dcc.Graph(id=id("masks-preview")),
                ],
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
    Output(id("execute-alert"), "is_open"),
    Input(id("image-filepath"), "value"),
    Input(id("grid-size"), "value"),
)
def handle_image_filepath_selection(image_filepath, grid_size):
    if not image_filepath:
        return {}, {}, False

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

    return ["Image height without scaling: {}".format(height)], image_fig, False


def figure_widget_for_image_and_masks(image, masks):
    masks_image = get_masks_img(masks, image, MasksColorOptions.RANDOM)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]

    image_and_masks_fig = go.FigureWidget()
    image_and_masks_fig.update_layout(autosize=False, width=1024, height=1024)
    image_and_masks_fig.add_trace(masks_trace)

    return image_and_masks_fig


def run_sam_for_image(
    metadata,
    image_filepath,
    sam_checkpoint_filepath,
    crop_n_layers,
    points_per_side,
    points_per_side_as_suffix: list,
):
    image_height_adjustment, scale_x0, scale_x1, micrometers = metadata.loc[
        metadata["filepath"] == image_filepath,
        ["height", "scale_x0", "scale_x1", "micrometers"],
    ].values[0]
    image_height_adjustment = int(image_height_adjustment)
    image = read_image(image_filepath, with_alpha=False)
    masks = run_sam(
        image[:image_height_adjustment, :, :],
        sam_checkpoint_filepath,
        crop_n_layers=crop_n_layers,
        points_per_side=points_per_side,
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

    suffix = ""
    if USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION in points_per_side_as_suffix:
        suffix = suffix_for_masks_file(points_per_side)
    save_masks(masks, image_filepath, suffix=suffix)

    return image, masks


@callback(
    Output(id("masks-preview"), "figure"),
    Output(id("loading-sam-output"), "children"),
    Output(id("execute-alert"), "is_open", allow_duplicate=True),
    Input(id("run-sam-button"), "n_clicks"),
    State(id("image-filepath"), "value"),
    State(id("sam-checkpoint-filepath"), "value"),
    State(id("grid-size"), "value"),
    State(id("crop-n-layers"), "value"),
    State(id("grid-size-as-suffix"), "value"),
    prevent_initial_call=True,
)
def handle_run_sam_button_click(
    n_clicks,
    image_filepath,
    sam_checkpoint_filepath,
    points_per_side,
    crop_n_layers,
    points_per_side_as_suffix: list,
):
    if not n_clicks or not image_filepath:
        raise PreventUpdate

    metadata = read_images_metadata()
    if is_completed(metadata, image_filepath):
        return {}, {}, True

    image, masks = run_sam_for_image(
        metadata=metadata,
        image_filepath=image_filepath,
        sam_checkpoint_filepath=sam_checkpoint_filepath,
        crop_n_layers=crop_n_layers,
        points_per_side=points_per_side,
        points_per_side_as_suffix=points_per_side_as_suffix,
    )

    return figure_widget_for_image_and_masks(image, masks), {}, False


def compute_and_write_masks_features(
    image_filepath, image, masks, points_per_side, points_per_side_as_suffix
):
    resnet_features, resnet_columns = compute_resnet_features(masks, image)
    measure_features, measure_columns = compute_measure_features(masks, image_filepath)

    all_features = np.concatenate([resnet_features, measure_features], axis=1)
    all_columns = resnet_columns + measure_columns
    features_df = construct_features_dataframe(
        image_filepath, masks, all_features, all_columns
    )
    suffix = ""
    if USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION in points_per_side_as_suffix:
        suffix = suffix_for_masks_file(points_per_side)
    write_masks_features(features_df, image_filepath, suffix=suffix)


@callback(
    Output(id("execute-alert"), "is_open", allow_duplicate=True),
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

    metadata = read_images_metadata()
    if is_completed(metadata, image_filepath):
        return True

    suffix = ""
    if USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION in points_per_side_as_suffix:
        suffix = suffix_for_masks_file(points_per_side)

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath, suffix=suffix)

    compute_and_write_masks_features(
        image_filepath, image, masks, points_per_side, points_per_side_as_suffix
    )

    return False


@callback(
    Output(id("run-for-all-button"), "style"),
    Input(id("run-for-all-button"), "n_clicks"),
    State(id("sam-checkpoint-filepath"), "value"),
    State(id("grid-size"), "value"),
    State(id("crop-n-layers"), "value"),
    State(id("grid-size-as-suffix"), "value"),
)
def handle_run_for_all_button(
    n_clicks,
    sam_checkpoint_filepath,
    points_per_side,
    crop_n_layers,
    points_per_side_as_suffix: list,
):
    if not n_clicks:
        raise PreventUpdate

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

    suffix = ""
    if USE_MESH_SIZE_FOR_MASKS_FILE_SUFFIX_OPTION in points_per_side_as_suffix:
        suffix = suffix_for_masks_file(points_per_side)

    for image_filepath in tqdm(image_filepaths, desc="Running pipeline"):
        print("Executing for", image_filepath)

        run_sam_for_image(
            metadata=metadata,
            image_filepath=image_filepath,
            sam_checkpoint_filepath=sam_checkpoint_filepath,
            crop_n_layers=crop_n_layers,
            points_per_side=points_per_side,
            points_per_side_as_suffix=points_per_side_as_suffix,
        )

        image = read_image(image_filepath)
        masks = read_masks_for_image(image_filepath, suffix=suffix)
        compute_and_write_masks_features(
            image_filepath, image, masks, points_per_side, points_per_side_as_suffix
        )

    return {}
