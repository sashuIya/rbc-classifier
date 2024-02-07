import base64
import os

import cv2
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    dash_table,
    dcc,
    html,
    register_page,
)
from dash.exceptions import PreventUpdate
from PIL import Image
from skimage import measure

from consts import (
    # Labels (options of Y_COLUMN)
    LABEL_UNLABELED,
    LABEL_WRONG,
    LABELING_APPROVED,
    LABELING_AUTO,
    LABELING_MANUAL,
    # Labeling modes
    LABELING_MODE_COLUMN,
    MASK_ID_COLUMN,
    # Features metadata
    Y_COLUMN,
)
from dash_util import id_factory
from draw_util import MasksColorOptions, get_masked_crop, get_masks_img
from filepath_util import (
    get_classifier_model_filepaths,
    get_masks_features_filepath,
    read_image,
    read_images_metadata,
    read_labels_metadata,
    read_masks_features,
    read_masks_for_image,
    write_images_metadata,
    write_masks_features,
)
from mask_util import is_point_in_mask
from pages.widgets.image_selector import (
    get_image_filepath_options,
    image_selection_dropdown,
    is_completed,
)
from train_classifier import (
    classify,
    train_pipeline,
)
from utils.timing import timeit

CHECKBOX_COMPLETED = "Completed"

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

ALL_MASKS_RADIO_BUTTONS_PREFIX = "all-masks"
SELECTED_MASKS_RADIO_BUTTONS_PREFIX = "selected-masks"


def masks_display_option(display_option: str) -> MasksColorOptions:
    option_to_option = {
        DISPLAY_IMAGE: MasksColorOptions.NONE,
        DISPLAY_MASKS: MasksColorOptions.RANDOM,
        DISPLAY_LABELED_DATA: MasksColorOptions.BY_LABEL,
        DISPLAY_UNLABELED_DATA: MasksColorOptions.NONE,
    }
    return option_to_option[display_option]


LABELS = {
    LABEL_UNLABELED: {"color": np.array([204, 0, 255])},
    LABEL_WRONG: {"color": np.array([203, 255, 0])},
    "red blood cell": {"color": np.array([255, 0, 0])},
    "spheroid cell": {"color": np.array([0, 255, 102])},
    "echinocyte": {"color": np.array([0, 101, 255])},
}

id = id_factory("label-cells")
register_page(__name__, order=2)

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                    image_selection_dropdown(
                        id=id("image-filepath"), predicate_fn=is_completed
                    ),
                    dcc.Dropdown(
                        CLASSIFIER_MODEL_FILEPATHS,
                        CLASSIFIER_MODEL_FILEPATHS[0],
                        id=id("classifier-model"),
                    ),
                    dbc.Button(
                        "Train classifier on labeled data",
                        id=id("train-classifier-button"),
                        className="me-1",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Run classifier",
                        id=id("run-classifier-button"),
                        className="me-1",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Save masks",
                        color="warning",
                        id=id("save-labels-button"),
                        className="me-1",
                        n_clicks=0,
                    ),
                    dcc.Checklist([CHECKBOX_COMPLETED], id=id("completed-checkbox")),
                    dcc.RadioItems(
                        DISPLAY_OPTIONS, DISPLAY_LABELED_DATA, id=id("display-options")
                    ),
                    html.Div(id=id("clicked-pixel-coords")),
                    dash_table.DataTable(
                        id=id("labels-stats"),
                        columns=[
                            {"name": "Label", "id": "Label"},
                            {"name": "Count", "id": "Count"},
                        ],
                        data=[{"Label": "X", "Count": "Y"}],
                        # Set the width of the table
                        style_table={"width": "10%"},  # Adjust the percentage as needed
                    ),
                    html.Div(style={"padding": "20px"}),
                    dcc.RadioItems(
                        list(LABELS.keys()), "red blood cell", id=id("active-label")
                    ),
                ],
            )
        ),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Preview",
                    tab_id="preview",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id=id("canvas"))),
                                dbc.Col(id=id("selected-masks")),
                            ],
                            justify="between",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Manual labeling",
                    tab_id="manual-labeling",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(id=id("all-masks"), width="auto"),
                            ],
                        ),
                    ],
                ),
            ],
            id=id("tabs"),
            active_tab="preview",
        ),
        dcc.Store(id=id("labeled-masks")),
        dcc.Store(id=id("modified-labels")),
        dcc.Store(id=id("figure-store")),
    ],
    fluid=True,
)


def create_color_by_mask_id(labels_df):
    color_by_mask_id = dict()
    for _, row in labels_df.iterrows():
        mask_id, label = row[MASK_ID_COLUMN], row[Y_COLUMN]
        if label in [LABEL_UNLABELED, LABEL_WRONG]:
            continue
        color_by_mask_id[mask_id] = LABELS[label]["color"]

    return color_by_mask_id


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
    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
    _, crop_png = cv2.imencode(".png", bgr_crop)
    crop_base64 = base64.b64encode(crop_png).decode("utf-8")
    crop_html = html.Img(
        src=f"data:image/png;base64,{crop_base64}",
        style={"display": "block", "margin-bottom": "10px"},
        className="img-item",
    )

    return crop_html


def generate_labeled_mask_preview_info(
    image: np.array, mask: dict, label: str
) -> tuple[np.array, np.array, str, dict]:
    """Generates a preview of the given `mask` labeled with the given `label`.

    Args:
        image (np.array): Image to crop.
        mask (dict): A mask from SAM.
        label (str): label of the mask.

    Returns:
        tuple: (highlited crop, original crop, label, mask)
    """
    return (
        get_masked_crop(
            image,
            mask,
            xy_threshold=20,
            with_highlighting=True,
            color_mask=LABELS[label]["color"],
        ),
        get_masked_crop(
            image,
            mask,
            xy_threshold=20,
            with_highlighting=False,
            color_mask=LABELS[label]["color"],
        ),
        label,
        mask,
    )


def generate_crop_with_radio(
    id_prefix, crop_with_highlighting, original_crop, labels, label, mask
) -> dbc.Form:
    mask_id = mask["id"]
    return dbc.Form(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            "mask_id: {}, area: {}".format(mask_id, mask["area"]),
                            id={
                                "type": id(f"{id_prefix}-mask-id-div"),
                                "index": mask_id,
                            },
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
                            id={
                                "type": id(f"{id_prefix}-radio-item"),
                                "index": mask_id,
                            },
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


def generate_labeled_masks_previews(
    radio_buttons_prefix: str,
    labeled_mask_preview_infos: list[tuple[np.array, np.array, str, dict]],
) -> list[dbc.Form]:
    labels = list(LABELS.keys())

    image_radio_items = []
    for (
        crop_with_highlighting,
        original_crop,
        label,
        mask,
    ) in labeled_mask_preview_infos:
        image_radio_item = generate_crop_with_radio(
            radio_buttons_prefix,
            crop_with_highlighting,
            original_crop,
            labels,
            label,
            mask,
        )
        image_radio_items.append(image_radio_item)

    return image_radio_items


@timeit
def image_with_masks_figure_as_scatters(
    image_filepath: str, display_option: str, labels_df: pd.DataFrame
) -> go.Figure:
    image = read_image(image_filepath, with_alpha=False)
    fig = px.imshow(image, binary_string=True, binary_backend="jpg")
    fig.update_layout(autosize=False, width=1024, height=1024)

    masks = read_masks_for_image(image_filepath)

    assert len(masks) == labels_df.shape[0], (
        "Labels do not correspond to the masks."
        "Probably you updated the masks."
        "Consider removing {}".format(get_masks_features_filepath(image_filepath))
    )

    color_by_mask_id = create_color_by_mask_id(labels_df)
    for mask in masks:
        mask_id = mask["id"]
        if mask_id not in color_by_mask_id:
            continue

        mask_reformed = mask["segmentation"][:, :].astype("uint8")
        contours, _ = cv2.findContours(
            mask_reformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        def rgb2rgb(rgb):
            r, g, b = rgb
            return "rgb({}, {}, {})".format(r, g, b)

        for contour_index in range(len(contours)):
            contour = contours[contour_index].reshape(-1, 2)
            x, y = contour.T
            (x0, y0, _, _) = mask["bbox"]
            x += int(x0)
            y += int(y0)

            fig.add_scatter(
                x=x,
                y=y,
                name=mask_id,
                opacity=0.35,
                mode="markers",
                line=dict(color=rgb2rgb(color_by_mask_id[mask_id]), width=1),
                hoverinfo="none",
                fill="toself",
            )

    return fig


def image_with_masks_figure(
    image_filepath: str, display_option: str, labels_df: pd.DataFrame
) -> go.Figure:
    image = read_image(image_filepath, with_alpha=False)
    masks = read_masks_for_image(image_filepath)

    assert len(masks) == labels_df.shape[0], (
        "Labels do not correspond to the masks."
        "Probably you updated the masks."
        "Consider removing {}".format(get_masks_features_filepath(image_filepath))
    )

    color_by_mask_id = create_color_by_mask_id(labels_df)
    image = get_masks_img(
        masks,
        image,
        masks_color_option=masks_display_option(display_option),
        color_by_mask_id=color_by_mask_id,
    )

    trace = px.imshow(image).data[0]

    image_fig = go.Figure()
    image_fig.update_layout(autosize=False, width=1024, height=1024)
    image_fig.add_trace(trace)

    return image_fig


@callback(
    Output(id("labeled-masks"), "data", allow_duplicate=True),
    Input(id("modified-labels"), "data"),
    State(id("labeled-masks"), "data"),
    prevent_initial_call=True,
)
@timeit
def update_labeled_masks(modified_labels, labeled_masks: dict):
    if not modified_labels:
        raise PreventUpdate

    labeled_masks = pd.DataFrame(labeled_masks)

    for mask_id, new_label, new_labeling_mode in modified_labels:
        labeled_masks.loc[
            labeled_masks[MASK_ID_COLUMN] == mask_id,
            [Y_COLUMN, LABELING_MODE_COLUMN],
        ] = (new_label, new_labeling_mode)

    return labeled_masks.to_dict()


@callback(
    Output(id("modified-labels"), "data", allow_duplicate=True),
    Input(
        {"type": id(f"{ALL_MASKS_RADIO_BUTTONS_PREFIX}-radio-item"), "index": ALL},
        "value",
    ),
    State(
        {"type": id(f"{ALL_MASKS_RADIO_BUTTONS_PREFIX}-radio-item"), "index": ALL}, "id"
    ),
    State(id("labeled-masks"), "data"),
    prevent_initial_call=True,
    suppress_callback_exceptions=True,
)
@timeit
def handle_all_masks_radio_button_click(labels, ids, labeled_masks: dict):
    labeled_masks = pd.DataFrame(labeled_masks)
    modified_labels = []
    for label, id in zip(labels, ids):
        mask_id = id["index"]

        loc = labeled_masks.loc[
            labeled_masks[MASK_ID_COLUMN] == mask_id,
            [Y_COLUMN, LABELING_MODE_COLUMN],
        ]

        if loc[Y_COLUMN].values[0] != label:
            modified_labels.append((mask_id, label, LABELING_MANUAL))
            loc = (label, LABELING_MANUAL)

    return modified_labels


@callback(
    Output(id("modified-labels"), "data", allow_duplicate=True),
    Input(
        {"type": id(f"{SELECTED_MASKS_RADIO_BUTTONS_PREFIX}-radio-item"), "index": ALL},
        "value",
    ),
    State(
        {"type": id(f"{SELECTED_MASKS_RADIO_BUTTONS_PREFIX}-radio-item"), "index": ALL},
        "id",
    ),
    State(id("labeled-masks"), "data"),
    prevent_initial_call=True,
    suppress_callback_exceptions=True,
)
@timeit
def handle_selected_masks_radio_button_click(labels, ids, labeled_masks: dict):
    labeled_masks = pd.DataFrame(labeled_masks)
    modified_labels = []
    for label, id in zip(labels, ids):
        mask_id = id["index"]

        loc = labeled_masks.loc[
            labeled_masks[MASK_ID_COLUMN] == mask_id,
            [Y_COLUMN, LABELING_MODE_COLUMN],
        ]

        if loc[Y_COLUMN].values[0] != label:
            modified_labels.append((mask_id, label, LABELING_MANUAL))
            loc = (label, LABELING_MANUAL)

    return modified_labels


@callback(
    Output(id("clicked-pixel-coords"), "children"),
    Output(id("selected-masks"), "children"),
    Output(id("modified-labels"), "data"),
    Input(id("canvas"), "clickData"),
    State(id("active-label"), "value"),
    State(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
    prevent_initial_call=True,
)
@timeit
def handle_canvas_click(
    click_data, active_label: str, labeled_masks_dict: dict, image_filepath: str
):
    if not click_data:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_dict)

    point = click_data["points"][0]
    x, y = point["x"], point["y"]

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath)

    assert len(masks) == labeled_masks_df.shape[0]

    modified_labels = []
    clicked_crop_infos = []
    for mask in masks:
        mask_id = mask["id"]
        if not is_point_in_mask(x, y, mask):
            continue

        label = active_label if len(clicked_crop_infos) == 0 else LABEL_WRONG
        modified_labels.append((mask_id, label, LABELING_MANUAL))
        clicked_crop_infos.append(
            generate_labeled_mask_preview_info(image, mask, label)
        )

    return (
        html.H3("x: {}, y: {}".format(x, y)),
        generate_labeled_masks_previews(
            SELECTED_MASKS_RADIO_BUTTONS_PREFIX, clicked_crop_infos
        ),
        modified_labels,
    )


@callback(
    Output(id("save-labels-button"), "style"),
    Input(id("save-labels-button"), "n_clicks"),
    State(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
)
@timeit
def handle_save_labels_button_click(n_clicks, labeled_masks_df, image_filepath):
    if n_clicks == 0 or not labeled_masks_df:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_df)

    labeled_masks_df.loc[
        (labeled_masks_df[Y_COLUMN] != LABEL_UNLABELED)
        # & (labeled_masks[Y_COLUMN] != LABEL_WRONG)
        & (labeled_masks_df[LABELING_MODE_COLUMN] == LABELING_AUTO),
        LABELING_MODE_COLUMN,
    ] = LABELING_APPROVED

    masks_features = read_masks_features(image_filepath)
    labeled_masks_df.index = masks_features.index
    masks_features[labeled_masks_df.columns] = labeled_masks_df

    write_masks_features(masks_features, image_filepath)

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath)
    color_by_mask_id = create_color_by_mask_id(labeled_masks_df)
    image = get_masks_img(
        masks,
        image,
        masks_color_option=MasksColorOptions.BY_LABEL,
        color_by_mask_id=color_by_mask_id,
    )

    result_filepath_base = os.path.splitext(image_filepath)[0] + "_result"
    image_result_filepath = result_filepath_base + ".tif"
    assert image_result_filepath != image_filepath
    pillow_image = Image.fromarray(image.astype(np.uint8))
    pillow_image.save(image_result_filepath, compression="tiff_lzw")

    labels_metadata = read_labels_metadata()
    # Group by labels and sort results by the order that is listed in labels_metadata.csv
    common_labels = set(labeled_masks_df[Y_COLUMN]).intersection(
        labels_metadata["label"]
    )
    filtered_df_data = labeled_masks_df[labeled_masks_df[Y_COLUMN].isin(common_labels)]
    filtered_df_order = labels_metadata[labels_metadata["label"].isin(common_labels)]
    label_counts = filtered_df_data[Y_COLUMN].value_counts()
    ordered_label_counts = label_counts.loc[filtered_df_order["label"]].reset_index()
    ordered_label_counts.columns = ["Label", "Count"]

    ordered_label_counts.to_csv(
        result_filepath_base + "_label_counts.tsv", sep="\t", index=False
    )

    return {}


@callback(
    Output(id("labels-stats"), "data"),
    Input(id("labeled-masks"), "data"),
)
@timeit
def perform_stats_change(labeled_masks_dict):
    if not labeled_masks_dict:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_dict)
    label_counts = labeled_masks_df[Y_COLUMN].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]

    return label_counts.to_dict("records")


@callback(
    Output(id("canvas"), "figure"),
    Input(id("display-options"), "value"),
    Input(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
)
@timeit
def perform_canvas_change(display_option, labeled_masks_dict, image_filepath):
    if not image_filepath:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_dict)

    # ? Maybe use `image_with_masks_figure_as_scatters` instead?
    return image_with_masks_figure(image_filepath, display_option, labeled_masks_df)


@callback(
    Output(id("labeled-masks"), "data"),
    Output(id("completed-checkbox"), "value"),
    Output(id("all-masks"), "children"),
    Input(id("image-filepath"), "value"),
)
@timeit
def handle_image_filepath_selection(image_filepath):
    if not image_filepath:
        return {}, {}, {}

    print("Image filepath changed")

    labeled_masks_df = read_masks_features(image_filepath)
    print("read labeled_masks, shape", labeled_masks_df.shape)
    print(labeled_masks_df[[MASK_ID_COLUMN, Y_COLUMN, LABELING_MODE_COLUMN]].head())

    labeled_masks_df = labeled_masks_df[
        [MASK_ID_COLUMN, Y_COLUMN, LABELING_MODE_COLUMN]
    ]

    images_metadata = read_images_metadata()
    completed = (
        image_filepath in images_metadata["filepath"].values
        and images_metadata.loc[
            images_metadata["filepath"] == image_filepath, "completed"
        ].iloc[0]
    )

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath)
    assert len(masks) == labeled_masks_df.shape[0]

    crop_infos = []
    for (_, row), mask in zip(labeled_masks_df.iterrows(), masks):
        assert mask["id"] == row[MASK_ID_COLUMN]
        crop_infos.append(
            generate_labeled_mask_preview_info(image, mask, row[Y_COLUMN])
        )

    all_mask_previews = generate_labeled_masks_previews(
        ALL_MASKS_RADIO_BUTTONS_PREFIX, crop_infos
    )

    return (
        labeled_masks_df.to_dict(),
        [CHECKBOX_COMPLETED] if completed else [],
        all_mask_previews,
    )


@callback(
    Output(id("classifier-model"), "options"),
    Output(id("classifier-model"), "value"),
    Input(id("train-classifier-button"), "n_clicks"),
)
@timeit
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
@timeit
def handle_run_classifier_button(n_clicks, classifier_model_filepath, image_filepath):
    if not n_clicks or not image_filepath or not classifier_model_filepath:
        raise PreventUpdate

    labeled_masks = read_masks_features(image_filepath)
    predictions = classify(
        labeled_masks[labeled_masks[Y_COLUMN] == LABEL_UNLABELED],
        classifier_model_filepath,
    )

    labeled_masks.loc[labeled_masks[Y_COLUMN] == LABEL_UNLABELED, Y_COLUMN] = (
        predictions
    )

    return labeled_masks.to_dict()


@callback(
    Output(id("image-filepath"), "options"),
    Input(id("completed-checkbox"), "value"),
    State(id("image-filepath"), "value"),
)
@timeit
def handle_completed_checkbox(selected_items, image_filepath):
    if not image_filepath:
        raise PreventUpdate

    df = read_images_metadata()
    if image_filepath not in df["filepath"].values:
        raise PreventUpdate

    df.loc[
        df["filepath"] == image_filepath,
        "completed",
    ] = (
        CHECKBOX_COMPLETED in selected_items
    )

    write_images_metadata(df)

    return get_image_filepath_options(predicate_fn=is_completed)
