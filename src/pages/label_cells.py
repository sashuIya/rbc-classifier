import base64
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
from typing import List

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
from dash_extensions import EventListener
from PIL import Image

from src.common.consts import (
    CONFIDENCE_COLUMN,
    # Labels (options of Y_COLUMN)
    LABEL_UNLABELED,
    LABEL_WRONG,
    LABELING_APPROVED,
    LABELING_AUTO,
    LABELING_MANUAL,
    # Labeling modes
    LABELING_MODE_COLUMN,
    MASK_ID_COLUMN,
    RAW_IMAGES_DIR,
    # Features metadata
    Y_COLUMN,
)
from src.common.filepath_util import (
    EmbedderMetadata,
    ImageDataReader,
    ImageDataWriter,
    LabelsMetadata,
    get_classifier_model_filepaths,
    read_images_metadata,
    write_images_metadata,
)
from src.models.train_classifier import (
    classify,
    train_pipeline,
)
from src.pages.widgets.image_selector import (
    get_image_filepath_options,
    image_selection_dropdown,
    is_completed,
)
from src.utils.dash_util import id_factory
from src.utils.draw_util import MasksColorOptions, get_masked_crop, get_masks_img
from src.utils.mask_util import is_point_in_mask
from src.utils.timing import timeit

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


@dataclass
class LabeledMaskPreviewInfo:
    highlighted_crop: np.ndarray
    original_crop: np.ndarray
    label: str
    mask: dict
    confidence_score: float


def is_valid_masks_option(masks_option: str) -> bool:
    return masks_option is not None and masks_option != "None"


def masks_display_option(display_option: str) -> MasksColorOptions:
    option_to_option = {
        DISPLAY_IMAGE: MasksColorOptions.NONE,
        DISPLAY_MASKS: MasksColorOptions.RANDOM,
        DISPLAY_LABELED_DATA: MasksColorOptions.BY_LABEL,
        DISPLAY_UNLABELED_DATA: MasksColorOptions.NONE,
    }
    return option_to_option[display_option]


def construct_embedder_metadata_container(embedder_filepath):
    metadata_series = EmbedderMetadata().load_embedder_metadata(embedder_filepath)
    if metadata_series is None:
        return ""

    def replace_tabs(x):
        if not isinstance(x, str):
            return x
        if "\t" not in x:
            return x

        return "\n- " + x.replace("\t", "\n- ")

    metadata_dict = metadata_series.to_dict()
    metadata_dict = {key: replace_tabs(value) for key, value in metadata_dict.items()}

    formatted_dict = "\n".join(
        [f"**{key}:** {metadata_dict[key]}\n" for key in metadata_series.index]
    )

    return formatted_dict


LABELS_METADATA = LabelsMetadata()
DEFAULT_LABEL = (
    "red blood cell"
    if "red blood cell" in LABELS_METADATA.get_list_of_labels()
    else LABEL_WRONG
)

id = id_factory("label-cells")
register_page(__name__, order=2)

CLASSIFIER_MODEL_FILEPATHS = get_classifier_model_filepaths(as_str=True)
if not CLASSIFIER_MODEL_FILEPATHS:
    CLASSIFIER_MODEL_FILEPATHS = ["none"]

event = {"event": "keydown", "props": ["key"]}

layout = dbc.Container(
    [
        EventListener(
            events=[event],
            logging=True,
            id=id("el"),
        ),
        html.H1(children="Labeling tool", style={"textAlign": "center"}),
        dbc.Row(
            [
                dbc.Col(html.Label("Image filepath:"), width=1),
                dbc.Col(
                    image_selection_dropdown(
                        id=id("image-filepath"), predicate_fn=is_completed
                    ),
                    width=11,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.Label("Classifier:"), width=1),
                dbc.Col(
                    dcc.Dropdown(
                        CLASSIFIER_MODEL_FILEPATHS,
                        CLASSIFIER_MODEL_FILEPATHS[0],
                        id=id("classifier-model"),
                    ),
                    width=11,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.Label("Masks set:"), width=1),
                dbc.Col(
                    dcc.Dropdown([], None, id=id("masks-options")),
                    width=11,
                ),
            ]
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
            "Save labels",
            color="warning",
            id=id("save-labels-button"),
            className="me-1",
            n_clicks=0,
        ),
        dbc.Button(
            "Reset labels",
            color="danger",
            id=id("reset-labels-button"),
            className="me-1",
            n_clicks=0,
        ),
        html.Div(style={"padding": "10px"}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Checklist(
                            [CHECKBOX_COMPLETED],
                            id=id("completed-checkbox"),
                            switch=True,
                        ),
                        html.Div(style={"padding": "10px"}),
                        dbc.Label("Clicked pixel:"),
                        html.Div("X, Y", id=id("clicked-pixel-coords")),
                    ],
                    width=1,
                ),
                dbc.Col(
                    [
                        dbc.Label("Selected label:"),
                        dbc.RadioItems(
                            LABELS_METADATA.get_list_of_labels(),
                            DEFAULT_LABEL,
                            id=id("active-label"),
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("Stats:"),
                        dash_table.DataTable(
                            id=id("labels-stats"),
                            columns=[
                                {"name": "Label", "id": "Label"},
                                {"name": "Count", "id": "Count"},
                            ],
                            style_cell_conditional=[
                                {"if": {"column_id": "Label"}, "textAlign": "left"}
                            ],
                            data=[{"Label": "X", "Count": "Y"}],
                            # Set the width of the table
                            style_table={
                                "width": "10%"
                            },  # Adjust the percentage as needed
                            style_as_list_view=True,
                        ),
                    ],
                ),
                dbc.Col(
                    [
                        dcc.Markdown(
                            id=id("embedder-metadata-container"),
                            children=construct_embedder_metadata_container(
                                CLASSIFIER_MODEL_FILEPATHS[0]
                            ),
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("Display options:"),
                        dbc.RadioItems(
                            DISPLAY_OPTIONS,
                            DISPLAY_LABELED_DATA,
                            id=id("display-options"),
                        ),
                    ]
                ),
            ]
        ),
        html.Div(style={"padding": "10px"}),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Preview",
                    tab_id="preview",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id=id("canvas"), config={"scrollZoom": True}
                                    )
                                ),
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
    ],
    fluid=True,
)


def get_confidence_score(labels_df: pd.DataFrame, mask_id: int) -> float:
    if CONFIDENCE_COLUMN not in labels_df.columns:
        return 0.0

    return labels_df[labels_df[MASK_ID_COLUMN] == mask_id][CONFIDENCE_COLUMN].values[0]


def create_color_by_mask_id(labels_df):
    color_by_mask_id = dict()
    for _, row in labels_df.iterrows():
        mask_id, label = row[MASK_ID_COLUMN], row[Y_COLUMN]
        if label in [LABEL_UNLABELED, LABEL_WRONG]:
            continue
        color_by_mask_id[mask_id] = LABELS_METADATA.get_color_by_label(label)

    return color_by_mask_id


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
    image: np.array, mask: dict, label: str, confidence_score: float
) -> LabeledMaskPreviewInfo:
    """Generates a preview of the given `mask` labeled with the given `label`.

    Args:
        image (np.array): Image to crop.
        mask (dict): A mask from SAM.
        label (str): label of the mask.
        confidence_score (float): Confidence score of the label.

    Returns:
        LabeledMaskPreview: An instance containing (highlighted_crop, original_crop, label, mask, confidence_score).
    """
    highlighted_crop = get_masked_crop(
        image,
        mask,
        xy_threshold=20,
        with_highlighting=True,
        color_mask=LABELS_METADATA.get_color_by_label(label),
    )
    original_crop = get_masked_crop(
        image,
        mask,
        xy_threshold=20,
        with_highlighting=False,
        color_mask=LABELS_METADATA.get_color_by_label(label),
    )
    return LabeledMaskPreviewInfo(
        highlighted_crop, original_crop, label, mask, confidence_score
    )


def generate_crop_with_radio(
    id_prefix: str,
    labeled_mask_preview_info: LabeledMaskPreviewInfo,
    labels: List[str],
) -> dbc.Form:
    mask = labeled_mask_preview_info.mask
    mask_id = mask["id"]
    return dbc.Form(
        [
            dbc.Row(
                html.Div(
                    "mask_id: {}, area: {}, confidence_score: {}".format(
                        mask_id,
                        mask["area"],
                        labeled_mask_preview_info.confidence_score,
                    ),
                    id={
                        "type": id(f"{id_prefix}-mask-id-div"),
                        "index": mask_id,
                    },
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        crop_html(labeled_mask_preview_info.highlighted_crop),
                        className="horizontal-space",
                        style={
                            "width": "{}px".format(
                                labeled_mask_preview_info.highlighted_crop.shape[1]
                            ),
                            "margin-right": "20px",
                        },
                        width="auto",
                    ),
                    dbc.Col(
                        crop_html(labeled_mask_preview_info.original_crop),
                        className="horizontal-space",
                        style={
                            "width": "{}px".format(
                                labeled_mask_preview_info.original_crop.shape[1]
                            ),
                            "margin-right": "20px",
                        },
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            options=labels,
                            value=labeled_mask_preview_info.label,
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
        ],
        className="labeled-mask-card",
    )


def generate_labeled_masks_previews(
    radio_buttons_prefix: str,
    labeled_mask_preview_infos: List[LabeledMaskPreviewInfo],
) -> List[dbc.Form]:
    labels = list(LABELS_METADATA.get_list_of_labels())

    image_radio_items = []
    for info in labeled_mask_preview_infos:
        image_radio_item = generate_crop_with_radio(
            id_prefix=radio_buttons_prefix,
            labeled_mask_preview_info=info,
            labels=labels,
        )
        image_radio_items.append(image_radio_item)

    return image_radio_items


@timeit
def image_with_masks_figure_as_scatters(
    image_filepath: str,
    display_option: str,
    labels_df: pd.DataFrame,
    selected_masks_option: str,
) -> go.Figure:
    image_data_reader = ImageDataReader(image_filepath)
    image = image_data_reader.image
    fig = px.imshow(image, binary_string=True, binary_backend="jpg")
    fig.update_layout(autosize=False, width=1024, height=1024)

    masks = image_data_reader.read_masks(selected_masks_option)

    assert len(masks) == labels_df.shape[0], (
        "Labels do not correspond to the masks." "Probably you updated the masks."
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
    image_filepath: str,
    display_option: str,
    labels_df: pd.DataFrame,
    selected_masks_option: str,
) -> go.Figure:
    image_data_reader = ImageDataReader(image_filepath, with_alpha=False)
    image = image_data_reader.image
    masks = image_data_reader.read_masks(selected_masks_option)

    assert len(masks) == labels_df.shape[0], (
        "Labels do not correspond to the masks." "Probably you updated the masks."
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
    Output(id("active-label"), "value"),
    Input(id("el"), "n_events"),
    State(id("el"), "event"),
    State(id("active-label"), "options"),
)
def handle_key_press(n_events, e, options):
    if n_events == 0 or e is None:
        raise PreventUpdate

    def is_single_digit(s):
        if s is None:
            return False
        return s.isdigit() and len(s) == 1

    if not is_single_digit(e["key"]):
        raise PreventUpdate

    label_index = int(e["key"]) - 1
    if label_index < 0:
        label_index = 10

    if label_index >= len(options):
        raise PreventUpdate

    return options[label_index]


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

    return labeled_masks.to_dict("records")


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
    State(id("masks-options"), "value"),
    prevent_initial_call=True,
)
@timeit
def handle_canvas_click(
    click_data,
    active_label: str,
    labeled_masks_dict: dict,
    image_filepath: str,
    selected_masks_option: str,
):
    if not click_data:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_dict)

    point = click_data["points"][0]
    x, y = point["x"], point["y"]

    image_data_reader = ImageDataReader(image_filepath)
    image = image_data_reader.image
    masks = image_data_reader.read_masks(selected_masks_option)

    assert len(masks) == labeled_masks_df.shape[0]

    modified_labels = []
    clicked_crop_infos = []
    for mask in masks:
        mask_id = mask["id"]
        if not is_point_in_mask(x, y, mask):
            continue

        label = active_label if len(clicked_crop_infos) == 0 else LABEL_WRONG
        modified_labels.append((mask_id, label, LABELING_MANUAL))
        confidence_score = get_confidence_score(labeled_masks_df, mask_id)
        clicked_crop_infos.append(
            generate_labeled_mask_preview_info(image, mask, label, confidence_score)
        )

    return (
        html.Div("x: {}, y: {}".format(x, y)),
        generate_labeled_masks_previews(
            SELECTED_MASKS_RADIO_BUTTONS_PREFIX, clicked_crop_infos
        ),
        modified_labels,
    )


@callback(
    Output(id("labeled-masks"), "data", allow_duplicate=True),
    Input(id("reset-labels-button"), "n_clicks"),
    State(id("labeled-masks"), "data"),
    prevent_initial_call=True,
)
@timeit
def handle_reset_labels_button_click(n_clicks, labeled_masks_df):
    if n_clicks == 0 or not labeled_masks_df:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_df)
    labeled_masks_df[Y_COLUMN] = LABEL_UNLABELED
    labeled_masks_df[CONFIDENCE_COLUMN] = 0.0

    return labeled_masks_df.to_dict("records")


@callback(
    Input(id("save-labels-button"), "n_clicks"),
    State(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
    State(id("masks-options"), "value"),
)
@timeit
def handle_save_labels_button_click(
    n_clicks, labeled_masks_df, image_filepath, selected_masks_option
):
    if n_clicks == 0 or not labeled_masks_df or not selected_masks_option:
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_df)

    labeled_masks_df.loc[
        (labeled_masks_df[Y_COLUMN] != LABEL_UNLABELED)
        # & (labeled_masks[Y_COLUMN] != LABEL_WRONG)
        & (labeled_masks_df[LABELING_MODE_COLUMN] == LABELING_AUTO),
        LABELING_MODE_COLUMN,
    ] = LABELING_APPROVED

    image_data_reader = ImageDataReader(image_filepath)
    masks_features = image_data_reader.read_masks_features(selected_masks_option)
    labeled_masks_df.index = masks_features.index
    masks_features[labeled_masks_df.columns] = labeled_masks_df

    image_data_writer = ImageDataWriter(image_filepath)
    image_data_writer.write_masks_features(masks_features, selected_masks_option)

    masks = image_data_reader.read_masks(selected_masks_option)
    color_by_mask_id = create_color_by_mask_id(labeled_masks_df)
    image = get_masks_img(
        masks,
        image_data_reader.image,
        masks_color_option=MasksColorOptions.BY_LABEL,
        color_by_mask_id=color_by_mask_id,
    )

    # ! TODO: Save results under /processed folder.
    result_filepath_base = os.path.splitext(image_filepath)[0] + "_result"
    image_result_filepath = result_filepath_base + ".tif"
    assert image_result_filepath != image_filepath
    pillow_image = Image.fromarray(image.astype(np.uint8))
    pillow_image.save(image_result_filepath, compression="tiff_lzw")

    # Group by labels and sort results by the order that is listed in labels_metadata.csv
    common_labels = set(labeled_masks_df[Y_COLUMN]).intersection(
        LABELS_METADATA.get_list_of_labels()
    )
    filtered_df_data = labeled_masks_df[labeled_masks_df[Y_COLUMN].isin(common_labels)]

    label_counts = filtered_df_data[Y_COLUMN].value_counts()

    # Order the counts according to the list of labels in label_metadata.
    ordered_label_counts = (
        label_counts.reindex(LABELS_METADATA.get_list_of_labels())
        .dropna()
        .reset_index()
    )
    ordered_label_counts.columns = ["Label", "Count"]

    ordered_label_counts.to_csv(
        result_filepath_base + "_label_counts.tsv", sep="\t", index=False
    )


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

    # Sort label_counts according to the order in LABELS_METADATA.get_list_of_labels()
    ordered_labels = LABELS_METADATA.get_list_of_labels()
    label_counts.set_index("Label", inplace=True)
    label_counts = label_counts.reindex(ordered_labels).dropna().reset_index()

    return label_counts.to_dict("records")


@callback(
    Output(id("canvas"), "figure"),
    Input(id("display-options"), "value"),
    Input(id("labeled-masks"), "data"),
    State(id("image-filepath"), "value"),
    State(id("masks-options"), "value"),
)
@timeit
def perform_canvas_change(
    display_option, labeled_masks_dict, image_filepath, selected_masks_option
):
    if (
        not image_filepath
        or not labeled_masks_dict
        or not is_valid_masks_option(selected_masks_option)
    ):
        raise PreventUpdate

    labeled_masks_df = pd.DataFrame(labeled_masks_dict)

    # ? Maybe use `image_with_masks_figure_as_scatters` instead?
    return image_with_masks_figure(
        image_filepath, display_option, labeled_masks_df, selected_masks_option
    )


@callback(
    Output(id("labeled-masks"), "data"),
    Output(id("all-masks"), "children"),
    Input(id("masks-options"), "value"),
    State(id("image-filepath"), "value"),
)
@timeit
def handle_masks_filepath_selection(selected_masks_option, image_filepath):
    if not image_filepath or not is_valid_masks_option(selected_masks_option):
        raise PreventUpdate

    image_data_reader = ImageDataReader(image_filepath)
    print("selected_masks_option:", selected_masks_option)
    masks = image_data_reader.read_masks(selected_masks_option)
    labeled_masks_df = image_data_reader.read_masks_features(selected_masks_option)
    print("read labeled_masks, shape", labeled_masks_df.shape)
    if CONFIDENCE_COLUMN not in labeled_masks_df.columns:
        labeled_masks_df[CONFIDENCE_COLUMN] = 0.0
    labeled_masks_df = labeled_masks_df[
        [MASK_ID_COLUMN, Y_COLUMN, LABELING_MODE_COLUMN, CONFIDENCE_COLUMN]
    ]
    print(labeled_masks_df.head())

    assert len(masks) == labeled_masks_df.shape[0]

    crop_infos: List[LabeledMaskPreviewInfo] = []
    for (_, row), mask in zip(labeled_masks_df.iterrows(), masks):
        assert mask["id"] == row[MASK_ID_COLUMN], (mask["id"], row[MASK_ID_COLUMN])
        confidence_score = get_confidence_score(labeled_masks_df, mask["id"])
        crop_infos.append(
            generate_labeled_mask_preview_info(
                image_data_reader.image, mask, row[Y_COLUMN], confidence_score
            )
        )

    sorted_crop_infos = sorted(
        crop_infos, key=lambda crop_info: crop_info.confidence_score
    )

    grouped_by_label = defaultdict(list)
    for crop_info in sorted_crop_infos:
        grouped_by_label[crop_info.label].append(crop_info)

    rows: List[dbc.Row] = []
    for crop_infos in grouped_by_label.values():
        mask_previews = generate_labeled_masks_previews(
            ALL_MASKS_RADIO_BUTTONS_PREFIX, crop_infos
        )
        rows.append(
            dbc.Row(
                [
                    dbc.Col(mask_preview, style={"padding": "0px"})
                    for mask_preview in mask_previews
                ],
                className="masks-class-preview-row",
            ),
        )

    return labeled_masks_df.to_dict("records"), rows


@callback(
    Output(id("completed-checkbox"), "value"),
    Output(id("masks-options"), "options"),
    Output(id("masks-options"), "value"),
    Input(id("image-filepath"), "value"),
)
@timeit
def handle_image_filepath_selection(image_filepath):
    if not image_filepath:
        return {}, {}, {}

    print("Image filepath changed")

    images_metadata = read_images_metadata()
    completed = (
        image_filepath in images_metadata["filepath"].values
        and images_metadata.loc[
            images_metadata["filepath"] == image_filepath, "completed"
        ].iloc[0]
    )

    image_data_reader = ImageDataReader(image_filepath)
    masks_options = image_data_reader.masks_options()
    masks_options_dict = [{"label": "None", "value": "None"}]
    if masks_options:
        masks_options_dict = [
            {"label": mask_name_option, "value": mask_name_option}
            for mask_name_option in masks_options
        ]

    selected_masks_option = masks_options_dict[0]["value"]

    return (
        [CHECKBOX_COMPLETED] if completed else [],
        masks_options_dict,
        selected_masks_option,
    )


@callback(
    Output(id("embedder-metadata-container"), "children"),
    Input(id("classifier-model"), "value"),
    suppress_callback_exceptions=True,
)
def handle_classifier_model_selection(embedder_filepath):
    return construct_embedder_metadata_container(embedder_filepath)


@callback(
    Output(id("classifier-model"), "options"),
    Output(id("classifier-model"), "value"),
    Input(id("train-classifier-button"), "n_clicks"),
    State(id("image-filepath"), "value"),
)
@timeit
def handle_train_classifier_button(n_clicks, image_filepath):
    if n_clicks == 0:
        raise PreventUpdate

    dir = None
    # Train on data that is under the same subfolder of dataset/ as the currently
    # selected image.
    if image_filepath is not None:
        dir = str(
            PurePath(RAW_IMAGES_DIR)
            / PurePath(image_filepath).relative_to(RAW_IMAGES_DIR).parts[0]
        )

    train_pipeline(dir=dir)
    model_filepaths = get_classifier_model_filepaths(as_str=True)
    if not model_filepaths:
        return ["none"], "none"

    return model_filepaths, model_filepaths[0]


@callback(
    Output(id("labeled-masks"), "data", allow_duplicate=True),
    Input(id("run-classifier-button"), "n_clicks"),
    State(id("classifier-model"), "value"),
    State(id("image-filepath"), "value"),
    State(id("masks-options"), "value"),
    prevent_initial_call=True,
)
@timeit
def handle_run_classifier_button(
    n_clicks, classifier_model_filepath, image_filepath, selected_masks_option
):
    if (
        not n_clicks
        or not image_filepath
        or not classifier_model_filepath
        or not is_valid_masks_option(selected_masks_option)
    ):
        raise PreventUpdate

    image_data_reader = ImageDataReader(image_filepath)
    labeled_masks = image_data_reader.read_masks_features(selected_masks_option)
    unlabled_rows = labeled_masks[Y_COLUMN] == LABEL_UNLABELED
    classify_results = classify(labeled_masks[unlabled_rows], classifier_model_filepath)

    labeled_masks.loc[unlabled_rows, Y_COLUMN] = classify_results.decoded_labels

    # Ensure the DataFrame has a column for confidence scores
    if CONFIDENCE_COLUMN not in labeled_masks.columns:
        labeled_masks[CONFIDENCE_COLUMN] = 0.0
    labeled_masks.loc[unlabled_rows, CONFIDENCE_COLUMN] = (
        classify_results.confidence_scores
    )

    return labeled_masks.to_dict("records")


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
