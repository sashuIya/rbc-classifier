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
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from draw_util import get_masks_img, get_masked_crop
from filepath_util import read_masks_for_image, read_image
from mask_util import is_point_in_mask

IMAGES_PATH =  os.path.normpath('./dataset/')
SAM_CHECKPOINTS_FOLDER = os.path.normpath('./model/sam/')
DEVICE = 'cpu'

DISPLAY_IMAGE = 'Image'
DISPLAY_MASKS = 'Masks'
DISPLAY_LABELED_DATA = 'Labeled data'
DISPLAY_UNLABELED_DATA = 'Unlabeled data'
DISPLAY_OPTIONS = [DISPLAY_IMAGE, DISPLAY_MASKS, DISPLAY_LABELED_DATA, DISPLAY_UNLABELED_DATA]

LABEL_UNLABELED = 'unlabeled'
LABEL_WRONG = 'wrong'
LABELS = {
    LABEL_UNLABELED: {'color': [255, 255, 255]},
    LABEL_WRONG: {'color': [0, 0, 0]},
    '0': {'color': [255, 0, 0]},
    '1': {'color': [0, 255, 0]},
    '2': {'color': [0, 0, 255]},
}

def sam_model_version(sam_checkpoint_filepath):
    if 'sam_vit_b' in sam_checkpoint_filepath:
        return 'vit_b'
    if 'sam_vit_h' in sam_checkpoint_filepath:
        return 'vit_h'
    if 'sam_vit_l' in sam_checkpoint_filepath:
        return 'vit_l'
    
    return None

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def get_files(folder_path, extension):
    tif_files = []
    search_pattern = folder_path + '/**/*.{}'.format(extension)
    tif_files = glob.glob(search_pattern, recursive=True)
    return tif_files


TIF_FILEPATHS = get_files(folder_path=IMAGES_PATH, extension='tif')
SAM_CHECKPOINT_FILEPATHS = get_files(folder_path=SAM_CHECKPOINTS_FOLDER, extension='pth')

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
                    dcc.Dropdown(TIF_FILEPATHS, TIF_FILEPATHS[0], id='image-filepath'),
                    dcc.Dropdown(SAM_CHECKPOINT_FILEPATHS, SAM_CHECKPOINT_FILEPATHS[0], id='sam-checkpoint-filepath'),
                    dcc.Dropdown(list(LABELS.keys()), '0', id='active-label'),
                    dcc.RadioItems(DISPLAY_OPTIONS, DISPLAY_IMAGE, id='display-option'),
                    html.Button('Run SAM', id='run-sam-button', n_clicks=0),
                    html.Div(id='clicked-pixel-coords'),
                ],
                width=12
            )
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='canvas'), width=6),
                dbc.Col(id='selected-masks', width=4)
            ],
            justify='between'
        ),

        dcc.Store(id='sam-masks'),
        dcc.Store(id='figure-store')
    ],
    fluid=True
)

@callback(
    Output('sam-masks', 'data'),
    Input('run-sam-button', 'n_clicks'),
    State('image-filepath', 'value'),
    State('sam-checkpoint-filepath', 'value')
)
def handle_run_sam_button_click(n_clicks, image_filepath, sam_checkpoint_filepath):
    if not n_clicks: return None
    if not image_filepath: return None
    if not sam_checkpoint_filepath: return None

    image = read_image(image_filepath)
    sam = sam_model_registry[sam_model_version(sam_checkpoint_filepath)](checkpoint=sam_checkpoint_filepath)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128,
        points_per_batch=64,
        # pred_iou_thresh=0.95,
        # stability_score_thresh=0.92,
        # crop_n_layers=0,
        # crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        output_mode='coco_rle',
    )

    masks = mask_generator.generate(image)
    print('found {} masks'.format(len(masks)))

    return masks

def image_to_base64(image_array):
    # Convert the image array to base64-encoded string
    image_base64 = base64.b64encode(image_array).decode('utf-8')
    return image_base64

def ndarray_to_b64(ndarray):
    """
    converts a np ndarray to a b64 string readable by html-img tags 
    """
    img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def generate_crop_with_radio(crop, labels, label, index):
    _, crop_png = cv2.imencode('.png', crop)
    crop_base64 = base64.b64encode(crop_png).decode('utf-8')
    crop_html = html.Img(src=f"data:image/png;base64,{crop_base64}", style={'display': 'block', 'margin-bottom': '10px'}, className="img-item")
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
                            className="ml-3"
                        )
                    )
                ],
                className="img-container",
                style={"margin-bottom": "5px"}
            ),
            dbc.Row(dbc.Col(html.Hr(style={"margin-top": "10px", "margin-bottom": "20px"})))
        ],
    )

@callback(
    Output('clicked-pixel-coords', 'children'),
    Output('selected-masks', 'children'),
    Input('canvas', 'clickData'),
    Input('image-filepath', 'value'),
    State('active-label', 'value')
)
def handle_click(click_data, image_filepath, active_label):
    if ctx.triggered_id == 'image-filepath':
        return [], []

    if ctx.triggered_id == 'active-label':
        raise PreventUpdate
    if not click_data:
        raise PreventUpdate

    point = click_data['points'][0]
    x, y = point['x'], point['y']

    image = read_image(image_filepath)
    masks = read_masks_for_image(image_filepath)
    crops = []
    for mask in masks:
        if is_point_in_mask(x, y, mask):
            label = active_label if len(crops) == 0 else LABEL_WRONG
            crops.append((get_masked_crop(image, mask), label))
            mask['label'] = label
    
    labels = list(LABELS.keys())

    image_radio_items = []
    for i, (crop, label) in enumerate(crops):
        image_radio_item = generate_crop_with_radio(crop, labels, label, i)
        image_radio_items.append(image_radio_item)

    return html.H3('x: {}, y: {}'.format(x, y)), image_radio_items


@callback(
    Output('canvas', 'figure', allow_duplicate=True),
    Input('display-option', 'value'),
    Input('canvas', 'figure'),
    prevent_initial_call=True
)
def handle_display_option_change(display_option, figure):
    if not figure:
        raise PreventUpdate
 
    # Assuming that data[0] is image layer and data[1] is masks layer (see
    # `update_graph`).
    if display_option == DISPLAY_IMAGE:
        figure['data'][1].update(opacity=0.0)
    if display_option == DISPLAY_MASKS:
        figure['data'][1].update(opacity=0.3)

    return figure

@callback(
    Output('canvas', 'figure'),
    Input('image-filepath', 'value'),
)
def update_graph(image_filepath):
    image_fig = go.FigureWidget()
    image_fig.update_layout(autosize=False, width=1024, height=1024)
    # image_fig.update_layout(autosize=True)

    if not image_filepath: return image_fig

    image = read_image(image_filepath)
    img_trace = px.imshow(image).data[0]
    image_fig.add_trace(img_trace)

    masks = read_masks_for_image(image_filepath)
    masks_image = get_masks_img(masks, image)[:, :, :3]
    masks_trace = px.imshow(masks_image).data[0]
    image_fig.add_trace(masks_trace)
    image_fig.data[1].update(opacity=0.0)

    return image_fig


if __name__ == '__main__':
    app.run(debug=True)
