import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from src.common.filepath_util import (
    ImageDataReader,
    read_images_metadata,
    write_images_metadata,
)
from src.pages.widgets.image_selector import (
    TIF_FILEPATHS,
    image_selection_dropdown,
    is_measured,
)
from src.utils.dash_util import id_factory
from src.utils.draw_util import draw_height_and_scale

id = id_factory("measure-image")
dash.register_page(__name__, path="/", order=0)

layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H2(
                    children="Measure image params",
                    style={"textAlign": "center"},
                ),
                image_selection_dropdown(id("image-filepath"), is_measured),
                html.Div(style={"padding-bottom": "20px"}),
            ]
        ),
        html.Div(
            children=[
                html.H4("Image height without scaling metadata"),
                html.Div("Typically 1024 or 2048"),
                daq.NumericInput(
                    id=id("height"),
                    value=2048,
                    max=1e10,
                    size=120,
                ),
                html.Div(style={"padding-bottom": "20px"}),
                #
                html.H4("Micrometers (printed at the scale)"),
                daq.NumericInput(
                    id=id("micrometers"),
                    value=0,
                    max=1e10,
                    size=120,
                ),
                html.Div(style={"padding-bottom": "20px"}),
                #
                html.H4("Scale leftmost pixel X coordinate"),
                daq.NumericInput(
                    id=id("scale-x0"),
                    value=1565,
                    max=1e10,
                    size=120,
                ),
                html.Div(style={"padding-bottom": "20px"}),
                #
                html.H4("Scale rightmost pixel X coordinate"),
                daq.NumericInput(
                    id=id("scale-x1"),
                    value=2776,
                    max=1e10,
                    size=120,
                ),
                html.Div(style={"padding-bottom": "20px"}),
                #
                dbc.Button(
                    "Save metadata",
                    color="primary",
                    className="me-1",
                    id=id("save-metadata"),
                ),
                html.Div(style={"padding-bottom": "20px"}),
                #
                dbc.Button(
                    "Next image",
                    color="primary",
                    className="me-1",
                    id=id("next-image-button"),
                ),
            ],
            style={"display": "inline-block", "margin-right": "200px"},
        ),
        html.Div(
            children=[
                html.H4("Image preview"),
                dcc.Graph(id=id("canvas")),
            ],
            style={"display": "inline-block", "vertical-align": "top"},
        ),
    ],
)


@callback(
    Output(id("canvas"), "figure"),
    Input(id("image-filepath"), "value"),
    Input(id("height"), "value"),
    Input(id("micrometers"), "value"),
    Input(id("scale-x0"), "value"),
    Input(id("scale-x1"), "value"),
)
def handle_measurements_update(image_filepath, height, micrometers, scale_x0, scale_x1):
    image_data_reader = ImageDataReader(image_filepath)
    image = image_data_reader.image
    image_fig = go.FigureWidget()
    image_fig.update_layout(autosize=False, width=1024, height=1024)
    img_trace = px.imshow(image).data[0]
    image_fig.add_trace(img_trace)

    draw_height_and_scale(
        image_fig, image.shape, height, scale_x0, scale_x1, micrometers
    )

    image_fig.update(layout_showlegend=False)

    return image_fig


@callback(
    Input(id("save-metadata"), "n_clicks"),
    State(id("height"), "value"),
    State(id("micrometers"), "value"),
    State(id("scale-x0"), "value"),
    State(id("scale-x1"), "value"),
    State(id("image-filepath"), "value"),
)
def handle_save_metadata_click(
    _, height, micrometers, scale_x0, scale_x1, image_filepath
):
    if ctx.triggered_id != id("save-metadata"):
        raise PreventUpdate

    df = read_images_metadata()
    series = df.loc[df["filepath"] == image_filepath]

    completed = False

    if len(series) > 0:
        completed = series.iloc[0]["completed"]
        df = df.drop(series.index)

    data = pd.DataFrame(
        dict(
            filepath=[image_filepath],
            height=[height],
            micrometers=[micrometers],
            scale_x0=[scale_x0],
            scale_x1=[scale_x1],
            measured=[True],
            completed=[completed],
        )
    )
    df = pd.concat([df, data], ignore_index=True)

    write_images_metadata(df)


@callback(
    Output(id("height"), "value"),
    Output(id("micrometers"), "value"),
    Output(id("scale-x0"), "value"),
    Output(id("scale-x1"), "value"),
    Input(id("image-filepath"), "value"),
)
def handle_image_selection(image_filepath):
    df = read_images_metadata()
    series = df.loc[df["filepath"] == image_filepath]

    assert len(series) <= 1

    if len(series) == 0:
        raise PreventUpdate

    return (
        series.iloc[0]["height"],
        series.iloc[0]["micrometers"],
        series.iloc[0]["scale_x0"],
        series.iloc[0]["scale_x1"],
    )


@callback(
    Output(id("image-filepath"), "value"),
    Input(id("next-image-button"), "n_clicks"),
    State(id("image-filepath"), "value"),
)
def handle_next_image_button_click(n_clicks, image_filepath):
    if not n_clicks:
        return image_filepath

    if ctx.triggered_id != id("next-image-button"):
        return image_filepath

    for i, f in enumerate(TIF_FILEPATHS):
        if f == image_filepath:
            next_index = i + 1
            if next_index < len(TIF_FILEPATHS):
                return TIF_FILEPATHS[next_index]

            return image_filepath

    return image_filepath
