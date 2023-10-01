from dash import Dash, html, dcc, page_registry, page_container
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

from pages import measure_image, execute_sam, label_cells

app.layout = html.Div(
    [
        html.H1("Cells labeling tool"),
        html.Div(
            [
                html.Div(
                    dcc.Link(
                        f"{page['name']} - {page['path']}", href=page["relative_path"]
                    ),
                    style={"margin-left": "20px"},
                )
                for page in page_registry.values()
            ]
        ),
        page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
