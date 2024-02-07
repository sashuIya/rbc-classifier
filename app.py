import os

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, page_container, page_registry
from werkzeug.middleware.profiler import ProfilerMiddleware

PROF_DIR = "/tmp/pprof"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)
server = app.server

from pages import execute_sam, label_cells, measure_image  # noqa: F401, E402

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

# Enable dash profiling, etc.
# app.enable_dev_tools(dev_tools_ui=True, dev_tools_hot_reload=False)

if __name__ == "__main__":
    if os.getenv("PROFILER", None):
        app.server.config["PROFILE"] = True
        app.server.wsgi_app = ProfilerMiddleware(
            app.server.wsgi_app,
            sort_by=["cumtime"],
            restrictions=[50],
            stream=None,
            profile_dir=PROF_DIR,
        )

    app.run(debug=True)
