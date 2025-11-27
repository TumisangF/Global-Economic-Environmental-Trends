import os
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

# -----------------------------
# Create the Dash app
# -----------------------------
app = Dash(
    __name__,
    suppress_callback_exceptions=True
)

# Expose server for Render
server = app.server

# -----------------------------
# Load your dataset(s)
# -----------------------------
# Example — replace with your real loading code
# df = pd.read_csv("data.csv")

# -----------------------------
# Dash Layout
# -----------------------------
app.layout = html.Div(
    [
        html.H1("Dashboard Updated... Render should detect this"),
        dcc.Graph(
            figure=px.scatter(x=[1, 2, 3], y=[4, 6, 1])
        )
    ]
)

# -----------------------------
# Launch (Render will not run this)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    #app.run_server(
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
