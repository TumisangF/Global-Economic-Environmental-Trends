import os
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State

# =========================================================================
# 1. DATA LOADING AND PREPROCESSING
# =========================================================================

# Data source: Our World in Data CO₂ and Greenhouse Gas Emissions
# URL is derived from the notebook: interactive_co2_analysis.ipynb
DATA_URL = "https://drive.google.com/uc?export=download&id=11y-7GuRe7vl226JSZGmXMLFxLCpBZn7e"
try:
    df_raw = pd.read_csv(DATA_URL)
except Exception as e:
    print(f"Error loading data: {e}")
    # Create a dummy DataFrame if loading fails for demonstration purposes
    df_raw = px.data.gapminder().rename(columns={'iso_alpha': 'iso_code', 'pop': 'co2', 'gdpPercap': 'gdp'})
    df_raw['country'] = df_raw['country'].astype(str)
    df_raw['iso_code'] = df_raw['iso_code'].astype(str)
    df_raw['co2'] = df_raw['co2'] / 1000 # scale down for realism

# Filter and clean the data as per the project requirements (1994-2024)
START_YEAR = 1994
END_YEAR = 2024

df_filtered = df_raw[
    (df_raw['year'] >= START_YEAR) &
    (df_raw['year'] <= END_YEAR) &
    (df_raw['iso_code'].notna())
].copy()

# Fill NaN CO2 values with 0 for map visualization (otherwise countries disappear)
df_filtered['co2'] = df_filtered['co2'].fillna(0)

# Get the latest available year for initial display
LATEST_YEAR = df_filtered['year'].max()

# =========================================================================
# 2. DASH APPLICATION SETUP
# =========================================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True
)

server = app.server

# Define a style for unselected items (used for cross-filtering)
HIGHLIGHT_COLOR = '#ff6347' # Tomato red
DEFAULT_COLOR = '#1f77b4' # Plotly blue

# =========================================================================
# 3. HELPER FUNCTIONS FOR VISUALIZATIONS
# =========================================================================

def create_choropleth_map(df_year, highlight_countries=[]):
    """Generates the Choropleth Map for a single year with country highlighting."""
    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="co2",
        hover_name="country",
        hover_data={"co2": ":,.2f", "iso_code": False},
        color_continuous_scale=px.colors.sequential.YlOrRd,
        projection="natural earth",
        title=f"CO₂ Emissions by Country: {df_year['year'].iloc[0]}"
    )

    # Highlight selected countries
    fig.update_traces(
        marker_opacity=[
            1.0 if c in highlight_countries else 0.4
            for c in df_year['country']
        ],
        selected_marker={'opacity': 1}
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        uirevision='map', # Keeps zoom/pan settings when data updates
        coloraxis_colorbar={'title': 'CO₂ (million tonnes)'}
    )
    return fig

def create_time_series(df_data, highlight_countries=[]):
    """Generates the Time Series Line Chart for all countries."""
    fig = px.line(
        df_data,
        x='year',
        y='co2',
        color='country',
        line_group='country',
        hover_data={'iso_code': False, 'country': True, 'co2': ":,.2f", 'year': False},
        title="Annual CO₂ Trends (1994-2024)"
    )

    # Highlight selected countries: only plot selected, grey out others
    for trace in fig.data:
        country = trace.name
        if country not in highlight_countries and highlight_countries:
            trace.line.color = 'lightgray'
            trace.line.width = 1
            trace.opacity = 0.5
        else:
            trace.line.width = 3
            trace.opacity = 1.0

    fig.update_layout(
        yaxis_title="CO₂ Emissions (million tonnes)",
        xaxis_title="Year",
        showlegend=False,
        uirevision='timeseries'
    )
    return fig

def create_bar_chart(df_year, highlight_countries=[]):
    """Generates the Bar Chart for Top Emitters with highlighting."""
    # Group and sort for Top 10
    df_top = df_year.nlargest(10, 'co2').sort_values('co2', ascending=True)

    # Determine colors for highlighting
    colors = [
        HIGHLIGHT_COLOR if c in highlight_countries else DEFAULT_COLOR
        for c in df_top['country']
    ]

    fig = px.bar(
        df_top,
        x='co2',
        y='country',
        orientation='h',
        color=colors,
        color_discrete_map='identity', # Use 'identity' to apply list of colors
        hover_data={'co2': ":,.2f", 'country': False},
        title=f"Top 10 CO₂ Emitters: {df_year['year'].iloc[0]}"
    )
    fig.update_layout(
        xaxis_title="CO₂ Emissions (million tonnes)",
        yaxis_title="Country",
        showlegend=False,
        uirevision='bar'
    )
    return fig

# =========================================================================
# 4. DASH LAYOUT
# =========================================================================

app.layout = html.Div(
    style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1("Interactive Global CO₂ Emissions Dashboard", style={'textAlign': 'center'}),
        html.Hr(),

        # Year Slider and Current Year Indicator
        html.Div([
            html.Label(f"Select Year: {LATEST_YEAR}", id='year-display', style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='year-slider',
                min=START_YEAR,
                max=END_YEAR,
                value=LATEST_YEAR,
                marks={i: str(i) for i in range(START_YEAR, END_YEAR + 1, 5)},
                step=1,
            )
        ], style={'padding': '20px'}),
        html.Hr(),

        # Graph Row 1: Map (Width 100%)
        html.Div([
            dcc.Graph(
                id='map-graph',
                config={'displayModeBar': False},
                style={'height': '600px'}
            )
        ]),
        html.Hr(),

        # Graph Row 2: Time Series and Bar Chart (Split 50/50)
        html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
            html.Div(style={'width': '50%'}, children=[
                dcc.Graph(
                    id='time-series-graph',
                    config={'displayModeBar': False},
                    style={'height': '400px'}
                )
            ]),
            html.Div(style={'width': '50%'}, children=[
                dcc.Graph(
                    id='bar-chart-graph',
                    config={'displayModeBar': False},
                    style={'height': '400px'}
                )
            ]),
        ])
    ]
)

# =========================================================================
# 5. DASH CALLBACKS (The Interactivity Engine)
# =========================================================================

# --- Callback 1: Update the graphs based on the Year Slider ---
@app.callback(
    [Output('year-display', 'children'),
     Output('map-graph', 'figure'),
     Output('time-series-graph', 'figure'),
     Output('bar-chart-graph', 'figure')],
    [Input('year-slider', 'value')]
)
def update_charts_on_year_change(selected_year):
    """Initializes and updates all charts based on the year slider."""
    year_data = df_filtered[df_filtered['year'] == selected_year]
    
    # Pass an empty list for highlights to show default views
    map_fig = create_choropleth_map(year_data, [])
    time_series_fig = create_time_series(df_filtered, [])
    bar_fig = create_bar_chart(year_data, [])
    
    year_display = f"Selected Year: {selected_year}"
    return year_display, map_fig, time_series_fig, bar_fig


# --- Callback 2: Cross-Filtering (All Viz React to Each Other) ---
@app.callback(
    [Output('map-graph', 'figure', allow_duplicate=True),
     Output('time-series-graph', 'figure', allow_duplicate=True),
     Output('bar-chart-graph', 'figure', allow_duplicate=True)],
    [Input('map-graph', 'clickData'),
     Input('time-series-graph', 'selectedData'),
     Input('bar-chart-graph', 'selectedData'),
     Input('bar-chart-graph', 'clickData')],
    [State('year-slider', 'value')],
    prevent_initial_call=True
)
def update_on_selection(map_click, time_select, bar_select, bar_click, current_year):
    """
    Handles selections from any graph and updates all three graphs to reflect
    the currently selected set of countries.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # --- 1. Get Selected ISO Codes ---
    selected_iso_codes = []

    if trigger_id == 'map-graph' and map_click:
        # Map click provides a single location
        selected_iso_codes = [map_click['points'][0]['location']]

    elif trigger_id == 'time-series-graph' and time_select and time_select['points']:
        # Time series selection provides points, need to map to 'country'
        countries = {p['curveNumber'] for p in time_select['points']}
        selected_iso_codes = [df_filtered['iso_code'].unique()[c] for c in countries]
        
    elif (trigger_id == 'bar-chart-graph') and (bar_select or bar_click):
        data = bar_select if bar_select else bar_click
        if data and data['points']:
            # Bar chart selection provides countries (y-axis label)
            selected_countries = [p['y'] for p in data['points']]
            # Convert country names to ISO codes
            country_to_iso = df_filtered.set_index('country')['iso_code'].dropna().to_dict()
            selected_iso_codes = [country_to_iso.get(c) for c in selected_countries if country_to_iso.get(c)]

    # --- 2. Convert ISOs to Country Names for Filtering ---
    iso_to_country = df_filtered.set_index('iso_code')['country'].dropna().to_dict()
    highlight_countries = [iso_to_country.get(iso) for iso in selected_iso_codes if iso_to_country.get(iso)]

    # If a selection was made, update all figures to highlight the new set.
    if highlight_countries:
        year_data = df_filtered[df_filtered['year'] == current_year]
        
        map_fig_new = create_choropleth_map(year_data, highlight_countries)
        time_series_fig_new = create_time_series(df_filtered, highlight_countries)
        bar_fig_new = create_bar_chart(year_data, highlight_countries)
        
        return map_fig_new, time_series_fig_new, bar_fig_new
        
    # If selection was cleared (e.g., clicking outside) or was not recognized, prevent update
    raise dash.exceptions.PreventUpdate


# =========================================================================
# 6. LAUNCH (Render will not run this)
# =========================================================================
if __name__ == "__main__":
    import dash
    port = int(os.environ.get("PORT", 8050))
    # Option 1: Use localhost (simplest)
    app.run(debug=True, host="127.0.0.1", port=8050)
