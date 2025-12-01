import os
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash
import numpy as np
import requests
from io import StringIO

# =========================================================================
# 1. DATA LOADING FROM GOOGLE SHEETS
# =========================================================================

print("Loading data from Google Sheets...")

# Your Google Sheets file ID from the link
# https://docs.google.com/spreadsheets/d/19WEOEzrtRukWyXaVK-8q2P-PnBgmnD47/edit
SHEET_ID = "19WEOEzrtRukWyXaVK-8q2P-PnBgmnD47"

try:
    # Method 1: Direct CSV export from Google Sheets (no authentication needed)
    print("Method 1: Direct CSV download from Google Sheets...")
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
    
    # Download the CSV data
    response = requests.get(CSV_URL)
    response.raise_for_status()  # Check if the request was successful
    
    # Read CSV data into DataFrame
    df_raw = pd.read_csv(StringIO(response.text))
    print("✓ Successfully loaded data via direct CSV download")
    
except Exception as e:
    print(f"❌ Error with direct download: {e}")
    
    # Method 2: Try alternative Google Sheets export format
    print("\nMethod 2: Trying alternative Google Sheets format...")
    try:
        # Alternative URL format
        ALT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"
        response = requests.get(ALT_URL)
        response.raise_for_status()
        df_raw = pd.read_csv(StringIO(response.text))
        print("✓ Successfully loaded data via alternative format")
    except Exception as e2:
        print(f"❌ Error with alternative format: {e2}")
        
        # Method 3: Load from local file if exists
        print("\nMethod 3: Checking for local data file...")
        local_files = ['global_co2_emissions_1994_2024_cleaned.csv', 'co2-data.csv']
        for file in local_files:
            if os.path.exists(file):
                df_raw = pd.read_csv(file)
                print(f"✓ Loaded data from local file: {file}")
                break
        else:
            # Method 4: Use the original dataset URL
            print("\nMethod 4: Loading from original data source...")
            try:
                DATA_URL = "https://drive.google.com/uc?export=download&id=11y-7GuRe7vl226JSZGmXMLFxLCpBZn7e"
                df_raw = pd.read_csv(DATA_URL)
                print("✓ Loaded fallback data from original source")
            except Exception as e3:
                print(f"❌ Fallback also failed: {e3}")
                print("\n⚠️ Creating sample data for demonstration...")
                # Create sample data for demonstration
                years = list(range(1994, 2025))
                countries = ['United States', 'China', 'India', 'Germany', 'Brazil', 'Japan', 'Russia', 'Canada', 'Australia', 'South Africa']
                iso_codes = ['USA', 'CHN', 'IND', 'DEU', 'BRA', 'JPN', 'RUS', 'CAN', 'AUS', 'ZAF']
                
                data = []
                for year in years:
                    for i, country in enumerate(countries):
                        co2 = np.random.uniform(10, 1000) * (1 + 0.02 * (year - 1994))
                        population = np.random.uniform(1e6, 1e9) * (1 + 0.01 * (year - 1994))
                        gdp = np.random.uniform(1e9, 1e13) * (1 + 0.03 * (year - 1994))
                        data.append({
                            'country': country,
                            'iso_code': iso_codes[i],
                            'year': year,
                            'co2': co2,
                            'population': population,
                            'gdp': gdp
                        })
                
                df_raw = pd.DataFrame(data)
                print("✓ Created sample data for demonstration")

# =========================================================================
# 2. DATA PREPROCESSING
# =========================================================================

print(f"\n{'='*60}")
print("DATA PREPROCESSING")
print('='*60)

# Filter and clean the data as per the project requirements (1994-2024)
START_YEAR = 1994
END_YEAR = 2024

print(f"\nFiltering data for years {START_YEAR} to {END_YEAR}...")

# Check if required columns exist
required_cols = ['country', 'year', 'co2']
missing_cols = [col for col in required_cols if col not in df_raw.columns]
if missing_cols:
    print(f"⚠️ Warning: Missing columns: {missing_cols}")
    print("Available columns:", df_raw.columns.tolist())
    
    # Try to rename columns if they have different names
    column_mapping = {}
    if 'Country' in df_raw.columns and 'country' not in df_raw.columns:
        column_mapping['Country'] = 'country'
    if 'Year' in df_raw.columns and 'year' not in df_raw.columns:
        column_mapping['Year'] = 'year'
    if 'CO2' in df_raw.columns and 'co2' not in df_raw.columns:
        column_mapping['CO2'] = 'co2'
    if 'CO₂' in df_raw.columns and 'co2' not in df_raw.columns:
        column_mapping['CO₂'] = 'co2'
    
    if column_mapping:
        df_raw = df_raw.rename(columns=column_mapping)
        print(f"✓ Renamed columns: {column_mapping}")

df_filtered = df_raw.copy()

# Ensure year column is numeric
if 'year' in df_filtered.columns:
    df_filtered['year'] = pd.to_numeric(df_filtered['year'], errors='coerce')

# Filter by year range
df_filtered = df_filtered[
    (df_filtered['year'] >= START_YEAR) &
    (df_filtered['year'] <= END_YEAR)
].copy()

# Ensure iso_code exists
if 'iso_code' not in df_filtered.columns:
    # Create ISO code mapping for common countries
    country_to_iso = {
        'United States': 'USA',
        'China': 'CHN',
        'India': 'IND',
        'Germany': 'DEU',
        'United Kingdom': 'GBR',
        'France': 'FRA',
        'Brazil': 'BRA',
        'Japan': 'JPN',
        'Russia': 'RUS',
        'Canada': 'CAN',
        'Australia': 'AUS',
        'South Africa': 'ZAF',
        'Mexico': 'MEX',
        'South Korea': 'KOR',
        'Italy': 'ITA',
        'Spain': 'ESP',
    }
    df_filtered['iso_code'] = df_filtered['country'].map(country_to_iso)
    
    # For countries without mapping, create a simple code
    missing_countries = df_filtered[df_filtered['iso_code'].isna()]['country'].unique()
    for country in missing_countries:
        if pd.notna(country):
            # Create a simple 3-letter code from country name
            code = ''.join([c for c in str(country)[:3].upper() if c.isalpha()])
            if len(code) == 3:
                country_to_iso[country] = code
    df_filtered['iso_code'] = df_filtered['country'].map(country_to_iso)

# Remove rows without ISO codes
df_filtered = df_filtered[df_filtered['iso_code'].notna()].copy()

print(f"\n✅ Data preprocessing complete!")
print(f"   • Filtered shape: {df_filtered.shape}")
print(f"   • Years in data: {df_filtered['year'].min()} - {df_filtered['year'].max()}")
print(f"   • Countries with ISO codes: {df_filtered['country'].nunique()}")
print(f"   • Total records: {len(df_filtered):,}")

print("\n📋 Column information:")
for col in df_filtered.columns:
    non_null = df_filtered[col].notna().sum()
    null = df_filtered[col].isna().sum()
    dtype = df_filtered[col].dtype
    print(f"   • {col}: {dtype}, {non_null:,} non-null, {null:,} null")

# Handle missing values
print("\n🔄 Handling missing values...")
for col in ['co2', 'gdp', 'population']:
    if col in df_filtered.columns:
        missing_before = df_filtered[col].isna().sum()
        missing_pct = (missing_before / len(df_filtered)) * 100
        
        if missing_before > 0:
            # Fill NaN with 0 for visualization
            df_filtered[col] = df_filtered[col].fillna(0)
            missing_after = df_filtered[col].isna().sum()
            print(f"   • {col}: {missing_before:,} NaN values ({missing_pct:.1f}%) → filled with 0")
        else:
            print(f"   • {col}: No missing values ✓")

# Convert numeric columns to appropriate types
numeric_cols = ['co2', 'gdp', 'population']
for col in numeric_cols:
    if col in df_filtered.columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

# Get the latest available year for initial display
LATEST_YEAR = int(df_filtered['year'].max())
print(f"\n📊 Latest year in data: {LATEST_YEAR}")

# Create a sample of the data
print("\n📄 Data sample (first 5 rows):")
print(df_filtered[['country', 'iso_code', 'year', 'co2', 'population', 'gdp']].head())

# =========================================================================
# 3. DASH APPLICATION SETUP
# =========================================================================

print(f"\n{'='*60}")
print("INITIALIZING DASHBOARD")
print('='*60)

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

server = app.server

# Define a style for unselected items (used for cross-filtering)
HIGHLIGHT_COLOR = '#ff6347'  # Tomato red
DEFAULT_COLOR = '#1f77b4'    # Plotly blue

# =========================================================================
# 4. HELPER FUNCTIONS FOR VISUALIZATIONS
# =========================================================================

def create_choropleth_map(df_year, highlight_countries=[]):
    """Generates the Choropleth Map for a single year with country highlighting."""
    # Check if we have data for this year
    if df_year.empty:
        fig = px.choropleth(title="No data available for selected year")
        fig.update_layout(
            annotations=[dict(
                text="No data available",
                showarrow=False,
                font=dict(size=14),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="co2",
        hover_name="country",
        hover_data={
            "co2": ":,.2f",
            "iso_code": False,
            "gdp": ":,.2f",
            "population": ":,.0f",
            "year": False
        },
        color_continuous_scale=px.colors.sequential.YlOrRd,
        projection="natural earth",
        title=f"Global CO₂ Emissions: {int(df_year['year'].iloc[0])}",
        labels={'co2': 'CO₂ Emissions (million tonnes)'},
        range_color=[0, df_year['co2'].quantile(0.95)]  # Cap color scale at 95th percentile
    )

    # Highlight selected countries with higher opacity
    if highlight_countries:
        fig.update_traces(
            marker_opacity=[
                1.0 if country in highlight_countries else 0.4
                for country in df_year['country']
            ]
        )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        uirevision='map',  # Keeps zoom/pan settings when data updates
        coloraxis_colorbar={
            'title': 'CO₂ (million tonnes)',
            'thickness': 20,
            'len': 0.75
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='lightblue'
        )
    )
    return fig

def create_time_series(df_data, highlight_countries=[]):
    """Generates the Time Series Line Chart for all countries."""
    # Aggregate data by year and country
    df_agg = df_data.groupby(['year', 'country'])['co2'].sum().reset_index()
    
    if df_agg.empty:
        fig = px.line(title="No data available")
        fig.update_layout(
            annotations=[dict(
                text="No data available",
                showarrow=False,
                font=dict(size=14),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    # Limit to top 20 countries by average emissions for better visualization
    avg_emissions = df_agg.groupby('country')['co2'].mean().sort_values(ascending=False)
    top_countries = avg_emissions.head(20).index.tolist()
    df_top = df_agg[df_agg['country'].isin(top_countries)]
    
    fig = px.line(
        df_top,
        x='year',
        y='co2',
        color='country',
        line_group='country',
        hover_data={
            'country': True,
            'co2': ":,.2f",
            'year': False
        },
        title="CO₂ Emission Trends (1994-2024)",
        labels={
            'co2': 'CO₂ Emissions (million tonnes)',
            'year': 'Year'
        }
    )
    
    # Update line styles for highlighted countries
    if highlight_countries:
        for trace in fig.data:
            country = trace.name
            if country in highlight_countries:
                trace.line.color = HIGHLIGHT_COLOR
                trace.line.width = 3
                trace.opacity = 1.0
                trace.line.dash = 'solid'
            else:
                trace.line.color = 'lightgray'
                trace.line.width = 1
                trace.opacity = 0.3
    
    fig.update_layout(
        yaxis_title="CO₂ Emissions (million tonnes)",
        xaxis_title="Year",
        showlegend=len(highlight_countries) > 0,  # Show legend only when countries are selected
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="Countries"
        ) if len(highlight_countries) > 0 else None,
        uirevision='timeseries',
        hovermode='x unified'
    )
    
    # Add year range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

def create_bar_chart(df_year, highlight_countries=[]):
    """Generates the Bar Chart for Top Emitters with highlighting."""
    # Filter out zero CO2 values and get top 10
    df_non_zero = df_year[df_year['co2'] > 0].copy()
    
    if df_non_zero.empty:
        fig = px.bar(title=f"Top CO₂ Emitters: {int(df_year['year'].iloc[0])}")
        fig.update_layout(
            annotations=[dict(
                text="No CO₂ data available for this year",
                showarrow=False,
                font=dict(size=12),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig
    
    df_top = df_non_zero.nlargest(10, 'co2').sort_values('co2', ascending=True)
    
    # Determine colors for highlighting
    colors = [
        HIGHLIGHT_COLOR if country in highlight_countries else DEFAULT_COLOR
        for country in df_top['country']
    ]
    
    fig = px.bar(
        df_top,
        x='co2',
        y='country',
        orientation='h',
        color=colors,
        color_discrete_map='identity',  # Use 'identity' to apply list of colors
        hover_data={
            'co2': ":,.2f",
            'country': False,
            'gdp': ":,.2f",
            'population': ":,.0f"
        },
        title=f"Top 10 CO₂ Emitters: {int(df_year['year'].iloc[0])}",
        labels={
            'co2': 'CO₂ Emissions (million tonnes)',
            'country': 'Country'
        }
    )
    
    fig.update_layout(
        xaxis_title="CO₂ Emissions (million tonnes)",
        yaxis_title="Country",
        showlegend=False,
        uirevision='bar'
    )
    
    # Add value labels on bars
    fig.update_traces(
        texttemplate='%{x:,.0f}',
        textposition='outside'
    )
    
    return fig

def create_scatter_plot(df_year, highlight_countries=[]):
    """Generates a scatter plot of CO₂ vs GDP for the selected year."""
    # Filter out rows where CO2 or GDP is 0
    df_scatter = df_year[(df_year['co2'] > 0) & (df_year['gdp'] > 0)].copy()
    
    # ====== INSUFFICIENT DATA BRANCH ======
    if df_scatter.empty or len(df_scatter) < 3:
        fig = px.scatter(title=f"CO₂ vs GDP: {int(df_year['year'].iloc[0])}")
        
        fig.update_layout(
            xaxis_title="GDP (USD)",
            yaxis_title="CO₂ Emissions (million tonnes)",
            annotations=[dict(
                text="Insufficient data for scatter plot",
                showarrow=False,
                font=dict(size=12),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig

    # ====== NORMAL SCATTER PLOT ======
    fig = px.scatter(
        df_scatter,
        x="gdp",
        y="co2",
        hover_name="country",
        color="country",
        size="co2",
        title=f"CO₂ vs GDP: {int(df_year['year'].iloc[0])}"
    )

    # >>> FIX: Prevent plot from overflowing container <<<
    fig.update_layout(
        xaxis_title="GDP (USD)",
        yaxis_title="CO₂ Emissions (million tonnes)",
        margin=dict(l=10, r=10, t=40, b=10),
        autosize=True,
        height=350,
        hovermode="closest",
    )

    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    return fig

    
    # Determine colors and sizes for highlighting
    if highlight_countries:
        colors = [
            HIGHLIGHT_COLOR if country in highlight_countries else 'lightgray'
            for country in df_scatter['country']
        ]
        sizes = [
            15 if country in highlight_countries else 8
            for country in df_scatter['country']
        ]
        opacities = [
            1.0 if country in highlight_countries else 0.4
            for country in df_scatter['country']
        ]
    else:
        colors = DEFAULT_COLOR
        sizes = 10
        opacities = 0.8
    
    # Create scatter plot
    fig = px.scatter(
        df_scatter,
        x='gdp',
        y='co2',
        hover_name='country',
        hover_data={
            'co2': ':,.2f',
            'gdp': ':,.2f',
            'population': ':,.0f'
        },
        size='population' if 'population' in df_scatter.columns and not df_scatter['population'].isna().all() else None,
        color=colors if isinstance(colors, list) else None,
        title=f"CO₂ vs GDP: {int(df_year['year'].iloc[0])}",
        labels={
            'co2': 'CO₂ Emissions (million tonnes)',
            'gdp': 'GDP (USD)',
            'population': 'Population'
        },
        log_x=True,
        log_y=True
    )
    
    # Update marker properties if we're using a list of colors
    if isinstance(colors, list):
        for i, trace in enumerate(fig.data):
            if i < len(opacities):
                trace.marker.opacity = opacities[i]
            if i < len(sizes) and sizes[i]:
                trace.marker.size = sizes[i]
    
    fig.update_layout(
        xaxis_title="GDP (log scale, USD)",
        yaxis_title="CO₂ Emissions (log scale, million tonnes)",
        showlegend=False,
        uirevision='scatter',
        # Add margins to prevent overflow
        margin=dict(l=20, r=20, t=60, b=40),
        height=420  # Slightly reduced height to fit better
    )
    
    # Add correlation annotation
    if len(df_scatter) > 2:
        correlation = df_scatter['gdp'].corr(df_scatter['co2'])
        fig.add_annotation(
            x=0.98,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"Correlation: {correlation:.2f}",
            showarrow=False,
            font=dict(size=12, color="darkblue"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="darkblue",
            borderwidth=1,
            borderpad=4,
        )
    
    return fig

# =========================================================================
# 5. DASH LAYOUT
# =========================================================================

app.layout = html.Div(
    style={
        'maxWidth': '1400px',
        'margin': 'auto',
        'padding': '20px',
        'fontFamily': 'Roboto, Arial, sans-serif',
        'backgroundColor': '#f8f9fa'
    },
    children=[
        # Header
        html.Div([
            html.H1(
                "🌍 Interactive Global CO₂ Emissions Dashboard",
                style={
                    'textAlign': 'center',
                    'color': '#1f2630',
                    'marginBottom': '10px',
                    'fontSize': '2.5rem'
                }
            ),
            html.P(
                "Explore CO₂ emissions trends, country comparisons, and economic relationships from 1994 to 2024",
                style={
                    'textAlign': 'center',
                    'color': '#6c757d',
                    'fontSize': '1.1rem',
                    'marginBottom': '30px'
                }
            )
        ]),
        
        # Year Selection
        html.Div([
            html.Div([
                html.Label(
                    "Select Year:",
                    style={
                        'fontWeight': 'bold',
                        'fontSize': '1.2rem',
                        'marginRight': '10px'
                    }
                ),
                html.Span(
                    id='year-display',
                    style={
                        'fontSize': '1.3rem',
                        'color': '#007bff',
                        'fontWeight': 'bold',
                        'padding': '5px 15px',
                        'backgroundColor': '#e7f3ff',
                        'borderRadius': '5px'
                    }
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
            
            dcc.Slider(
                id='year-slider',
                min=START_YEAR,
                max=END_YEAR,
                value=LATEST_YEAR,
                marks={i: {'label': str(i),
                           'style':
                                     {
                                      'whiteSpace': 'nowrap'}} 
                       for i in range(START_YEAR, END_YEAR + 1, 5)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            )
        ], style={
            'padding': '25px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Map Visualization
        html.Div([
            html.H3(
                "Global CO₂ Emissions Map",
                style={
                    'marginBottom': '15px',
                    'color': '#343a40',
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '10px'
                }
            ),
            html.P(
                "Click on countries to highlight them across all visualizations",
                style={'color': '#6c757d', 'marginBottom': '20px'}
            ),
            dcc.Graph(
                id='map-graph',
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                },
                style={'height': '550px', 'borderRadius': '8px'}
            )
        ], style={
            'padding': '25px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # CHANGED SECTION: "Comparative Analysis" header moved above the two graphs
        html.Div([
            html.H3(
                "Comparative Analysis",
                style={
                    'marginBottom': '25px',
                    'color': '#343a40',
                    'fontSize': '1.5rem'
                }
            ),
            
            # Two charts in first row
            html.Div([
                # Time Series
                html.Div([
                    html.H4(
                        "Emission Trends Over Time",
                        style={'marginBottom': '15px', 'color': '#495057', 'fontSize': '1.2rem'}
                    ),
                    html.P(
                        "Select lines in legend to highlight countries",
                        style={'color': '#6c757d', 'marginBottom': '15px', 'fontSize': '0.9rem'}
                    ),
                    dcc.Graph(
                        id='time-series-graph',
                        config={'displayModeBar': True},
                        style={'height': '450px', 'borderRadius': '8px', 'width': '100%'}
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                    'flex': '1',
                    'minWidth': '350px',
                    'marginRight': '15px'
                }),
                
                # Bar Chart
                html.Div([
                    html.H4(
                        "Top 10 Emitters",
                        style={'marginBottom': '15px', 'color': '#495057', 'fontSize': '1.2rem'}
                    ),
                    html.P(
                        "Click bars to select countries",
                        style={'color': '#6c757d', 'marginBottom': '15px', 'fontSize': '0.9rem'}
                    ),
                    dcc.Graph(
                        id='bar-chart-graph',
                        config={'displayModeBar': True},
                        style={'height': '450px', 'borderRadius': '8px', 'width': '100%'}
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                    'flex': '1',
                    'minWidth': '350px'
                })
            ], style={
                'display': 'flex',
                'gap': '20px',
                'flexWrap': 'wrap',
                'marginBottom': '30px'
            }),
            
            # Scatter Plot in second row -
            html.Div([
                html.H4(
                    "CO₂ vs GDP Relationship",
                    style={'marginBottom': '15px', 'color': '#495057', 'fontSize': '1.2rem'}
                ),
                html.P(
                    "Lasso or box select points to highlight",
                    style={'color': '#6c757d', 'marginBottom': '15px', 'fontSize': '0.9rem'}
                ),
                html.Div([
                    dcc.Graph(
                        id='scatter-graph',
                        config={'displayModeBar': True},
                        style={'height': '420px', 'borderRadius': '8px', 'width': '100%'}
                    )
                ], style={'overflow': 'hidden', 'borderRadius': '8px'})  
            ], style={
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                'width': '97%',
                'overflow': 'hidden'  
            })
        ], style={
            'marginBottom': '30px'
        }),
        
        # Data Summary and Info
        html.Div([
            html.Div([
                html.H4(
                    "📋 Data Summary",
                    style={'marginBottom': '15px', 'color': '#343a40'}
                ),
                html.Div([
                    html.Div([
                        html.Span(" ", style={'marginRight': '10px'}),
                        html.Span(f"Data Range: {START_YEAR} - {END_YEAR}")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Span(" ", style={'marginRight': '10px'}),
                        html.Span(f"Countries: {df_filtered['country'].nunique()}")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Span("", style={'marginRight': '10px'}),
                        html.Span(f"Total Records: {len(df_filtered):,}")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Span(" ", style={'marginRight': '10px'}),
                        html.Span(f"Countries with GDP Data: {df_filtered[df_filtered['gdp'] > 0]['country'].nunique()}")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Span(" ", style={'marginRight': '10px'}),
                        html.Span(f"Countries with CO₂ Data: {df_filtered[df_filtered['co2'] > 0]['country'].nunique()}")
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'lineHeight': '1.6'})
            ], style={
                'padding': '25px',
                'backgroundColor': '#e3f2fd',
                'borderRadius': '10px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                'flex': '1',
                'marginRight': '20px'
            }),
            
            html.Div([
                html.H4(
                    "🎯 How to Use This Dashboard",
                    style={'marginBottom': '15px', 'color': '#343a40'}
                ),
                html.Ul([
                    html.Li([
                        html.Span(" ", style={'marginRight': '5px'}),
                        html.Span("Map: Click countries to highlight across all charts")
                    ], style={'marginBottom': '8px'}),
                    html.Li([
                        html.Span(" ", style={'marginRight': '5px'}),
                        html.Span("Time Series: Click legend items to select countries")
                    ], style={'marginBottom': '8px'}),
                    html.Li([
                        html.Span(" ", style={'marginRight': '5px'}),
                        html.Span("Bar Chart: Click bars or use box/lasso selection")
                    ], style={'marginBottom': '8px'}),
                    html.Li([
                        html.Span(" ", style={'marginRight': '5px'}),
                        html.Span("Scatter Plot: Shows GDP vs CO₂ correlation")
                    ], style={'marginBottom': '8px'}),
                    html.Li([
                        html.Span(" ", style={'marginRight': '5px'}),
                        html.Span("All charts sync with year slider and selections")
                    ], style={'marginBottom': '8px'}),
                    html.Li([
                        html.Span(" ", style={'marginRight': '5px'}),
                        html.Span("Responsive design for all screen sizes")
                    ])
                ], style={'lineHeight': '1.6', 'paddingLeft': '20px'})
            ], style={
                'padding': '25px',
                'backgroundColor': '#e3f2fd',
                'borderRadius': '10px',
                'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={
            'display': 'flex',
            'gap': '20px',
            'flexWrap': 'wrap',
            'marginBottom': '30px'
        }),
        
        # Footer
        html.Div([
            html.Hr(style={'margin': '30px 0'}),
            html.Div([
                html.P([
                    "📚 Data Source: ",
                    html.A(
                        "Our World in Data CO₂ and Greenhouse Gas Emissions",
                        href="https://github.com/owid/co2-data",
                        target="_blank",
                        style={'color': '#007bff', 'textDecoration': 'none'}
                    )
                ], style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '10px'}),
                html.P([
                    "📊 Dashboard Data: ",
                    html.A(
                        "Google Sheets Source",
                        href=f"https://docs.google.com/spreadsheets/d/{SHEET_ID}",
                        target="_blank",
                        style={'color': '#007bff', 'textDecoration': 'none'}
                    )
                ], style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '10px'}),
                html.P(
                    "Interactive Dashboard for Global CO₂ Emissions Analysis | Built with Plotly Dash",
                    style={'textAlign': 'center', 'color': '#6c757d', 'fontSize': '0.9rem'}
                )
            ])
        ])
    ]
)

# =========================================================================
# 6. DASH CALLBACKS
# =========================================================================

# Store for selected countries
selected_countries_store = []

@app.callback(
    [
        Output('year-display', 'children'),
        Output('map-graph', 'figure'),
        Output('time-series-graph', 'figure'),
        Output('bar-chart-graph', 'figure'),
        Output('scatter-graph', 'figure')
    ],
    [Input('year-slider', 'value')]
)
def update_charts_on_year_change(selected_year):
    """Initializes and updates all charts based on the year slider."""
    global selected_countries_store
    
    # Filter data for selected year
    year_data = df_filtered[df_filtered['year'] == selected_year].copy()
    
    # Update all charts with current selection
    map_fig = create_choropleth_map(year_data, selected_countries_store)
    time_series_fig = create_time_series(df_filtered, selected_countries_store)
    bar_fig = create_bar_chart(year_data, selected_countries_store)
    scatter_fig = create_scatter_plot(year_data, selected_countries_store)
    
    year_display = f"{selected_year}"
    return year_display, map_fig, time_series_fig, bar_fig, scatter_fig

@app.callback(
    [
        Output('map-graph', 'figure', allow_duplicate=True),
        Output('time-series-graph', 'figure', allow_duplicate=True),
        Output('bar-chart-graph', 'figure', allow_duplicate=True),
        Output('scatter-graph', 'figure', allow_duplicate=True)
    ],
    [
        Input('map-graph', 'clickData'),
        Input('time-series-graph', 'clickData'),
        Input('bar-chart-graph', 'clickData'),
        Input('scatter-graph', 'selectedData'),
        Input('scatter-graph', 'clickData')
    ],
    [
        State('year-slider', 'value'),
        State('map-graph', 'figure'),
        State('time-series-graph', 'figure'),
        State('bar-chart-graph', 'figure'),
        State('scatter-graph', 'figure')
    ],
    prevent_initial_call=True
)
def update_on_selection(
    map_click, time_click, bar_click, scatter_select, scatter_click,
    current_year, map_fig, time_fig, bar_fig, scatter_fig
):
    """Handles selections from any graph and updates all charts."""
    global selected_countries_store
    
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    event_type = ctx.triggered[0]['prop_id'].split('.')[1]
    
    # Get selected countries based on trigger
    selected_countries = []
    
    # Map click
    if trigger_id == 'map-graph' and map_click and map_click['points']:
        country = map_click['points'][0].get('hovertext')
        if country:
            selected_countries = [country]
    
    # Time series click (legend)
    elif trigger_id == 'time-series-graph' and time_click and time_click['points']:
        # Get country from curve number
        curve_num = time_click['points'][0].get('curveNumber')
        if curve_num is not None:
            # Get unique countries in order
            unique_countries = df_filtered['country'].unique()
            if curve_num < len(unique_countries):
                selected_countries = [unique_countries[curve_num]]
    
    # Bar chart click
    elif trigger_id == 'bar-chart-graph' and bar_click and bar_click['points']:
        country = bar_click['points'][0].get('y')
        if country:
            selected_countries = [country]
    
    # Scatter plot selection or click
    elif trigger_id == 'scatter-graph':
        if (event_type == 'selectedData' and scatter_select and scatter_select['points']):
            # Multiple selection from scatter plot
            selected_countries = [
                point.get('hovertext') for point in scatter_select['points']
                if point.get('hovertext')
            ]
        elif (event_type == 'clickData' and scatter_click and scatter_click['points']):
            # Single click on scatter plot
            country = scatter_click['points'][0].get('hovertext')
            if country:
                selected_countries = [country]
    
    # Update selected countries store
    if selected_countries:
        # Toggle selection: if country already selected, remove it
        for country in selected_countries:
            if country in selected_countries_store:
                selected_countries_store.remove(country)
            else:
                selected_countries_store.append(country)
    
    # Clear selection if clicking outside (no country selected)
    elif not selected_countries and trigger_id == 'map-graph':
        selected_countries_store = []
    
    # Filter data for current year
    year_data = df_filtered[df_filtered['year'] == current_year].copy()
    
    # Update all charts
    map_fig_new = create_choropleth_map(year_data, selected_countries_store)
    time_series_fig_new = create_time_series(df_filtered, selected_countries_store)
    bar_fig_new = create_bar_chart(year_data, selected_countries_store)
    scatter_fig_new = create_scatter_plot(year_data, selected_countries_store)
    
    return map_fig_new, time_series_fig_new, bar_fig_new, scatter_fig_new

# =========================================================================
# 7. RUN THE APPLICATION
# =========================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("STARTING INTERACTIVE CO₂ EMISSIONS DASHBOARD")
    print('='*60)
    print(f"Data loaded successfully!")
    print(f"   • Records: {len(df_filtered):,}")
    print(f"   • Countries: {df_filtered['country'].nunique()}")
    print(f"   • Year range: {int(df_filtered['year'].min())} - {int(df_filtered['year'].max())}")
    print(f"   • Latest year: {LATEST_YEAR}")
    
    print(f"\n Dashboard URL: http://127.0.0.1:8050")
    print(f" Open this URL in your web browser")
    print(f" Press Ctrl+C to stop the server")
    print('='*60)
    
    try:
        app.run(
            debug=True,
            host="127.0.0.1",
            port=8050,
            dev_tools_hot_reload=True
        )
    except Exception as e:
        print(f"\n Error starting server: {e}")
        print("\n Troubleshooting tips:")
        print("1. Make sure port 8050 is not already in use")
        print("2. Try: app.run_server(debug=True, port=8051)")
        print("3. Check if all required packages are installed")
        print("4. Try running: pip install dash pandas plotly requests numpy")