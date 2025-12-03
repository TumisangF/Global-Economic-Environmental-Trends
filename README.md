# 🌍 Global Economic & Environmental Trends Dashboard

Interactive Geo-Spatial & Multivariate Analysis of CO₂ Emissions, GDP, and Population (1994–2024)

## Links

- **Dashboard URL:** [https://global-economic-environmental-trends.onrender.com/](https://global-economic-environmental-trends.onrender.com/)
- **Data Exploration Notebook (Colab):** [https://colab.research.google.com/drive/1rnk34dyY-QeOcgb1Elg0A-O66VewMSWG#scrollTo=5HEbj-MnQ2m4](https://colab.research.google.com/drive/1rnk34dyY-QeOcgb1Elg0A-O66VewMSWG#scrollTo=5HEbj-MnQ2m4)

## Project Overview

This project presents an interactive dashboard designed to explore global environmental and economic trends between **1994 and 2024**. It integrates **geo-spatial analysis**, **time-series trends**, and **multivariate visualizations** to help users investigate relationships between:

- CO₂ emissions
- GDP
- Population

The dashboard enables intuitive exploration of global patterns and inequalities, supporting research, policy analysis, and educational use.

## Key Features

### Choropleth Map (Geo-Spatial CO₂ Emissions)
- Global CO₂ snapshot by year
- Clickable countries with linked updates
- ISO-based mapping for accuracy
- Highlights spatial patterns & regional disparities

### CO₂ vs GDP Scatter Plot
- Lasso & box selection
- Outlier detection
- Cluster identification
- Fully synchronized with the map

### Time-Series Trend (1994–2024)
- Displays CO₂ trajectory when a country is selected
- Supports long-term temporal analysis

### Summary KPI Cards
- Global GDP
- Total CO₂ emissions
- World population
- Number of countries available

## Tech Stack

**Framework & Backend:**
- Dash (Plotly)
- Flask (via Dash)

**Visualization:**
- Plotly Express
- Dash Core Components

**Data Processing:**
- pandas
- NumPy

**Deployment:**
- Render
