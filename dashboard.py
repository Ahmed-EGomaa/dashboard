import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import numpy as np
from datetime import datetime
import base64
import io
import warnings
import webbrowser  # Added for browser auto-open

warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variable to store the dataframe
df_global = None

def parse_uploaded_file(contents, filename):
    """Parse uploaded CSV file"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, "Unsupported file format. Please upload CSV or Excel files."
    except Exception as e:
        return None, f"There was an error processing this file: {str(e)}"
    
    return df, None

def clean_data(df):
    """Clean and prepare the data"""
    try:
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean the data - handle different column name variations
        expected_columns = ['Company', 'Valuation ($B)', 'Date Joined', 'Industry', 'Country', 'City']
        
        # Check if required columns exist (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        
        # Map column names
        column_mapping = {}
        for expected_col in expected_columns:
            for actual_col in df.columns:
                if expected_col.lower().replace(' ', '').replace('($b)', '') in actual_col.lower().replace(' ', '').replace('($b)', ''):
                    column_mapping[actual_col] = expected_col
                    break
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Filter out rows where essential columns are missing
        df = df.dropna(subset=['Company', 'Valuation ($B)', 'Industry'])
        
        # Clean valuation column
        if 'Valuation ($B)' in df.columns:
            df['Valuation ($B)'] = df['Valuation ($B)'].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
            df['Valuation ($B)'] = pd.to_numeric(df['Valuation ($B)'], errors='coerce')
            df = df.dropna(subset=['Valuation ($B)'])
        
        # Clean date column
        if 'Date Joined' in df.columns:
            df['Date Joined'] = pd.to_datetime(df['Date Joined'], errors='coerce')
            df['Year Joined'] = df['Date Joined'].dt.year
        
        # Fill missing values for optional columns
        for col in ['Country', 'City']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df, None
        
    except Exception as e:
        return None, f"Error cleaning data: {str(e)}"

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🦄 Unicorn Startups Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P("Interactive analysis of global unicorn startups valued at $1B+", 
               style={'textAlign': 'center', 'fontSize': 18, 'color': '#7f8c8d'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # File Upload Section
    html.Div([
        html.H3("📁 Upload Your Dataset", style={'color': '#2c3e50', 'marginBottom': 15}),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className='fas fa-cloud-upload-alt', style={'fontSize': '48px', 'color': '#3498db'}),  # Fixed: removed typo in color
                    html.Br(),
                    'Drag and Drop or ',
                    html.A('Select CSV/Excel Files', style={'color': '#3498db', 'textDecoration': 'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '120px',
                    'lineHeight': '120px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'borderColor': '#3498db',
                    'textAlign': 'center',
                    'backgroundColor': '#f8f9fa',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease'
                },
                multiple=False
            ),
        ]),
        html.Div(id='upload-status', style={'marginTop': '10px', 'textAlign': 'center'}),
        html.Hr(),
        html.Div([
            html.H4("📋 Expected CSV Format:", style={'color': '#2c3e50'}),
            html.P("Your CSV should contain the following columns:"),
            html.Ul([
                html.Li("Company - Name of the startup"),
                html.Li("Valuation ($B) - Valuation in billions (e.g., 140, $95, 45.60)"),
                html.Li("Date Joined - Date when company became unicorn"),
                html.Li("Industry - Industry/sector category"),
                html.Li("Country - Country of origin"),
                html.Li("City - City location (optional)")
            ]),
            html.P("💡 The system will automatically detect and clean your data format!", 
                   style={'fontStyle': 'italic', 'color': '#7f8c8d'})
        ], style={'backgroundColor': '#f1f2f6', 'padding': '15px', 'borderRadius': '8px'})
    ], id='upload-section', style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px', 'border': '1px solid #ddd'}),
    
    # Data Preview Section (initially hidden)
    html.Div(id='data-preview-section', style={'display': 'none'}),
    
    # Interactive Controls Section (initially hidden)
    html.Div(id='controls-section', style={'display': 'none'}),
    
    # Charts Section (initially hidden)
    html.Div(id='charts-section', style={'display': 'none'}),
    
    # Statistical Analysis Section (initially hidden)
    html.Div(id='analysis-section', style={'display': 'none'})  # Fixed: changed from Chinese '顯示' to English 'display'
])

# Callback for file upload
@app.callback(
    [Output('upload-status', 'children'),
     Output('data-preview-section', 'children'),
     Output('data-preview-section', 'style'),
     Output('controls-section', 'children'),
     Output('controls-section', 'style'),
     Output('charts-section', 'children'),
     Output('charts-section', 'style'),
     Output('analysis-section', 'children'),
     Output('analysis-section', 'style')],
    [Input('upload-data', 'contents')],
    [dash.dependencies.State('upload-data', 'filename')]  # Fixed: added missing import
)
def update_output(contents, filename):
    global df_global
    
    if contents is None:
        # No file uploaded yet
        return (
            html.Div("👆 Please upload your dataset to begin analysis", 
                    style={'color': '#7f8c8d', 'fontStyle': 'italic'}),
            "", {'display': 'none'},
            "", {'display': 'none'},
            "", {'display': 'none'},
            "", {'display': 'none'}
        )
    
    # Parse uploaded file
    df_raw, error = parse_uploaded_file(contents, filename)
    
    if error:
        return (
            html.Div([
                html.I(className='fas fa-exclamation-triangle', style={'color': '#e74c3c', 'marginRight': '5px'}),
                f"Error: {error}"
            ], style={'color': '#e74c3c', 'fontWeight': 'bold'}),
            "", {'display': 'none'},
            "", {'display': 'none'},
            "", {'display': 'none'},
            "", {'display': 'none'}
        )
    
    # Clean the data
    df_cleaned, clean_error = clean_data(df_raw)
    
    if clean_error:
        return (
            html.Div([
                html.I(className='fas fa-exclamation-triangle', style={'color': '#e74c3c', 'marginRight': '5px'}),
                f"Data cleaning error: {clean_error}"
            ], style={'color': '#e74c3c', 'fontWeight': 'bold'}),
            "", {'display': 'none'},
            "", {'display': 'none'},
            "", {'display': 'none'},
            "", {'display': 'none'}
        )
    
    # Store cleaned data globally
    df_global = df_cleaned
    
    # Success message
    upload_status = html.Div([
        html.I(className='fas fa-check-circle', style={'color': '#27ae60', 'marginRight': '5px'}),
        f"✅ Successfully loaded {len(df_cleaned)} companies from {filename}"
    ], style={'color': '#27ae60', 'fontWeight': 'bold'})
    
    # Data Preview Section
    preview_section = html.Div([
        html.H3("👀 Data Preview", style={'color': '#2c3e50'}),
        html.P(f"Showing first 10 rows of {len(df_cleaned)} total records:"),
        dash_table.DataTable(
            data=df_cleaned.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_cleaned.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            page_size=10
        )
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px', 'border': '1px solid #ddd'})
    
    # Interactive Controls Section
    controls_section = html.Div([
        html.H3("🎛️ Interactive Filters", style={'color': '#2c3e50'}),
        html.Div([
            html.Div([
                html.Label("Select Industries:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='industry-dropdown',
                    options=[{'label': 'All Industries', 'value': 'all'}] + 
                            [{'label': industry, 'value': industry} for industry in sorted(df_cleaned['Industry'].unique())],
                    value='all',
                    multi=True
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Valuation Range (Billions $):", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='valuation-slider',
                    min=df_cleaned['Valuation ($B)'].min(),
                    max=df_cleaned['Valuation ($B)'].max(),
                    step=1,
                    marks={i: f'${i}B' for i in range(0, int(df_cleaned['Valuation ($B)'].max()) + 1, 20)},
                    value=[df_cleaned['Valuation ($B)'].min(), df_cleaned['Valuation ($B)'].max()]
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ])
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'})
    
    # Charts Section
    charts_section = html.Div([
        # First Row - Two Charts
        html.Div([
            html.Div([
                html.H4("📊 Industry Distribution", style={'textAlign': 'center'}),
                dcc.Graph(id='industry-pie-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H4("🌍 Geographic Distribution", style={'textAlign': 'center'}),
                dcc.Graph(id='country-bar-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        ]),
        
        # Second Row - Two Charts
        html.Div([
            html.Div([
                html.H4("📈 Valuation vs Time Analysis", style={'textAlign': 'center'}),
                dcc.Graph(id='timeline-scatter')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H4("💰 Top 15 Highest Valued Unicorns", style={'textAlign': 'center'}),
                dcc.Graph(id='top-companies-bar')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        ])
    ])
    
    # Statistical Analysis Section
    analysis_section = html.Div([
        html.H2("📖 Data Story & Statistical Insights", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        
        html.Div(id='statistical-analysis', style={
            'backgroundColor': '#f8f9fa', 
            'padding': '30px', 
            'borderRadius': '10px',
            'fontSize': '16px',
            'lineHeight': '1.6'
        })
    ], style={'marginTop': '30px'})
    
    return (
        upload_status,
        preview_section, {'display': 'block'},
        controls_section, {'display': 'block'},
        charts_section, {'display': 'block'},
        analysis_section, {'display': 'block'}
    )

# Callback for updating charts based on filters
@app.callback(
    [Output('industry-pie-chart', 'figure'),
     Output('country-bar-chart', 'figure'),
     Output('timeline-scatter', 'figure'),
     Output('top-companies-bar', 'figure'),
     Output('statistical-analysis', 'children')],
    [Input('industry-dropdown', 'value'),
     Input('valuation-slider', 'value')]
)
def update_dashboard(selected_industries, valuation_range):
    global df_global
    
    if df_global is None:
        # Return empty figures if no data is loaded
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="Please upload data first", 
                               xref="paper", yref="paper", 
                               x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, empty_fig, empty_fig, "No data available"
    
    # Filter data based on selections
    filtered_df = df_global.copy()
    
    # Apply valuation filter
    if valuation_range:
        filtered_df = filtered_df[
            (filtered_df['Valuation ($B)'] >= valuation_range[0]) &
            (filtered_df['Valuation ($B)'] <= valuation_range[1])
        ]
    
    # Apply industry filter
    if selected_industries != 'all' and selected_industries:
        if not isinstance(selected_industries, list):
            selected_industries = [selected_industries]
        if 'all' not in selected_industries:
            filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
    
    # Chart 1: Industry Distribution (Pie Chart)
    industry_counts = filtered_df['Industry'].value_counts()
    pie_fig = px.pie(
        values=industry_counts.values,
        names=industry_counts.index,
        title="Distribution by Industry Sector",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Chart 2: Country Distribution (Horizontal Bar Chart)
    country_counts = filtered_df['Country'].value_counts().head(10)
    country_fig = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        title="Top 10 Countries by Number of Unicorns",
        color=country_counts.values,
        color_continuous_scale='viridis'
    )
    country_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # Chart 3: Timeline Scatter Plot (if date column exists)
    if 'Date Joined' in filtered_df.columns and not filtered_df['Date Joined'].isna().all():
        timeline_fig = px.scatter(
            filtered_df,
            x='Date Joined',
            y='Valuation ($B)',
            color='Industry',
            size='Valuation ($B)',
            hover_data=['Company', 'Country'],
            title="Unicorn Emergence Timeline vs Valuation"
        )
    else:
        # Alternative visualization if no date data
        timeline_fig = px.histogram(
            filtered_df,
            x='Industry',
            y='Valuation ($B)',
            title="Valuation Distribution by Industry",
            color='Industry'
        )
    timeline_fig.update_layout(showlegend=True)
    
    # Chart 4: Top Companies Bar Chart
    top_companies = filtered_df.nlargest(15, 'Valuation ($B)')
    top_fig = px.bar(
        top_companies,
        x='Valuation ($B)',
        y='Company',
        orientation='h',
        color='Industry',
        title="Highest Valued Companies",
        text='Valuation ($B)'
    )
    top_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    top_fig.update_traces(texttemplate='$%{text}B', textposition='outside')
    
    # Statistical Analysis and Storytelling
    total_companies = len(filtered_df)
    total_valuation = filtered_df['Valuation ($B)'].sum()
    avg_valuation = filtered_df['Valuation ($B)'].mean()
    median_valuation = filtered_df['Valuation ($B)'].median()
    
    # Industry analysis
    top_industry = industry_counts.index[0] if len(industry_counts) > 0 else "N/A"
    top_industry_count = industry_counts.iloc[0] if len(industry_counts) > 0 else 0
    
    # Geographic analysis
    country_counts = filtered_df['Country'].value_counts()
    top_country = country_counts.index[0] if len(country_counts) > 0 else "N/A"
    top_country_count = country_counts.iloc[0] if len(country_counts) > 0 else 0
    
    # Create storytelling content
    story_content = html.Div([
        html.H3("🔍 Key Findings", style={'color': '#e74c3c'}),
        
        html.Div([
            html.H4("📊 Market Overview"),
            html.P([
                f"Our analysis reveals {total_companies} unicorn startups with a combined valuation of ",
                html.Strong(f"${total_valuation:.1f} billion"), 
                f". The average unicorn is valued at ${avg_valuation:.1f}B, while the median valuation is ${median_valuation:.1f}B, "
                f"indicating {'significant concentration in mega-unicorns' if avg_valuation > median_valuation * 1.5 else 'relatively balanced distribution'}."
            ])
        ]),
        
        html.Div([
            html.H4("🏭 Industry Landscape"),
            html.P([
                html.Strong(f"{top_industry}"), f" dominates the unicorn ecosystem with {top_industry_count} companies ",
                f"({(top_industry_count/total_companies)*100:.1f}% of total). This reflects current market trends and ",
                "investor preferences in the startup ecosystem."
            ])
        ]),
        
        html.Div([
            html.H4("🌎 Geographic Concentration"),
            html.P([
                html.Strong(f"{top_country}"), f" leads with {top_country_count} unicorns ",
                f"({(top_country_count/total_companies)*100:.1f}% of total), highlighting the importance of ",
                "established tech ecosystems, access to capital, and regulatory environments."
            ])
        ]),
        
        html.Div([
            html.H4("💡 Strategic Implications"),
            html.Ul([
                html.Li("Investment concentration in specific industries creates both opportunities and risks"),
                html.Li("Geographic clustering suggests the importance of ecosystem effects and network benefits"),
                html.Li("Valuation distribution indicates market maturity and investor sophistication"),
                html.Li("Understanding these patterns can guide investment and business strategies")
            ])
        ]),
        
        html.Div([
            html.H4("🎯 Conclusion"),
            html.P([
                "The unicorn landscape reflects broader economic and technological trends. Success patterns show ",
                "the critical importance of timing, geography, and industry selection. For investors and entrepreneurs, ",
                "these insights suggest focusing on emerging sectors while leveraging established ecosystems for ",
                "maximum growth potential."
            ])
        ])
    ])
    
    return pie_fig, country_fig, timeline_fig, top_fig, story_content

# Run the app and open the browser
if __name__ == '__main__':
    # Define the port for the Dash app
    port = 8050
    # Open the default web browser to the Dash app's URL
    webbrowser.open(f'http://127.0.0.1:{port}')
    # Start the Dash server
    app.run(debug=True, port=port)
