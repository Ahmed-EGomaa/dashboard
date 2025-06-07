import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🦄 Unicorn Startups Dashboard",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        background: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦄 Unicorn Startups Dashboard</h1>
        <p>Interactive analysis of global unicorn startups valued at $1B+</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your unicorn startups dataset"
    )
    
    # Show expected format
    with st.sidebar.expander("📋 Expected CSV Format"):
        st.write("""
        Your file should contain these columns:
        - **Company**: Name of the startup
        - **Valuation ($B)**: Valuation in billions
        - **Date Joined**: Date when company became unicorn
        - **Industry**: Industry/sector category
        - **Country**: Country of origin
        - **City**: City location (optional)
        
        💡 The system will automatically detect and clean your data format!
        """)
    
    if uploaded_file is None:
        st.info("👆 Please upload your dataset using the sidebar to begin analysis")
        st.stop()
    
    # Load and clean data
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        df, error = clean_data(df_raw)
        
        if error:
            st.error(f"Error cleaning data: {error}")
            st.stop()
        
        st.success(f"✅ Successfully loaded {len(df)} companies from {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()
    
    # Data Preview
    with st.expander("👀 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Sidebar Filters
    st.sidebar.header("🎛️ Filters")
    
    # Industry filter
    industries = ['All'] + sorted(df['Industry'].unique().tolist())
    selected_industries = st.sidebar.multiselect(
        "Select Industries",
        industries,
        default=['All']
    )
    
    # Country filter
    countries = ['All'] + sorted(df['Country'].unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        countries,
        default=['All']
    )
    
    # Valuation filter
    val_min, val_max = float(df['Valuation ($B)'].min()), float(df['Valuation ($B)'].max())
    valuation_range = st.sidebar.slider(
        "Valuation Range (Billions $)",
        min_value=val_min,
        max_value=val_max,
        value=(val_min, val_max),
        step=1.0
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply valuation filter
    filtered_df = filtered_df[
        (filtered_df['Valuation ($B)'] >= valuation_range[0]) &
        (filtered_df['Valuation ($B)'] <= valuation_range[1])
    ]
    
    # Apply industry filter
    if 'All' not in selected_industries and selected_industries:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
    
    # Apply country filter
    if 'All' not in selected_countries and selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    
    # Key Metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Companies", len(filtered_df))
    with col2:
        st.metric("Total Valuation", f"${filtered_df['Valuation ($B)'].sum():.1f}B")
    with col3:
        st.metric("Average Valuation", f"${filtered_df['Valuation ($B)'].mean():.1f}B")
    with col4:
        st.metric("Median Valuation", f"${filtered_df['Valuation ($B)'].median():.1f}B")
    with col5:
        st.metric("Industries", len(filtered_df['Industry'].unique()))
    
    # Charts Section
    st.header("📈 Interactive Visualizations")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Industry Distribution")
        industry_counts = filtered_df['Industry'].value_counts()
        pie_fig = px.pie(
            values=industry_counts.values,
            names=industry_counts.index,
            title="Distribution by Industry Sector",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        pie_fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Companies: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        pie_fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
            height=500
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🌍 Geographic Distribution")
        country_counts = filtered_df['Country'].value_counts().head(10)
        country_fig = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="Top 10 Countries by Number of Unicorns",
            color=country_counts.values,
            color_continuous_scale='viridis',
            text=country_counts.values
        )
        country_fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            xaxis_title="Number of Unicorns",
            yaxis_title="Country"
        )
        country_fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Unicorns: %{x}<extra></extra>'
        )
        st.plotly_chart(country_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Valuation Distribution")
        hist_fig = px.histogram(
            filtered_df,
            x='Valuation ($B)',
            nbins=25,
            title="Distribution of Company Valuations",
            color_discrete_sequence=['#3498db'],
            opacity=0.8
        )
        hist_fig.update_layout(
            xaxis_title="Valuation (Billions $)",
            yaxis_title="Number of Companies",
            showlegend=False,
            height=500
        )
        hist_fig.update_traces(
            hovertemplate='Valuation Range: $%{x}B<br>Companies: %{y}<extra></extra>'
        )
        st.plotly_chart(hist_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Timeline Trend Analysis")
        if 'Date Joined' in filtered_df.columns and not filtered_df['Date Joined'].isna().all():
            # Group by year for trend analysis
            yearly_data = filtered_df.groupby(filtered_df['Date Joined'].dt.year).agg({
                'Company': 'count',
                'Valuation ($B)': 'sum'
            }).reset_index()
            yearly_data.columns = ['Year', 'New Unicorns', 'Total Valuation ($B)']
            
            timeline_fig = px.line(
                yearly_data,
                x='Year',
                y='New Unicorns',
                title="Unicorn Emergence Trend Over Time",
                markers=True,
                color_discrete_sequence=['#e74c3c']
            )
            timeline_fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of New Unicorns",
                height=500
            )
            timeline_fig.update_traces(
                mode='lines+markers',
                marker=dict(size=8),
                line=dict(width=3),
                hovertemplate='Year: %{x}<br>New Unicorns: %{y}<extra></extra>'
            )
        else:
            # Fallback scatter plot if no date data
            timeline_fig = px.scatter(
                filtered_df,
                x='Industry',
                y='Valuation ($B)',
                color='Industry',
                size='Valuation ($B)',
                title="Valuation by Industry",
                hover_data=['Company', 'Country'],
                size_max=20
            )
            timeline_fig.update_layout(height=500)
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Third row - Top companies chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("💰 Top 15 Highest Valued Unicorns")
        top_companies = filtered_df.nlargest(15, 'Valuation ($B)')
        top_fig = px.bar(
            top_companies,
            x='Valuation ($B)',
            y='Company',
            orientation='h',
            color='Industry',
            title="Highest Valued Companies",
            text='Valuation ($B)',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        top_fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            xaxis_title="Valuation (Billions $)",
            yaxis_title="Company"
        )
        top_fig.update_traces(
            texttemplate='$%{text}B',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Valuation: $%{x}B<br>Industry: %{color}<extra></extra>'
        )
        st.plotly_chart(top_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Statistical Analysis Dashboard")
        
        # Create a comprehensive statistical analysis chart
        fig_stats = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Valuation Quartiles', 'Industry vs Country Heatmap', 
                          'Valuation Outliers', 'Growth Distribution'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "box"}, {"type": "violin"}]]
        )
        
        # 1. Quartile Analysis
        quartiles = filtered_df['Valuation ($B)'].quantile([0.25, 0.5, 0.75, 1.0])
        fig_stats.add_trace(
            go.Bar(
                x=['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'Q4 (100%)'],
                y=quartiles.values,
                name='Quartiles',
                marker_color='lightblue',
                text=[f'${v:.1f}B' for v in quartiles.values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Industry vs Country Heatmap (top combinations)
        if len(filtered_df) > 0:
            cross_tab = pd.crosstab(
                filtered_df['Industry'].head(10), 
                filtered_df['Country'].head(10)
            )
            fig_stats.add_trace(
                go.Heatmap(
                    z=cross_tab.values,
                    x=cross_tab.columns,
                    y=cross_tab.index,
                    colorscale='Blues',
                    name='Heatmap'
                ),
                row=1, col=2
            )
        
        # 3. Box plot for outlier detection
        fig_stats.add_trace(
            go.Box(
                y=filtered_df['Valuation ($B)'],
                name='Valuation Outliers',
                marker_color='lightgreen',
                boxpoints='outliers'
            ),
            row=2, col=1
        )
        
        # 4. Violin plot for distribution shape
        fig_stats.add_trace(
            go.Violin(
                y=filtered_df['Valuation ($B)'],
                name='Distribution Shape',
                box_visible=True,
                meanline_visible=True,
                fillcolor='lightcoral',
                opacity=0.6
            ),
            row=2, col=2
        )
        
        fig_stats.update_layout(
            height=600,
            showlegend=False,
            title_text="Statistical Analysis Dashboard"
        )
        
        st.plotly_chart(fig_stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistical Summary
    st.header("📖 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Descriptive Statistics")
        stats_df = filtered_df['Valuation ($B)'].describe()
        st.dataframe(stats_df.round(2))
    
    with col2:
        st.subheader("🏭 Industry Insights")
        industry_stats = filtered_df.groupby('Industry')['Valuation ($B)'].agg(['count', 'mean', 'sum']).round(2)
        industry_stats.columns = ['Count', 'Avg Valuation', 'Total Valuation']
        st.dataframe(industry_stats.head(10))
    
    with col3:
        st.subheader("🌍 Geographic Insights")
        country_stats = filtered_df.groupby('Country')['Valuation ($B)'].agg(['count', 'mean', 'sum']).round(2)
        country_stats.columns = ['Count', 'Avg Valuation', 'Total Valuation']
        st.dataframe(country_stats.head(10))
    
    # Data Story
    st.header("📖 Data Story & Key Insights")
    
    # Calculate statistics
    total_companies = len(filtered_df)
    total_valuation = filtered_df['Valuation ($B)'].sum()
    avg_valuation = filtered_df['Valuation ($B)'].mean()
    median_valuation = filtered_df['Valuation ($B)'].median()
    
    # Industry analysis
    industry_counts = filtered_df['Industry'].value_counts()
    top_industry = industry_counts.index[0] if len(industry_counts) > 0 else "N/A"
    top_industry_count = industry_counts.iloc[0] if len(industry_counts) > 0 else 0
    
    # Geographic analysis
    country_counts = filtered_df['Country'].value_counts()
    top_country = country_counts.index[0] if len(country_counts) > 0 else "N/A"
    top_country_count = country_counts.iloc[0] if len(country_counts) > 0 else 0
    
    # Create analysis sections
    st.subheader("🔍 Key Findings")
    
    st.markdown("#### 📊 Market Overview")
    st.write(f"""
    Our analysis reveals **{total_companies} unicorn startups** with a combined valuation of 
    **${total_valuation:.1f} billion**. The average unicorn is valued at ${avg_valuation:.1f}B, 
    while the median valuation is ${median_valuation:.1f}B, indicating 
    {'significant concentration in mega-unicorns' if avg_valuation > median_valuation * 1.5 else 'relatively balanced distribution'}.
    """)
    
    st.markdown("#### 🏭 Industry Landscape")
    st.write(f"""
    **{top_industry}** dominates the unicorn ecosystem with {top_industry_count} companies 
    ({(top_industry_count/total_companies)*100:.1f}% of total). This reflects current market trends and 
    investor preferences in the startup ecosystem.
    """)
    
    st.markdown("#### 🌎 Geographic Concentration")
    st.write(f"""
    **{top_country}** leads with {top_country_count} unicorns 
    ({(top_country_count/total_companies)*100:.1f}% of total), highlighting the importance of 
    established tech ecosystems, access to capital, and regulatory environments.
    """)
    
    st.markdown("#### 💡 Strategic Implications")
    st.write("""
    - Investment concentration in specific industries creates both opportunities and risks
    - Geographic clustering suggests the importance of ecosystem effects and network benefits
    - Valuation distribution indicates market maturity and investor sophistication
    - Statistical analysis reveals outliers that may represent breakthrough innovations
    """)

if __name__ == '__main__':
    main()
