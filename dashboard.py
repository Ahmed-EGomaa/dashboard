import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
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
    
    # Key Metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Companies", len(filtered_df))
    with col2:
        st.metric("Total Valuation", f"${filtered_df['Valuation ($B)'].sum():.1f}B")
    with col3:
        st.metric("Average Valuation", f"${filtered_df['Valuation ($B)'].mean():.1f}B")
    with col4:
        st.metric("Median Valuation", f"${filtered_df['Valuation ($B)'].median():.1f}B")
    
    # Charts
    st.header("📈 Interactive Visualizations")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Industry Distribution")
        industry_counts = filtered_df['Industry'].value_counts()
        pie_fig = px.pie(
            values=industry_counts.values,
            names=industry_counts.index,
            title="Distribution by Industry Sector",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Geographic Distribution")
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
        st.plotly_chart(country_fig, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Valuation Distribution")
        # Create histogram showing valuation distribution
        hist_fig = px.histogram(
            filtered_df,
            x='Valuation ($B)',
            nbins=20,
            title="Distribution of Company Valuations",
            color_discrete_sequence=['#3498db'],
            opacity=0.7
        )
        hist_fig.update_layout(
            xaxis_title="Valuation (Billions $)",
            yaxis_title="Number of Companies",
            showlegend=False
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with col2:
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
                yaxis_title="Number of New Unicorns"
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
                hover_data=['Company', 'Country']
            )
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Third row - Top companies chart
    st.subheader("💰 Top 15 Highest Valued Unicorns")
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
    
    # Statistical Analysis
    st.header("📖 Data Story & Statistical Insights")
    
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
    - Understanding these patterns can guide investment and business strategies
    """)
    
    st.markdown("#### 🎯 Conclusion")
    st.write("""
    The unicorn landscape reflects broader economic and technological trends. Success patterns show 
    the critical importance of timing, geography, and industry selection. For investors and entrepreneurs, 
    these insights suggest focusing on emerging sectors while leveraging established ecosystems for 
    maximum growth potential.
    """)

if __name__ == '__main__':
    main()
