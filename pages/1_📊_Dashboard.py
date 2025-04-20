import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re
import joblib
from datetime import datetime
import statsmodels.api as sm

st.set_page_config(page_title="Agricultural Dashboard", page_icon="📊", layout="wide")

# Cache the data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("merged_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

st.title("🌾 Agricultural Data Dashboard")

# Load the data
df = load_data()

if df is not None:
    # # Display column names for debugging
    # st.sidebar.subheader("Dataset Columns")
    # st.sidebar.write(df.columns.tolist())
    
    # Show basic statistics
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Number of Crops", df['commodity_desc'].nunique())
    with col3:
        st.metric("Year Range", f"{df['year'].min()} - {df['year'].max()}" if 'year' in df.columns else "N/A")

    # State Performance Analysis
    st.subheader("State Performance Analysis")
    
    # Calculate top and bottom performing states by average yield
    state_yield = df.groupby('state')[['YIELD, MEASURED IN BU / ACRE']].agg(['mean', 'std']).round(2)
    state_yield = state_yield.sort_values(('YIELD, MEASURED IN BU / ACRE', 'mean'), ascending=False)
    
    # Display top and bottom states
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 States by Average Yield")
        top_states = state_yield.head()
        st.dataframe(top_states)
    
    with col2:
        st.subheader("Bottom 5 States by Average Yield")
        bottom_states = state_yield.tail()
        st.dataframe(bottom_states)
    
    # Regional Comparison: Bar charts for average yield by state
    st.subheader("Regional Yield Comparison")
    
    state_avg_yield = df.groupby('state')['YIELD, MEASURED IN BU / ACRE'].mean().reset_index()
    state_avg_yield = state_avg_yield.sort_values('YIELD, MEASURED IN BU / ACRE', ascending=False)
    
    fig_yield = px.bar(
        state_avg_yield, 
        x='state', 
        y='YIELD, MEASURED IN BU / ACRE',
        title='Average Yield by State',
        labels={'state': 'State', 'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)'},
        color='YIELD, MEASURED IN BU / ACRE',
        color_continuous_scale='viridis'
    )
    fig_yield.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_yield, use_container_width=True)
    
    # Regional Comparison: Bar charts for average production by state
    st.subheader("Regional Production Comparison")
    
    state_avg_prod = df.groupby('state')['PRODUCTION, MEASURED IN BU'].mean().reset_index()
    state_avg_prod = state_avg_prod.sort_values('PRODUCTION, MEASURED IN BU', ascending=False)
    
    fig_prod = px.bar(
        state_avg_prod, 
        x='state', 
        y='PRODUCTION, MEASURED IN BU',
        title='Average Production by State',
        labels={'state': 'State', 'PRODUCTION, MEASURED IN BU': 'Production (BU)'},
        color='PRODUCTION, MEASURED IN BU',
        color_continuous_scale='plasma'
    )
    fig_prod.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_prod, use_container_width=True)
    
    # Temperature and Yield Analysis
    st.subheader("Temperature and Yield Analysis")
    
    # Check if temperature column exists
    temp_columns = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
    
    if temp_columns:
        # Use the first temperature column found
        temp_column = temp_columns[0]
        st.write(f"Using temperature data from column: {temp_column}")
        
        # Create temperature ranges
        try:
            df['temp_range'] = pd.qcut(df[temp_column], q=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Very Warm'])
            
            # Create boxplot using plotly
            fig_temp = px.box(
                df, 
                x='temp_range', 
                y='YIELD, MEASURED IN BU / ACRE',
                color='commodity_desc',
                title='Yield Distribution by Temperature Range and Commodity',
                labels={
                    'temp_range': 'Temperature Range',
                    'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)',
                    'commodity_desc': 'Commodity'
                }
            )
            fig_temp.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_temp, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating temperature ranges: {str(e)}")
            st.write("Temperature data may not be suitable for creating ranges. Showing scatter plot instead.")
            
            # Create scatter plot instead
            fig_temp_scatter = px.scatter(
                df, 
                x=temp_column, 
                y='YIELD, MEASURED IN BU / ACRE',
                color='commodity_desc',
                title='Yield vs Temperature by Commodity',
                labels={
                    temp_column: 'Temperature',
                    'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)',
                    'commodity_desc': 'Commodity'
                },
                trendline="ols"
            )
            st.plotly_chart(fig_temp_scatter, use_container_width=True)
    else:
        st.warning("No temperature column found in the dataset. Skipping temperature analysis.")
    
    # Crop Selection
    selected_crop = st.selectbox(
        "Select Crop Type",
        ["All Crops"] + sorted(df['commodity_desc'].unique().tolist())
    )

    # Filter data based on crop selection
    if selected_crop != "All Crops":
        filtered_df = df[df['commodity_desc'] == selected_crop]
    else:
        filtered_df = df

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Yield Analysis", 
        "Variable Relationships", 
        "Time Series",
        "Correlation Analysis"
    ])

    with tab1:
        st.subheader("Yield Analysis")
        
        # Time series of yield
        yearly_yield = filtered_df.groupby('year')['YIELD, MEASURED IN BU / ACRE'].mean().reset_index()
        
        fig_yearly = px.line(
            yearly_yield, 
            x='year', 
            y='YIELD, MEASURED IN BU / ACRE',
            title=f'Average Yield Over Time for {selected_crop}',
            labels={'year': 'Year', 'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)'}
        )
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Yield distribution
        fig_dist = px.histogram(
            filtered_df, 
            x='YIELD, MEASURED IN BU / ACRE',
            title=f'Yield Distribution for {selected_crop}',
            labels={'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)', 'count': 'Frequency'},
            nbins=30
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Yield by state
        state_yield = filtered_df.groupby('state')['YIELD, MEASURED IN BU / ACRE'].mean().reset_index()
        state_yield = state_yield.sort_values('YIELD, MEASURED IN BU / ACRE', ascending=False)
        
        fig_state = px.bar(
            state_yield, 
            x='state', 
            y='YIELD, MEASURED IN BU / ACRE',
            title=f'Average Yield by State for {selected_crop}',
            labels={'state': 'State', 'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)'},
            color='YIELD, MEASURED IN BU / ACRE',
            color_continuous_scale='viridis'
        )
        fig_state.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_state, use_container_width=True)

    with tab2:
        st.subheader("Variable Relationships with Yield")
        
        # Get numeric columns excluding yield and year
        numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['YIELD, MEASURED IN BU / ACRE', 'year', 'FIPS Code', 'state_ansi', 'county_ansi', 'asd_code']]
        
        # Variable selection
        selected_var = st.selectbox(
            "Select Variable to Compare with Yield",
            numeric_cols
        )

        if selected_crop == "All Crops":
            # Create a single scatter plot with different colors for each crop
            fig = px.scatter(
                df, 
                x=selected_var, 
                y='YIELD, MEASURED IN BU / ACRE',
                color='commodity_desc',
                title=f'Yield vs {selected_var} for All Crops',
                labels={'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)'}
            )
            
            # Add trend lines for each crop
            for crop in sorted(df['commodity_desc'].unique()):
                crop_df = df[df['commodity_desc'] == crop]
                X = crop_df[selected_var]
                y = crop_df['YIELD, MEASURED IN BU / ACRE']
                
                if len(X) > 1:  # Only add trendline if we have enough points
                    z = np.polyfit(X, y, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(X.min(), X.max(), 100)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            name=f'{crop} trend',
                            line=dict(color='rgba(0,0,0,0.3)'),
                            showlegend=False
                        )
                    )
            
            # Update layout
            fig.update_layout(
                height=600,
                title=f'Yield vs {selected_var} for All Crops',
                xaxis_title=selected_var,
                yaxis_title='Yield (BU/ACRE)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show regression statistics
            st.write("### Regression Statistics by Crop")
            stats_data = []
            for crop in sorted(df['commodity_desc'].unique()):
                crop_df = df[df['commodity_desc'] == crop]
                X = crop_df[selected_var]
                y = crop_df['YIELD, MEASURED IN BU / ACRE']
                if len(X) > 1:
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    stats_data.append({
                        'Crop': crop,
                        'R-squared': model.rsquared,
                        'P-value': model.pvalues[1],
                        'Coefficient': model.params[1]
                    })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df.style.format({
                'R-squared': '{:.3f}',
                'P-value': '{:.3e}',
                'Coefficient': '{:.3f}'
            }))
            
        else:
            # Scatter plot with trend line for single crop
            fig = px.scatter(filtered_df, 
                           x=selected_var, 
                           y='YIELD, MEASURED IN BU / ACRE',
                           trendline="ols",
                           title=f'Yield vs {selected_var} for {selected_crop}',
                           labels={'YIELD, MEASURED IN BU / ACRE': 'Yield (BU/ACRE)'})
          
            st.plotly_chart(fig, use_container_width=True)

            # Show regression statistics
            X = filtered_df[selected_var]
            y = filtered_df['YIELD, MEASURED IN BU / ACRE']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-squared", f"{model.rsquared:.3f}")
            with col2:
                st.metric("P-value", f"{model.pvalues[1]:.3e}")
            with col3:
                st.metric("Coefficient", f"{model.params[1]:.3f}")

    with tab3:
        st.subheader("Yield Trends Over Time")
        
        if selected_crop == "All Crops":
            # Line plot for all crops
            fig = px.line(df.groupby(['year', 'commodity_desc'])['YIELD, MEASURED IN BU / ACRE'].mean().reset_index(),
                         x='year', y='YIELD, MEASURED IN BU / ACRE', color='commodity_desc',
                         title='Average Yield Trends by Crop')
        else:
            # Line plot for selected crop
            fig = px.line(filtered_df.groupby('year')['YIELD, MEASURED IN BU / ACRE'].mean().reset_index(),
                         x='year', y='YIELD, MEASURED IN BU / ACRE',
                         title=f'Average Yield Trend for {selected_crop}')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Correlation Analysis")
        
        # Select variables for correlation
        correlation_vars = st.multiselect(
            "Select Variables for Correlation Analysis",
            numeric_cols + ['YIELD, MEASURED IN BU / ACRE'],
            default=['YIELD, MEASURED IN BU / ACRE'] + numeric_cols[:3] if len(numeric_cols) > 3 else ['YIELD, MEASURED IN BU / ACRE'] + numeric_cols
        )
        
        if correlation_vars:
            # Create correlation matrix
            corr_matrix = filtered_df[correlation_vars].corr()
            
            # Plot heatmap
            fig = px.imshow(
                corr_matrix,
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show strongest correlations with yield
            if 'YIELD, MEASURED IN BU / ACRE' in correlation_vars:
                st.write("### Strongest Correlations with Yield")
                correlations = corr_matrix['YIELD, MEASURED IN BU / ACRE'].sort_values(ascending=False)
                correlations = correlations.drop('YIELD, MEASURED IN BU / ACRE')
                st.write(pd.DataFrame({
                    'Variable': correlations.index,
                    'Correlation': correlations.values
                }).style.background_gradient(cmap='RdYlBu'))

else:
    st.error("Failed to load the dataset. Please check if the file exists and is accessible.") 