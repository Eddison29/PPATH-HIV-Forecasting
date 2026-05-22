import streamlit as st
import pandas as pd
import plotly.express as px
import json
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="P-PATH National Dashboard")

# Initialize session state
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# 2. Data Loading
@st.cache_data
def load_data():
    df = pd.read_excel("Cleaned_Philippines_HIV_SIR_Data.xlsx")
    df_aug = pd.read_excel("Calibrated_Philippines_HIV_SIR_Data.xlsx")
    with open('phregions.json', 'r', encoding='utf-8') as f:
        geo = json.load(f)
    return df, df_aug, geo

# Unpack the two dataframes
df, df_aug, geo = load_data()

# 3. Header
st.title("PPATH (Philippines’ Predictive Analytics for Targeted Healthcare): A GAN-Augmented Transformer Framework for HIV Trend Forecasting")

# --- DYNAMIC CONTROLS Logic ---
latest_year = df['Year'].max()
latest_q = df[df['Year'] == latest_year]['Quarter'].unique()
latest_quarter = sorted(latest_q)[-1] 

year_list = sorted(df['Year'].unique(), reverse=True)
quarter_list = ['Q1', 'Q2', 'Q3', 'Q4']

# 4. Main Layout
col_map, col_bar = st.columns([1.2, 1.8])

# A. Map (Left)
with col_map:
    st.subheader(f"Interactive Geographic Risk Map")
    # Filters are applied inside the columns now
    fig_map = px.choropleth_mapbox(
        df[(df['Year'] == latest_year) & (df['Quarter'] == latest_quarter)], # Initial load
        geojson=geo, locations='Region Name', 
        featureidkey="properties.REGION", color='Infected (I)', 
        color_continuous_scale="Reds", mapbox_style="carto-positron", 
        zoom=4.5, center={"lat": 12.8797, "lon": 121.7740}, opacity=0.7
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="main_geo_map")
    
    if event and event["selection"]["points"]:
        st.session_state.selected_region = event["selection"]["points"][0]["location"]

    all_regions = sorted(df['Region Name'].unique())
    current_idx = 0
    if st.session_state.selected_region in all_regions:
        current_idx = all_regions.index(st.session_state.selected_region) + 1
    
    selected_from_dropdown = st.selectbox("", ["Select a Region"] + all_regions, index=current_idx)
    if selected_from_dropdown:
        st.session_state.selected_region = selected_from_dropdown

# B. Horizontal Bar Chart (Right)
with col_bar:
    # 1. Dropdowns at the top of the right column
    c1, c2 = st.columns(2)
    selected_year = c1.selectbox("Select Year", year_list, index=year_list.index(latest_year))
    selected_quarter = c2.selectbox("Select Quarter", quarter_list, index=quarter_list.index(latest_quarter))
    
    # 2. Filter data based on dropdowns
    filtered_df = df[(df['Year'] == selected_year) & (df['Quarter'] == selected_quarter)]
    
    # 3. Bar Chart construction
    st.subheader(f"Infections by Region ({selected_year} {selected_quarter})")
    df_sorted = filtered_df.sort_values(by='Infected (I)', ascending=True)
    
    fig_top_bar = px.bar(
        df_sorted, x='Infected (I)', y='Region Name', orientation='h', text_auto='.0f',
        color='Infected (I)', color_continuous_scale='Reds',
    )
    
    # 4. Hide the redundant color legend
    fig_top_bar.update_layout(coloraxis_showscale=False)
    
    st.plotly_chart(fig_top_bar, use_container_width=True)



import plotly.graph_objects as go

# C. Historical Trend (Full Width - Overlay Mode)
if st.session_state.selected_region and st.session_state.selected_region != "Select a Region":
    st.divider()
    st.markdown(f"<h3 style='text-align: center;'>{st.session_state.selected_region}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>HISTORICAL TREND</h3>", unsafe_allow_html=True)

    region_df = df[df['Region Name'] == st.session_state.selected_region].copy()
    quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    region_df['Q_Num'] = region_df['Quarter'].map(quarter_map)
    region_df = region_df.sort_values(['Year', 'Q_Num'])
    region_df['Period'] = region_df['Year'].astype(str) + '-' + region_df['Quarter']
    
    fig_trend = go.Figure()
    
    # 1. Add Infected (Back layer)
    fig_trend.add_trace(go.Bar(
        x=region_df['Period'], y=region_df['Infected (I)'],
        name='Infected (I)',
        marker_color="#b51b1b",
        opacity=1 # Slightly transparent so you can see through it
    ))
    
    # 2. Add Removed (Front layer)
    fig_trend.add_trace(go.Bar(
        x=region_df['Period'], y=region_df['Removed (R)'],
        name='Removed (R)',
        marker_color="#ff4b4b",
        opacity=1 # Higher opacity to stand out in front
    ))
    
    fig_trend.update_layout(
        barmode='overlay'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Click a region on the map or use the dropdown above to see its historical trend.")

# D. GAN-Augmented Data Shadow Analysis (Hover-Integrated)
if st.session_state.selected_region and st.session_state.selected_region != "Select a Region":
    st.divider()
    st.markdown(f"<h3 style='text-align: center;'>GAN-AUGMENTED DATA</h3>", unsafe_allow_html=True)
    
    r_df = df_aug[df_aug['Region Name'] == st.session_state.selected_region].copy()
    r_df['Period'] = r_df['Year'].astype(str) + '-' + r_df['Quarter']
    
    fig = go.Figure()

    # 1. Total (Blue Bar) - Contains Shadow data in Hover
    fig.add_trace(go.Bar(
        x=r_df['Period'], y=r_df['Estimated Total Infected'],
        name='Estimated Data<br>Shadow and Total',
        marker_color="#7275ff",
        opacity=1,
        # Pack both shadow and total into customdata
        customdata=r_df[['Generated Data Shadow (Cases)', 'Estimated Total Infected']],
        hovertemplate='<b>Total</b>: %{customdata[1]}<br><b>Shadow</b>: %{customdata[0]}<extra></extra>'
    ))

    # 2. Infected (Red Bar)
    fig.add_trace(go.Bar(
        x=r_df['Period'], y=r_df['Infected (I)'],
        name='Infected (I)',
        marker_color="#282a99",
        opacity=1
    ))

    # Set barmode to 'overlay' so they align perfectly
    fig.update_layout(
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

        











    # Load your calibrated data
    df = pd.read_excel('Calibrated_Philippines_HIV_SIR_Data.xlsx')

    def run_forecast():
        region = st.session_state.get('selected_region', df['Region Name'].unique()[0])
    
        # 2. Filter your data using that variable
        reg_df = df[df['Region Name'] == region].copy()
        reg_df = reg_df.sort_values(['Year', 'Quarter'])
        y_data = reg_df['Estimated Total Infected'].values.astype(float)

        # --- 1. Load Data First so we can compare ---
        df_2025 = pd.read_excel('2025_Isolated_HIV_SIR_Data.xlsx')
        df_2025.columns = df_2025.columns.str.strip()
        comparison_data = df_2025[df_2025['Region Name'] == region]
        
        # Get actual value if available
        actual = comparison_data.iloc[-1]['Estimated Total Infected'] if not comparison_data.empty else None

        # --- 2. Existing Forecasting Logic ---
        model_sarima = sm.tsa.statespace.SARIMAX(y_data, order=(1,1,1), seasonal_order=(1,1,0,4)).fit(disp=False)
        pred_sarima = model_sarima.forecast(steps=1)[0]
        
        model_hw = ExponentialSmoothing(y_data, seasonal_periods=4, trend='add', seasonal='add').fit()
        pred_hw = model_hw.forecast(steps=1)[0]

      

       # --- 3. Clean UI Design ---
        st.subheader(f"FORECASTING ANALYSIS: {region} 2025 Q4")
        col1, col2 = st.columns(2)
        
  


        with col1:
            st.info("### SARIMA")
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: baseline; min-height: 50px;">
                    <span style="font-size: 2.25rem;">{int(round(pred_sarima)):,}</span>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.info("### HOLT-WINTER")
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: baseline; min-height: 50px;">
                    <span style="font-size: 2.25rem;">{int(round(pred_hw)):,}</span>
                </div>
            """, unsafe_allow_html=True)

        # --- 4. Historical Baseline Comparison ---
        st.markdown("---")
        st.markdown("### Historical Baseline Comparison: 2025 Q4")
        
        if not comparison_data.empty:
            latest_2025 = comparison_data.iloc[-1]
            comp1, comp2, comp3 = st.columns(3)
            
            with comp1:
                st.error("### Observed Infected")
                st.metric(label="Actual Cases", value=f"{int(latest_2025['Infected (I)']):,}")
            with comp2:
                st.error("### Generated Shadow")
                st.metric(label="Hidden Cases", value=f"{int(latest_2025['Generated Data Shadow (Cases)']):,}")
            with comp3:
                st.error("### Total Estimate")
                st.metric(label="Total Population Cases", value=f"{int(latest_2025['Estimated Total Infected']):,}")
        else:
            st.error("No 2025 Q4 isolated data found for this region.")

    # Call the function
    if st.button("Generate Next Quarter Forecast"):
        run_forecast()






# --- 5. National Forecast Comparison (All Regions) SARIMA ---
st.divider()
st.markdown(f"<h3 style='text-align: center;'>NATIONAL FORECAST: ALL REGIONS</h3>", unsafe_allow_html=True)
# 1. Load and Clean Data
# We clean columns immediately to avoid KeyErrors
df_full = pd.read_excel('Calibrated_Philippines_HIV_SIR_Data.xlsx')
df_full.columns = df_full.columns.str.strip() 

def get_sarima_forecast(region_name):
    # Filter and prepare data
    reg_df = df_full[df_full['Region Name'] == region_name].copy()
    reg_df = reg_df.sort_values(['Year', 'Quarter'])
    
    # Ensure the column exists after cleaning
    if 'Estimated Total Infected' not in reg_df.columns:
        return 0
        
    y_data = reg_df['Estimated Total Infected'].values.astype(float)
    
    # Fit SARIMA
    try:
        model = sm.tsa.statespace.SARIMAX(y_data, order=(1,1,1), seasonal_order=(1,1,0,4)).fit(disp=False)
        forecast = model.forecast(steps=1)[0]
        return int(round(forecast))
    except:
        return 0 

# 2. Generate predictions for all unique regions
all_regions = sorted(df_full['Region Name'].unique())
forecast_results = []

for region in all_regions:
    pred = get_sarima_forecast(region)
    forecast_results.append({'Region': region, 'Predicted Infections': pred})

# 3. Create DataFrame and Plot
df_forecast = pd.DataFrame(forecast_results)

fig_national = px.bar(
    df_forecast.sort_values(by='Predicted Infections', ascending=True), 
    x='Predicted Infections', 
    y='Region', 
    orientation='h',
    text_auto='.0f',
    title="SARIMA (Seasonal Autoregressive Integrated Moving Average)",
    color='Predicted Infections',
    color_continuous_scale='Reds'
)

fig_national.update_layout(height=600, coloraxis_showscale=False)
st.plotly_chart(fig_national, use_container_width=True)


# --- 6. National Forecast: Holt-Winters (All Regions - HOLT-WINTER ---
st.divider()

def get_holtwinters_forecast(region_name):
    # Filter and prepare data
    reg_df = df_full[df_full['Region Name'] == region_name].copy()
    reg_df = reg_df.sort_values(['Year', 'Quarter'])
    
    if 'Estimated Total Infected' not in reg_df.columns:
        return 0
        
    y_data = reg_df['Estimated Total Infected'].values.astype(float)
    
    try:
        # Holt-Winters Exponential Smoothing
        model = ExponentialSmoothing(y_data, seasonal_periods=4, trend='add', seasonal='add').fit()
        forecast = model.forecast(steps=1)[0]
        return int(round(forecast))
    except:
        return 0 

# Generate predictions for all unique regions
holt_results = []
for region in all_regions:
    pred = get_holtwinters_forecast(region)
    holt_results.append({'Region': region, 'Predicted Infections': pred})

# Create DataFrame and Plot
df_holt = pd.DataFrame(holt_results)

fig_holt = px.bar(
    df_holt.sort_values(by='Predicted Infections', ascending=True), 
    x='Predicted Infections', 
    y='Region', 
    orientation='h',
    text_auto='.0f',
    title="Holt-Winters Exponential Smoothing",
    color='Predicted Infections',
    color_continuous_scale='Blues' # Used Blues to distinguish from SARIMA (Reds)
)

fig_holt.update_layout(height=600, coloraxis_showscale=False)
st.plotly_chart(fig_holt, use_container_width=True)

# --- REVISED FOOTER ---
with st.container():
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: grey; font-style: italic; font-size: 0.9em; margin-bottom: 0px;'>"
        "PPATH (Philippines’ Predictive Analytics for Targeted Healthcare) | 2026"
        "</p>", 
        unsafe_allow_html=True
    )

