import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

# ----------------- CONFIGURATION -----------------
st.set_page_config(page_title="Power BI Style Sales Dashboard", page_icon="📊", layout="wide")

# Custom CSS for a clean, sleek "Power BI" look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #f2c80f;
    }
    .metric-title {
        color: #555555;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .metric-value {
        color: #1f1f1f;
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    # Load detailed data for historical analysis
    try:
        df = pd.read_csv("dataset/Sample - Superstore.csv", encoding='windows-1252')
        # Standardize dates
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_prophet_data():
    # Attempt to load processed data or fallback
    try:
        df_processed = pd.read_csv("dataset/processed_sales.csv")
        df_processed['Order Date'] = pd.to_datetime(df_processed['Order Date'])
        sales = df_processed.groupby('Order Date')['Sales'].sum().reset_index()
        return sales
    except:
        df = load_data()
        if not df.empty:
            sales = df.groupby('Order Date')['Sales'].sum().reset_index()
            return sales
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/cf/New_Power_BI_Logo.svg", width=50) # Just for aesthetic "Power BI" vibe
st.sidebar.title("Dashboard Controls")

# View selector
view_option = st.sidebar.radio(
    "Navigation",
    ["📊 Historical Insights", "🤖 Predict Future Sales"]
)

st.sidebar.markdown("---")

if view_option == "📊 Historical Insights":
    st.sidebar.subheader("Filters")
    
    # Date Range Filter
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Depending on date input, unpacking might be a tuple of 1 or 2
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0]
        end_date = max_date
        
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Region Filter
    regions = ["All"] + list(df['Region'].dropna().unique())
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # Category Filter
    categories = ["All"] + list(df['Category'].dropna().unique())
    selected_category = st.sidebar.selectbox("Category", categories)
    
    # Filter Dataset
    mask = (df['Order Date'] >= start_date) & (df['Order Date'] <= end_date)
    filtered_df = df.loc[mask]
    
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    # ----------------- MAIN PAGE: HISTORICAL -----------------
    st.title("Enterprise Sales Analytics")
    
    # KPIs
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else 0
    total_orders = filtered_df['Order ID'].nunique() if 'Order ID' in filtered_df.columns else len(filtered_df)
    
    # Custom HTML KPIs to mimic cards
    kpi1_html = f"""
    <div class="metric-card">
        <div class="metric-title">Total Revenue</div>
        <div class="metric-value">${total_sales:,.0f}</div>
    </div>
    """
    kpi2_html = f"""
    <div class="metric-card">
        <div class="metric-title">Total Profit</div>
        <div class="metric-value">${total_profit:,.0f}</div>
    </div>
    """
    kpi3_html = f"""
    <div class="metric-card">
        <div class="metric-title">Unique Orders</div>
        <div class="metric-value">{total_orders:,.0f}</div>
    </div>
    """
    
    col1, col2, col3 = st.columns(3)
    col1.markdown(kpi1_html, unsafe_allow_html=True)
    col2.markdown(kpi2_html, unsafe_allow_html=True)
    col3.markdown(kpi3_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1 Charts
    c1, c2 = st.columns((2,1))
    
    with c1:
        # Time Series
        monthly_sales = filtered_df.resample('M', on='Order Date')['Sales'].sum().reset_index()
        fig_time = px.area(monthly_sales, x='Order Date', y='Sales', title='Revenue Trend Over Time',
                           color_discrete_sequence=['#1f77b4'])
        fig_time.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="white", xaxis_title="", yaxis_title="Sales ($)")
        st.plotly_chart(fig_time, use_container_width=True)
        
    with c2:
        # Donut Chart for Region
        region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
        fig_region = px.pie(region_sales, names='Region', values='Sales', hole=0.4, title='Sales by Region')
        fig_region.update_traces(textposition='inside', textinfo='percent+label')
        fig_region.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)

    # Row 2 Charts
    c3, c4 = st.columns(2)
    
    with c3:
        # Horizontal Bar for Sub-Category
        subcat_sales = filtered_df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=True)
        fig_subcat = px.bar(subcat_sales, x='Sales', y='Sub-Category', orientation='h', title='Sales by Sub-Category',
                            color='Sales', color_continuous_scale='Blues')
        fig_subcat.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="white", showlegend=False, xaxis_title="Sales ($)", yaxis_title="")
        st.plotly_chart(fig_subcat, use_container_width=True)
        
    with c4:
        # Scatter Plot Sales vs Profit
        if 'Profit' in filtered_df.columns:
            fig_scatter = px.scatter(filtered_df, x='Sales', y='Profit', color='Category', title='Sales vs Profit correlation',
                                     hover_data=['Sub-Category', 'Region'])
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor="white")
            st.plotly_chart(fig_scatter, use_container_width=True)


# ----------------- MAIN PAGE: FORECAST -----------------
elif view_option == "🤖 Predict Future Sales":
    st.title("Sales Forecasting Intelligence")

    forecast_days = st.sidebar.slider("Days to Forecast", 30, 365, 90)
    
    st.markdown("### View Options")
    view_range = st.radio(
        "Select Graph View Range (Zoom in to see predictions clearly):", 
        ["Last 1 Month + Prediction", "Last 1 Year + Prediction", "All Data"], 
        horizontal=True,
        index=1
    )
    
    sales = load_prophet_data()
    
    if sales.empty:
        st.error("No historical data found to run forecasts.")
    else:
        with st.spinner(f"Training Prophet ML Model to forecast next {forecast_days} days..."):
            prophet_df = sales.rename(columns={'Order Date':'ds','Sales':'y'})
            
            model = Prophet()
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
        st.success("Forecast generated successfully!")
        
        # Determine the start date for the view
        max_actual_date = prophet_df['ds'].max()
        if "1 Month" in view_range:
            view_start = max_actual_date - pd.Timedelta(days=30)
        elif "1 Year" in view_range:
            view_start = max_actual_date - pd.Timedelta(days=365)
        else:
            view_start = prophet_df['ds'].min()
            
        # Filter Data for Plotting
        plot_df = prophet_df[prophet_df['ds'] >= view_start]
        plot_forecast = forecast[forecast['ds'] >= view_start]
        
        # Interactive Plotly Forecast Chart
        fig = go.Figure()
        
        # Historical Data
        fig.add_trace(go.Scatter(x=plot_df['ds'], y=plot_df['y'], mode='lines', name='Actual', line=dict(color='black', width=1)))
        
        # Forecasted Data
        fig.add_trace(go.Scatter(x=plot_forecast['ds'], y=plot_forecast['yhat'], mode='lines', name='Forecasted Trend', line=dict(color='rgb(31, 119, 180)', width=2)))
        
        # Uncertainty Intervals
        fig.add_trace(go.Scatter(
            x=list(plot_forecast['ds']) + list(plot_forecast['ds'])[::-1],
            y=list(plot_forecast['yhat_upper']) + list(plot_forecast['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"Predictive Trend for next {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            plot_bgcolor="white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Predicted Values Data")
        forecast_view = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        forecast_view = forecast_view.rename(columns={'ds': 'Order Date', 'yhat': 'Predicted Sales', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
        st.dataframe(forecast_view.style.format({'Predicted Sales': '${:,.2f}', 'Lower Bound': '${:,.2f}', 'Upper Bound': '${:,.2f}'}), use_container_width=True)