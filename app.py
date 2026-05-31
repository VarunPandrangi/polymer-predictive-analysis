import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import json
import re
import warnings
from statsmodels.tsa.holtwinters import Holt

warnings.filterwarnings("ignore")

try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except KeyError:
    st.error("Deployment Error: DEEPSEEK_API_KEY not found in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="Hybrid Statistical-Macro Forecaster", layout="wide")
st.title("Targeted Material Price Analysis: Math + Macro")

@st.cache_data
def load_excel_data(file):
    xls = pd.ExcelFile(file)
    return {sheet: pd.read_excel(file, sheet_name=sheet) for sheet in xls.sheet_names}

def isolate_time_series(row, columns):
    timeline = {}
    for col in columns:
        match = re.search(r'(\d{2}\.\d{2}\.\d{4})', str(col))
        if match and pd.notna(row[col]):
            try:
                timeline[match.group(1)] = float(row[col])
            except ValueError:
                continue
    return timeline

def calculate_significant_shifts(timeline, threshold):
    dates = list(timeline.keys())
    prices = list(timeline.values())
    shifts = []
    
    for i in range(1, len(prices)):
        prev_price = prices[i-1]
        curr_price = prices[i]
        
        if prev_price == 0:
            continue
            
        change_pct = (curr_price - prev_price) / prev_price
        if abs(change_pct) >= threshold:
            shifts.append({
                "date": dates[i],
                "previous_price": prev_price,
                "new_price": curr_price,
                "percentage_change": round(change_pct * 100, 2)
            })
    return shifts

def calculate_statistical_forecast(timeline):
    """Executes Holt's Linear Trend on the historical data to provide a deterministic baseline."""
    df_ts = pd.DataFrame(list(timeline.items()), columns=['Date', 'Price'])
    df_ts['Date'] = pd.to_datetime(df_ts['Date'], format='%d.%m.%Y')
    df_ts = df_ts.sort_values('Date')
    df_ts = df_ts.set_index('Date')
    
    # Resample to monthly to regularize the time series
    ts_monthly = df_ts['Price'].resample('M').last().fillna(method='ffill')
    
    if len(ts_monthly) < 3:
        return {"1_month_math": ts_monthly.iloc[-1], "3_month_math": ts_monthly.iloc[-1], "status": "Insufficient data for trend, returning flatlined."}
        
    try:
        model = Holt(ts_monthly).fit(optimized=True)
        forecast = model.forecast(3)
        return {
            "1_month_math": round(forecast.iloc[0], 2),
            "3_month_math": round(forecast.iloc[2], 2),
            "status": "Success"
        }
    except Exception as e:
        last_price = df_ts['Price'].iloc[-1]
        return {"1_month_math": last_price, "3_month_math": last_price, "status": f"Math error fallback to flatline: {e}"}

def query_deepseek_analysis(material_name, timeline, shifts, math_forecast):
    prompt = f"""
    You are a quantitative procurement and macroeconomic auditor. 
    Analyze the raw material '{material_name}'.
    
    Full Timeline Data (Date: Price in INR):
    {json.dumps(timeline)}
    
    High Volatility Events:
    {json.dumps(shifts)}
    
    Statistical Baseline Forecast (Holt's Linear Trend calculation):
    - 1-Month Math Prediction: INR {math_forecast['1_month_math']}
    - 3-Month Math Prediction: INR {math_forecast['3_month_math']}
    
    Directives:
    1. Historical Macro-Correlation: Review the dates of the significant volatility events. Identify major macroeconomic, supply chain, or geopolitical events around those periods that likely caused these price shifts. 
    2. Sentiment Audit: Do current global market conditions (inflation, crude oil, regional trade constraints) support the mathematical baseline forecast provided above? Should a buyer expect the math to hold, or will geopolitical/market realities force a deviation?
    
    You must format your response STRICTLY as a JSON object matching this structure:
    {{
        "macro_correlations": "Detailed analysis of geopolitical/market events during historical price shifts.",
        "sentiment_audit": "Explanation of whether current market sentiment supports or invalidates the statistical math.",
        "final_adjusted_1M": 0.00,
        "final_adjusted_3M": 0.00,
        "executive_summary": "Exactly 2 to 3 lines predicting the trajectory."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a rigid analytical engine auditing mathematical data against real-world macroeconomics. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 
        )
        
        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
            
        return json.loads(raw_output)
    except Exception as e:
        return {"error": str(e)}

# --- UI Execution ---

uploaded_file = st.file_uploader("Upload Pricing Excel Matrix", type=["xlsx", "xls"])

if uploaded_file:
    sheets_dict = load_excel_data(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox("1. Select Material Category (Sheet)", list(sheets_dict.keys()))
        df_selected = sheets_dict[selected_category]
        
        name_col = next((col for col in df_selected.columns if 'name' in str(col).lower() or 'description' in str(col).lower() or 'material' in str(col).lower()), None)
        
        if not name_col:
            st.error("Matrix layout invalid: Missing material description column.")
            st.stop()
            
        valid_items = df_selected[df_selected[name_col].notna()][name_col].unique()
        selected_item = st.selectbox("2. Select Specific Material", valid_items)

    item_row = df_selected[df_selected[name_col] == selected_item].iloc[0]
    historical_data = isolate_time_series(item_row, df_selected.columns)
    
    if not historical_data:
        st.warning("No valid chronological pricing data found for this item.")
        st.stop()

    df_chart = pd.DataFrame(list(historical_data.items()), columns=['Date', 'Price'])
    df_chart['Date'] = pd.to_datetime(df_chart['Date'], format='%d.%m.%Y')
    df_chart = df_chart.sort_values('Date')

    st.subheader(f"Historical Trend: {selected_item}")
    
    fig = px.line(df_chart, x='Date', y='Price', markers=True, title=f"Price Trajectory (INR) - {selected_item}")
    fig.update_layout(yaxis_title="Price (INR)", xaxis_title="Date", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Analysis Parameters")
    
    threshold_options = {
        "5% (High Sensitivity)": 0.05,
        "8% (Standard Volatility)": 0.08,
        "10% (Major Shifts)": 0.10,
        "15% (Severe Shocks)": 0.15,
        "20% (Crisis Level)": 0.20
    }
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        selected_threshold_label = st.selectbox("Select Volatility Threshold", list(threshold_options.keys()), index=1)
        active_threshold = threshold_options[selected_threshold_label]

    if st.button(f"Generate Hybrid Prediction for {selected_item}"):
        with st.spinner("Calculating statistical trend and auditing against macro-sentiment..."):
            
            significant_shifts = calculate_significant_shifts(historical_data, threshold=active_threshold)
            math_forecast = calculate_statistical_forecast(historical_data)
            
            result = query_deepseek_analysis(selected_item, historical_data, significant_shifts, math_forecast)
            
            if "error" in result:
                st.error(f"Execution failed: {result['error']}")
            else:
                st.markdown("### Forecast Comparison Matrix")
                
                c1, c2, c3 = st.columns(3)
                last_price = df_chart['Price'].iloc[-1]
                c1.metric("Last Recorded Price", f"₹{last_price:.2f}")
                c2.metric("Statistical Baseline (1M)", f"₹{math_forecast['1_month_math']:.2f}")
                c3.metric("AI Macro-Adjusted (1M)", f"₹{result.get('final_adjusted_1M', 0):.2f}")

                st.markdown("### Market Audit & Synthesis")
                st.info(f"**Historical Volatility Drivers:**\n{result.get('macro_correlations', 'N/A')}")
                st.warning(f"**Current Sentiment Audit:**\n{result.get('sentiment_audit', 'N/A')}")
                
                st.markdown("### Executive Summary")
                st.success(result.get('executive_summary', 'N/A'))
                
                if significant_shifts:
                    with st.expander(f"View Detected High-Volatility Dates (>{int(active_threshold*100)}% shift)"):
                        st.json(significant_shifts)
                else:
                    st.info(f"No price shifts exceeding the {int(active_threshold*100)}% threshold were detected.")