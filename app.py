import streamlit as st
import pandas as pd
import altair as alt
from engine import get_market_analysis, get_comparative_analysis, get_gemini_report, get_weekly_movers

st.set_page_config(page_title="Risk Diagnostics", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville&display=swap');
    html, body, [class*="st-"] { font-family: 'Libre+Baskerville', serif !important; }
    .stMetric [data-testid="stMetricDelta"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("QUANTITATIVE RISK DIAGNOSTICS")
st.text("TOTAL RISK, SYSTEMIC SENSITIVITY & PERFORMANCE ATTRIBUTION")

mode = st.radio("SELECT MODE:", ["SINGLE ASSET", "COMPARISON"])

def render_dynamic_chart(df, y_title, y_format="Q"):
    """Renders a clean line chart adapting to either Volatility or Returns."""
    df_plot = df.reset_index().melt('Date', var_name='Asset', value_name='Value')
    
    chart = alt.Chart(df_plot).mark_line(
        strokeWidth=1.5 
    ).encode(
        x=alt.X('Date:T', axis=alt.Axis(title='TIMELINE', gridColor='#333333', format='%b %d')),
        y=alt.Y(f'Value:{y_format}', axis=alt.Axis(title=y_title, gridColor='#333333')),
        color=alt.Color('Asset:N', scale=alt.Scale(range=['#4A90E2', '#E74C3C']), legend=alt.Legend(orient='top'))
    ).properties(height=400).configure_view(
        fill='#0E1117', strokeWidth=1, stroke='#333333'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

if mode == "SINGLE ASSET":
    c1, c2 = st.columns(2)
    with c1: t1 = st.text_input("ENTER TICKER (E.G. ALPHA.AT):", "ALPHA.AT")
    with c2: b1 = st.text_input("BENCHMARK (LEAVE BLANK FOR AUTO):", "")
    
    if st.button("EXECUTE") and t1:
        with st.spinner("Compiling Market Data..."):
            try:
                r1 = get_market_analysis(t1, b1)
                st.text(f"BENCHMARK APPLIED: {r1['benchmark']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("TOTAL VOLATILITY", f"{r1['volatility']:.2f}%")
                m2.metric("BETA (SYSTEMIC)", f"{r1['beta']:.2f}")
                m3.metric("ANN. SHARPE (252D)", f"{r1['sharpe']:.2f}")
                
                tab1, tab2 = st.tabs([" 21-DAY ROLLING VOLATILITY", " CUMULATIVE RETURN"])
                with tab1:
                    render_dynamic_chart(pd.DataFrame({r1['ticker']: r1['rolling_vol']}), 'ANNUALIZED VOLATILITY (%)')
                with tab2:
                    render_dynamic_chart(pd.DataFrame({r1['ticker']: r1['cumulative_returns']}), 'CUMULATIVE RETURN (%)')
                
                st.subheader("QUANTITATIVE INFERENCE")
                st.write(get_gemini_report(r1))
            except Exception as e:
                st.error(f"DIAGNOSTIC FAILED: {str(e)}")

else:
    c1, c2 = st.columns(2)
    with c1: 
        t1 = st.text_input("PRIMARY ASSET:", "ALPHA.AT")
        b1 = st.text_input("PRIMARY BENCHMARK (OPTIONAL):", "")
    with c2: 
        t2 = st.text_input("COMPARISON ASSET:", "JNJ")
        b2 = st.text_input("COMPARISON BENCHMARK (OPTIONAL):", "")
    
    if st.button("COMPARE") and t1 and t2:
        with st.spinner("Compiling Comparative Data..."):
            try:
                r1, r2, aligned_vol_df, aligned_cum_df = get_comparative_analysis(t1, t2, b1, b2)
                
                if r1['benchmark'] != r2['benchmark']:
                    st.warning(f"CAUTION: Cross-Benchmark Comparison. {r1['ticker']} β vs {r1['benchmark']} | {r2['ticker']} β vs {r2['benchmark']}.")
                
                st.text(f"METRICS: {r1['ticker']} ({r1['benchmark']})")
                pa1, pa2, pa3 = st.columns(3)
                pa1.metric("VOLATILITY", f"{r1['volatility']:.2f}%")
                pa2.metric("BETA", f"{r1['beta']:.2f}")
                pa3.metric("ANN. SHARPE", f"{r1['sharpe']:.2f}")

                st.text(f"METRICS: {r2['ticker']} ({r2['benchmark']})")
                ca1, ca2, ca3 = st.columns(3)
                ca1.metric("VOLATILITY", f"{r2['volatility']:.2f}%")
                ca2.metric("BETA", f"{r2['beta']:.2f}")
                ca3.metric("ANN. SHARPE", f"{r2['sharpe']:.2f}")
                
                tab1, tab2 = st.tabs([" 21-DAY ROLLING VOLATILITY", " CUMULATIVE RETURN"])
                with tab1:
                    render_dynamic_chart(aligned_vol_df, 'ANNUALIZED VOLATILITY (%)')
                with tab2:
                    render_dynamic_chart(aligned_cum_df, 'CUMULATIVE RETURN (%)')
                
                st.subheader("RELATIVE RISK INFERENCE")
                st.write(get_gemini_report(r1, r2))
            except Exception as e:
                st.error(f"DIAGNOSTIC FAILED: {str(e)}")

st.divider()

st.subheader("7-DAY MARKET HEAT (BY ASSET CLASS)")
st.caption("Annualization factors adjusted dynamically for market trading hours.")

try:
    heat_data = get_weekly_movers()
    tabs = st.tabs(list(heat_data.keys()))
    
    for tab, category in zip(tabs, heat_data.keys()):
        with tab:
            st.dataframe(heat_data[category], hide_index=True, use_container_width=True)
except Exception as e:
    st.error(f"Failed to load heat map: {str(e)}")