import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from datetime import datetime

# V3.4.9 [MASTER RECOVERY: NO MORE LIES]
st.set_page_config(
    page_title="Alpha Engine v3.4.9",
    page_icon="🏛️",
    layout="wide"
)

# Force clear session
if 'clear_cache' not in st.session_state:
    st.cache_data.clear()
    st.session_state['clear_cache'] = True

st.markdown("""
<style>
    .main { background-color: #ffffff; color: #000000; }
    .stMetric { background-color: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

def load_data():
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", "r", encoding='utf-8') as f:
            return json.load(f)
    return []

data = load_data()

if not data:
    st.title("🏛️ Alpha Engine v3.4.9")
    st.warning("데이터를 불러올 수 없습니다.")
else:
    st.title("🚀 Alpha Engine Sigma v3.4.9")
    
    # 1. Metrics
    cols = st.columns(len(data))
    for i, item in enumerate(data):
        with cols[i]:
            st.metric(label=item['name'], value=f"{item['price']:,.0f}")

    st.markdown("---")
    
    # 2. Chart Section
    st.subheader("📈 AI Precision Financial Chart")
    selected_name = st.selectbox("차트 선택", options=[item['name'] for item in data])
    selected_item = [item for item in data if item['name'] == selected_name][0]
    
    if "history" in selected_item and selected_item["history"]:
        # [CRITICAL] Create DataFrame with explicit columns
        h_df = pd.DataFrame(selected_item["history"])
        
        # [FORCE CONVERSION]
        h_df['Close'] = pd.to_numeric(h_df['Close'], errors='coerce')
        h_df['Date'] = pd.to_datetime(h_df['Date'])
        h_df = h_df.dropna(subset=['Close', 'Date']).sort_values('Date')
        
        # [PRICE RANGE GUARD]
        c_min = float(h_df['Close'].min())
        c_max = float(h_df['Close'].max())
        y_range = [c_min * 0.95, c_max * 1.05]
        
        st.info(f"✅ 데이터 확인: {selected_name} | 현재가: {h_df['Close'].iloc[-1]:,.0f}원 | 분석범위: {c_min:,.0f} ~ {c_max:,.0f}")
        
        fig = go.Figure()
        
        # The Line (Blue, Bold)
        fig.add_trace(go.Scatter(
            x=h_df['Date'], 
            y=h_df['Close'],
            mode='lines+markers',
            name='Actual Price',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=4, color='#2563eb')
        ))
        
        # Signal Markers
        buys = h_df[h_df['Sig'] > 0]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Close'],
                mode='markers', name='BUY Signal',
                marker=dict(symbol='triangle-up', size=14, color='green', line=dict(width=1, color='black'))
            ))
            
        sells = h_df[h_df['Sig'] < 0]
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['Date'], y=sells['Close'],
                mode='markers', name='SELL Signal',
                marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=1, color='black'))
            ))

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Date (Time)"),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#f1f5f9', 
                title="Price (KRW Value)", 
                tickformat=',',
                range=y_range # [FORCE PRICE SCALE]
            ),
            height=500,
            margin=dict(l=60, r=40, t=20, b=60),
            font=dict(family="Arial", size=12, color="black")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Check raw data for verification
        with st.expander("데이터 세부값 확인"):
            st.dataframe(h_df.sort_values('Date', ascending=False), use_container_width=True)
    else:
        st.info("데이터를 불러오지 못했습니다.")

st.markdown("---")
st.caption("Alpha Engine Sigma v3.4.9 | Real Price Validation Mode")
if st.button("🔄 즉시 새로고침"):
    st.rerun()
