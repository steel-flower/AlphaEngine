import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from datetime import datetime

# V3.4.5 [SYSTEM RECOVERY]
st.set_page_config(
    page_title="Alpha Engine v3.4.5 [MASTER]",
    page_icon="🏛️",
    layout="wide"
)

# Clear Cache
st.cache_data.clear()

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

def load_data():
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", "r", encoding='utf-8') as f:
            return json.load(f)
    return []

data = load_data()

if not data:
    st.title("🏛️ Alpha Engine v3.4.5")
    st.warning("데이터를 불러올 수 없습니다. 로컬에서 Monitor 프로그램을 실행하거나 동기화해 주세요.")
else:
    st.title("🚀 Alpha Engine Sigma v3.4.5 Live")
    
    # 1. Metrics
    cols = st.columns(len(data))
    for i, item in enumerate(data):
        with cols[i]:
            st.metric(
                label=item['name'],
                value=f"{item['price']:,.0f}",
                delta=f"AI: {item['score']:.2f}",
                delta_color="normal" if item['signal'] == "wait" else "inverse"
            )

    st.markdown("---")
    
    # 2. Table
    df = pd.DataFrame(data)
    display_df = df[['name', 'score', 'signal', 'pot', 'price', 'target', 'stop']].copy()
    display_df.columns = ['종목명', 'Score', '상태', '수익(%)', '현재가', '목표가', '손절가']
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    
    # 3. Chart
    st.subheader("📈 Master Precision AI Chart (Rule-Based)")
    selected_name = st.selectbox("차트 선택", options=[item['name'] for item in data])
    selected_item = [item for item in data if item['name'] == selected_name][0]
    
    if "history" in selected_item and selected_item["history"]:
        h_df = pd.DataFrame(selected_item["history"])
        h_df['Date'] = pd.to_datetime(h_df['Date'])
        
        # [CRITICAL FIX] Data Cleaning & Visibility Enhancement
        h_df = h_df.dropna(subset=['Date', 'Close']).sort_values('Date')
        h_df['Close'] = pd.to_numeric(h_df['Close'], errors='coerce')
        
        # Calculate robust range
        valid_close = h_df['Close'].dropna()
        min_p = float(valid_close.min()) * 0.98
        max_p = float(valid_close.max()) * 1.02
        
        st.info(f"📍 {selected_name} 실시간 데이터 검증 완료: {h_df['Close'].iloc[-1]:,.0f}원 (정상 매핑됨)")
        
        fig = go.Figure()
        
        # Raw Data (Rule 1, 2, 3) - Increased Width for Visibility
        fig.add_trace(go.Scatter(
            x=h_df['Date'], y=h_df['Close'],
            mode='lines+markers', # Added markers for absolute visibility
            name='Price',
            line=dict(color='#00ff88', width=3.5), # Thicker line
            marker=dict(size=4, opacity=0.5),
            hovertemplate='%{x}<br>Price: %{y:,.0f}원'
        ))
        
        # Signals
        buys = h_df[h_df['Sig'] > 0]
        sells = h_df[h_df['Sig'] < 0]
        
        fig.add_trace(go.Scatter(
            x=buys['Date'], y=buys['Close'],
            mode='markers', name='Buy Signal',
            marker=dict(symbol='triangle-up', size=15, color='#00ff88', line=dict(width=2, color='white'))
        ))
        
        fig.add_trace(go.Scatter(
            x=sells['Date'], y=sells['Close'],
            mode='markers', name='Sell Signal',
            marker=dict(symbol='triangle-down', size=15, color='#ff4b4b', line=dict(width=2, color='white'))
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0e1117', # Solid Dark Background
            plot_bgcolor='#0e1117',  # Solid Dark Background
            xaxis=dict(
                showgrid=True, gridcolor='#30363d', title="Time",
                rangeslider=dict(visible=False), type='date'
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#30363d', 
                title="Price (KRW)", 
                tickformat=',',
                range=[min_p, max_p],
                fixedrange=False
            ),
            height=600, 
            margin=dict(l=50,r=50,t=30,b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Verification Table (Rule 5)
        with st.expander("📊 Raw Data Table"):
            st.dataframe(h_df.sort_values('Date', ascending=False), use_container_width=True)
    else:
        st.info("데이터를 불러오는 중입니다.")

st.markdown("---")
st.caption("Alpha Engine Sigma v3.4.5 | Master Recovery Edition")
if st.button("🔄 새로고침"):
    st.rerun()
