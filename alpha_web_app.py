import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from datetime import datetime

# V3.4.8 [ULTRA-STABLE CHART]
st.set_page_config(
    page_title="Alpha Engine v3.4.8 [MASTER]",
    page_icon="🏛️",
    layout="wide"
)

# Clear Cache
st.cache_data.clear()

st.markdown("""
<style>
    .main { background-color: #ffffff; color: #000000; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db; }
</style>
""", unsafe_allow_html=True)

def load_data():
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", "r", encoding='utf-8') as f:
            return json.load(f)
    return []

data = load_data()

if not data:
    st.title("🏛️ Alpha Engine v3.4.8")
    st.warning("데이터를 불러올 수 없습니다. 로컬에서 Monitor 프로그램을 실행하거나 동기화해 주세요.")
else:
    st.title("🚀 Alpha Engine Sigma v3.4.8 Live")
    
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
    st.subheader("📈 AI Precision Price Chart")
    selected_name = st.selectbox("차트 선택", options=[item['name'] for item in data])
    selected_item = [item for item in data if item['name'] == selected_name][0]
    
    if "history" in selected_item and selected_item["history"]:
        h_df = pd.DataFrame(selected_item["history"])
        
        # [ULTRA-STABLE FIX] Force formats
        h_df['Close'] = pd.to_numeric(h_df['Close'], errors='coerce').fillna(0)
        h_df['Date'] = pd.to_datetime(h_df['Date'])
        h_df = h_df.sort_values('Date')
        
        st.write(f"📊 종목: {selected_name} | 데이터 포인트: {len(h_df)}개 | 최종가: {h_df['Close'].iloc[-1]:,.0f}원")
        
        fig = go.Figure()
        
        # Simple Line & Markers (Maximum Visibility)
        fig.add_trace(go.Scatter(
            x=h_df['Date'], 
            y=h_df['Close'],
            mode='lines+markers',
            name='Price',
            line=dict(color='blue', width=2), # Blue line for visibility on white
            marker=dict(size=4, color='blue')
        ))
        
        # Signals
        buys = h_df[h_df['Sig'] > 0]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Close'],
                mode='markers', name='BUY (Signal)',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ))
            
        sells = h_df[h_df['Sig'] < 0]
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['Date'], y=sells['Close'],
                mode='markers', name='SELL (Signal)',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ))

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray', title="Time"),
            yaxis=dict(showgrid=True, gridcolor='lightgray', title="Price (KRW)", tickformat=','),
            height=500,
            margin=dict(l=50, r=50, t=20, b=50),
            font=dict(color='black')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("데이터를 불러오는 중입니다.")

st.markdown("---")
st.caption("Alpha Engine Sigma v3.4.8 | Ultra-Stable Charting Mode")
if st.button("🔄 새로고침"):
    st.rerun()
