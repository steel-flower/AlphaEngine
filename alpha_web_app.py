import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# V3.5.2 [MASTER RECOVERY]
st.set_page_config(
    page_title="Alpha Engine v3.5.2",
    page_icon="🏛️",
    layout="wide"
)

# Force refresh
st.cache_data.clear()

def load_data():
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", "r", encoding='utf-8') as f:
            return json.load(f)
    return []

data = load_data()

if not data:
    st.title("🏛️ Alpha Engine v3.5.2")
    st.error("❌ 데이터를 로드할 수 없습니다. (dashboard_data.json 부재)")
else:
    st.title("🚀 Alpha Engine Sigma v3.5.2 Live")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Metrics Overview
    st.subheader("📊 시장 현황")
    cols = st.columns(len(data))
    for i, item in enumerate(data):
        with cols[i]:
            st.metric(label=item['name'], value=f"{item['price']:,.0f}")

    st.markdown("---")
    
    # 2. Advanced Analysis Chart
    st.subheader("📈 가격 분석 차트 (데이터 무결성 검증 모드)")
    selected_name = st.selectbox("종목 선택", options=[item['name'] for item in data])
    selected_item = [item for item in data if item['name'] == selected_name][0]
    
    if "history" in selected_item and selected_item["history"]:
        # [STEP 1] Raw Data Processing
        h_df = pd.DataFrame(selected_item["history"])
        h_df['Close'] = pd.to_numeric(h_df['Close'], errors='coerce')
        h_df['Date'] = pd.to_datetime(h_df['Date'])
        h_df = h_df.dropna(subset=['Close', 'Date']).sort_values('Date')
        
        # [STEP 2] Data Verification Text (Hard Evidence)
        st.write(f"🔍 **데이터 검증 결과**: 총 `{len(h_df)}`개의 가격 포인트가 탐지되었습니다. (최하가: {h_df['Close'].min():,.0f} ~ 최고가: {h_df['Close'].max():,.0f})")
        
        # [STEP 3] Dual Charting Technique
        tab1, tab2 = st.tabs(["🚀 고정밀 Plotly 차트", "📋 기본 안정형 차트"])
        
        with tab1:
            # Using Plotly Express for maximum reliability
            fig = px.line(
                h_df, x='Date', y='Close', 
                title=f"{selected_name} 가격 추이",
                markers=True,
                labels={'Close': 'Price (KRW)', 'Date': 'Time'}
            )
            fig.update_traces(line=dict(width=3, color='#2563eb'), marker=dict(size=6))
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Streamlit native chart - Nearly impossible to fail if data exists
            native_data = h_df.set_index('Date')['Close']
            st.line_chart(native_data)
            
        # [STEP 4] Raw Data Table
        with st.expander("📄 원본 데이터 수치 직접 확인 (Y축 대조용)"):
            st.table(h_df.sort_values('Date', ascending=False).head(10))
            
    else:
        st.warning("⚠️ 해당 종목의 히스토리 데이터가 존재하지 않습니다.")

st.markdown("---")
st.caption("Alpha Engine Sigma v3.5.0 | Data Sync Verification Mode")
if st.button("🔄 서버 데이터 즉시 갱신"):
    st.rerun()
