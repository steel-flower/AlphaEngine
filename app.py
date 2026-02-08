"""
Alpha Engine v7.7 - Streamlit Web Application (Hybrid Responsive Mode)
데스크톱과 모바일 사용자 기기를 고려한 자동 레이아웃 최적화 버전
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import time

# 페이지 설정
st.set_page_config(
    page_title="Alpha Engine v7.7",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 📱 모바일/데스크톱 대응 커스텀 CSS
st.markdown("""
<style>
    /* 전체 폰트 및 배경 최적화 */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        font-size: clamp(1.8rem, 5vw, 2.8rem); /* 기기 크기에 따라 폰트 크기 자동 조절 */
        font-weight: 800;
        background: -webkit-linear-gradient(#1f77b4, #08c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* 반응형 지표 카드 */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-label { font-size: 0.9rem; color: #666; margin-bottom: 5px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1f77b4; }

    /* 신호 알림창 */
    .signal-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    .signal-buy { background: #e6fdf5; color: #11998e; border: 2px solid #11998e; }
    .signal-hold { background: #f8f9fa; color: #666; border: 2px solid #ddd; }

    /* 모바일에서 사이드바 가독성 높이기 */
    @media (max-width: 640px) {
        .main-header { margin-top: 1rem; }
        .stMetric { background: #f8f9fa; padding: 10px; border-radius: 10px; }
    }
</style>
""", unsafe_allow_html=True)

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.markdown("<div class='main-header'>🔐 Alpha Engine Access</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pw = st.text_input("🔑 Password", type="password", key="login_pw")
        if st.button("Unlock Dashboard"):
            if pw == st.secrets.get("password", "alpha2026"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Invalid Password")
    return False

@st.cache_data(ttl=300) # 5분 간격 캐시 (모니터 주기와 일치)
def load_web_data(ticker):
    ticker_clean = ticker.replace('.KS', '')
    file_path = f"web_data_{ticker_clean}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    return None, f"데이터를 기다리는 중... ({ticker_clean})"

def create_performance_chart(daily_data):
    df = pd.DataFrame(daily_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=(df['strategy']-1)*100, name='Alpha Engine', line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=df['date'], y=(df['market']-1)*100, name='시장(B&H)', line=dict(color='#ff4b4b', dash='dash')))
    fig.update_layout(
        title='누적 수익률 (전체 기간)',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450 if st.session_state.get('is_mobile', False) else 500
    )
    return fig

def create_monthly_chart(monthly_data):
    df = pd.DataFrame(monthly_data).tail(12)
    colors = ['#1f77b4' if x >= 0 else '#ff4b4b' for x in df['return']]
    fig = px.bar(df, x='month', y='return', text=df['return'].apply(lambda x: f"{x*100:.1f}%"))
    fig.update_traces(marker_color=colors, textposition='outside')
    fig.update_layout(
        title='최근 12개월 실적',
        yaxis_tickformat='.1%',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def main():
    if not check_password(): return

    # 사이드바 설정
    with st.sidebar:
        st.image("https://www.freeiconspng.com/uploads/stock-exchange-icon-png-11.png", width=80)
        st.title("Alpha Engine v7.7")
        menu = st.radio("Navigation", ["📊 Dashboard", "🔍 Analysis", "⚙️ Systems"])
        
        st.divider()
        if os.path.exists('assets.json'):
            with open('assets.json', 'r', encoding='utf-8') as f:
                assets = json.load(f)
            selected_asset = st.selectbox("Select Asset", range(len(assets)), format_func=lambda i: f"{assets[i]['name']}")
            ticker = assets[selected_asset]['ticker']
            name = assets[selected_asset]['name']
        
        st.caption("Powered by Antigravity v7.7 Hybrid")

    # 데이터 로드
    data, error = load_web_data(ticker)
    if error:
        st.warning(error)
        return

    summary = data['summary']

    if menu == "📊 Dashboard":
        st.markdown(f"<div class='main-header'>{name} Monitoring</div>", unsafe_allow_html=True)
        st.caption(f"Last updated: {data['updated']}")

        # 반응형 컬럼 (모바일에서는 자동으로 위아래로 쌓임)
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1: st.metric("현재가", f"{summary['current_price']:,.0f}원")
        with col2: st.metric("AI Score", f"{data['latest_signal']['ai_score']:.2f}", f"Gap {data['latest_signal']['ai_score'] - data['latest_signal']['entry_threshold']:.2f}")
        with col3: st.metric("누적 수익률", f"{summary['total_return']*100:.1f}%")
        with col4: st.metric("연평균(CAGR)", f"{summary['cagr']*100:.1f}%")

        # 매매 신호
        latest_ai = data['latest_signal']['ai_score']
        threshold = data['latest_signal']['entry_threshold']
        tech = data['latest_signal']['tech_score']
        
        if latest_ai > threshold and tech > 0.3:
            st.markdown("<div class='signal-box signal-buy'>🟢 매수 진입 유효 (AI+기술 분석가 합의)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='signal-box signal-hold'>⚪ 관망 및 분석 중 (조건 탐색 단계)</div>", unsafe_allow_html=True)

        # 차트 영역 (좌우 배치, 모바일 시 위아래)
        c1, c2 = st.columns([3, 2])
        with c1: st.plotly_chart(create_performance_chart(data['daily_performance']), use_container_width=True)
        with c2: st.plotly_chart(create_monthly_chart(data['monthly_performance']), use_container_width=True)

    elif menu == "🔍 Analysis":
        st.markdown(f"## 🔍 {name} 상세 리포트")
        tab1, tab2 = st.tabs(["📅 Yearly Stats", "📝 Trade History"])
        
        with tab1:
            ydf = pd.DataFrame(data['yearly_performance'])
            ydf['return'] = ydf['return'].apply(lambda x: f"{x*100:.1f}%")
            ydf['win_rate'] = ydf['win_rate'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(ydf.rename(columns={'year':'연도', 'return':'수익률', 'trades':'거래', 'win_rate':'적중률', 'avg_hold':'보유일'}), use_container_width=True, hide_index=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe", f"{summary['sharpe']:.2f}")
            c2.metric("MDD", f"{summary['mdd']*100:.1f}%")
            c3.metric("Win/Loss", f"{summary['win_loss_ratio']:.2f}")

        with tab2:
            tdf = pd.DataFrame(data['trade_history'])
            st.table(tdf.rename(columns={'date':'날짜', 'price':'가격', 'signal':'신호', 'ai_score':'AI점수', 'tech_score':'기술점수'}))

    elif menu == "⚙️ Systems":
        st.markdown("## ⚙️ 시스템 상태")
        st.write(f"현재 참조 파일: `web_data_{ticker.replace('.KS','')}.json`")
        if st.button("🔄 캐시 새로고침"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
