"""
Alpha Engine v7.7 - Streamlit Web Application (Hybrid Mode)
로컬의 정밀 분석 데이터를 서버(웹)로 불러와 호출하는 전문 공시 모드
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

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .signal-hold {
        background: linear-gradient(135deg, #bdc3c7 0%, #95a5a6 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 인증 시스템
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets.get("password", "alpha2026"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<div class='main-header'>🔐 Alpha Engine v7.7</div>", unsafe_allow_html=True)
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("🔑 비밀번호를 입력하세요", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.error("❌ 비밀번호가 올바르지 않습니다.")
        return False
    return True

# 데이터 로드 (Hybrid: 로컬 JSON 파일 호출)
@st.cache_data(ttl=600) # 10분 캐시
def load_web_data(ticker):
    ticker_clean = ticker.replace('.KS', '')
    file_path = f"web_data_{ticker_clean}.json"
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    else:
        return None, f"로컬 분석 데이터 파일({file_path})을 찾을 수 없습니다. 로컬에서 먼저 분석을 실행해주세요."

def create_performance_chart(daily_data):
    df = pd.DataFrame(daily_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=(df['strategy']-1)*100, name='Alpha Engine', line=dict(color='#667eea')))
    fig.add_trace(go.Scatter(x=df['date'], y=(df['market']-1)*100, name='시장 (Buy&Hold)', line=dict(color='#f45c43', dash='dash')))
    fig.update_layout(title='누적 수익률 (상장 이후 전체 기간)', template='plotly_white', height=450)
    return fig

def create_monthly_chart(monthly_data):
    """월간 실적율 차트 (사용자 요청 기능)"""
    df = pd.DataFrame(monthly_data)
    # 최근 12개월만 필터링
    df = df.tail(12)
    
    # 색상 결정 (양수면 파랑, 음수면 빨강)
    colors = ['#667eea' if x >= 0 else '#f45c43' for x in df['return']]
    
    fig = px.bar(df, x='month', y='return', title='최근 12개월 월간 수익률 (%)',
                 text=df['return'].apply(lambda x: f"{x*100:.1f}%"))
    fig.update_traces(marker_color=colors, textposition='outside')
    fig.update_layout(yaxis_tickformat='.1%', template='plotly_white', height=350)
    return fig

def main():
    if not check_password(): return
    
    st.markdown("<div class='main-header'>📈 Alpha Engine v7.7 (Hybrid)</div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 메뉴")
        menu = st.radio("선택", ["📊 대시보드", "🔍 상세 성과", "⚙️ 설정"], label_visibility="collapsed")
        
        st.markdown("---")
        if os.path.exists('assets.json'):
            with open('assets.json', 'r', encoding='utf-8') as f:
                assets = json.load(f)
            selected_asset = st.selectbox("종목 선택", range(len(assets)), format_func=lambda i: f"{assets[i]['name']} ({assets[i]['ticker']})")
            ticker = assets[selected_asset]['ticker']
            name = assets[selected_asset]['name']
        else:
            st.error("assets.json이 없습니다.")
            return

    # 데이터 로드
    data, error = load_web_data(ticker)
    
    if error:
        st.warning(error)
        st.info("💡 작동 원리: 본 시스템은 로컬 컴퓨터의 정밀 분석 결과를 웹으로 호출합니다. 로컬에서 해당 종목을 먼저 분석해주세요.")
        return

    summary = data['summary']
    
    if menu == "📊 대시보드":
        st.markdown(f"## 📊 {name} 실시간 모니터링")
        st.caption(f"최종 업데이트: {data['updated']} (정밀 분석 데이터)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재가", f"{summary['current_price']:,.0f}원")
        col2.metric("AI 점수", f"{data['latest_signal']['ai_score']:.2f}", f"기준 {data['latest_signal']['entry_threshold']:.2f}")
        col3.metric("누적 수익률", f"{summary['total_return']*100:.1f}%")
        col4.metric("연평균 (CAGR)", f"{summary['cagr']*100:.1f}%")
        
        st.markdown("---")
        # 매매 신호
        latest_ai = data['latest_signal']['ai_score']
        threshold = data['latest_signal']['entry_threshold']
        tech = data['latest_signal']['tech_score']
        
        if latest_ai > threshold and tech > 0.3:
            st.markdown("<div class='signal-buy'>🟢 매수 진입 권장 (AI+기술 조건 충족)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='signal-hold'>⚪ 관망 상태 (조건 미달)</div>", unsafe_allow_html=True)
            
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_performance_chart(data['daily_performance']), use_container_width=True)
        with col2:
            st.plotly_chart(create_monthly_chart(data['monthly_performance']), use_container_width=True)

    elif menu == "🔍 상세 성과":
        st.markdown(f"## 🔍 {name} 정밀 분석 리포트")
        
        tab1, tab2 = st.tabs(["📅 연도별 성과", "📝 거래 내역"])
        
        with tab1:
            ydf = pd.DataFrame(data['yearly_performance'])
            ydf['return'] = ydf['return'].apply(lambda x: f"{x*100:.1f}%")
            ydf['win_rate'] = ydf['win_rate'].apply(lambda x: f"{x*100:.1f}%")
            ydf.columns = ['연도', '수익률', '거래횟수', '적중률', '평균보유일']
            st.dataframe(ydf, use_container_width=True, hide_index=True)
            
            st.markdown("### 💡 성과 요약")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe Ratio", f"{summary['sharpe']:.2f}")
            col2.metric("MDD (최대낙폭)", f"{summary['mdd']*100:.2f}%")
            col3.metric("손익비", f"{summary['win_loss_ratio']:.2f}")

        with tab2:
            st.markdown("### 최근 10건의 정밀 거래 내역")
            tdf = pd.DataFrame(data['trade_history'])
            tdf.columns = ['날짜', '가격', '신호', 'AI점수', '기술점수']
            st.table(tdf)

    elif menu == "⚙️ 설정":
        st.markdown("## ⚙️ 시스템 설정 (Hybrid Mode)")
        st.info("본 앱은 로컬 환경의 `web_data_*.json` 파일을 참조합니다.")
        st.write(f"현재 참조 중인 파일: `web_data_{ticker.replace('.KS','')}.json`")
        if st.button("🔄 캐시 강제 새로고침"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
