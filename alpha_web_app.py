import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# 페이지 설정
st.set_page_config(
    page_title="Alpha Engine v3.4 Live Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (프리미엄 미학)
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .status-buy { color: #00ff88; font-weight: bold; }
    .status-wait { color: #ffbc00; font-weight: bold; }
    .css-1offfwp { background-color: #161b22 !important; }
</style>
""", unsafe_allow_html=True)

# 비밀번호 보안 (간이)
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        st.title("🏛️ Alpha Engine 보안 접속")
        pwd = st.text_input("Access Password", type="password")
        if st.button("Unlock"):
            if pwd == "alpha77": # 기본 비밀번호
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid Password")
        return False
    return True

if check_password():
    # 사이드바 설정
    st.sidebar.title("🏛️ Alpha Engine v3.4")
    st.sidebar.markdown("---")
    st.sidebar.info("실시간 시장 감시 및 AI 전략 대시보드")
    
    # 데이터 로드
    def load_data():
        if os.path.exists("dashboard_data.json"):
            with open("dashboard_data.json", "r", encoding='utf-8') as f:
                return json.load(f)
        return []

    data = load_data()
    
    if not data:
        st.warning("경고: 실시간 데이터 파일을 찾을 수 없습니다. Monitor 프로그램을 먼저 실행하세요.")
    else:
        df = pd.DataFrame(data)
        last_update = df['timestamp'].iloc[0]
        
        st.title("🚀 Real-Time Market Intelligence")
        st.caption(f"Last Updated: {last_update} (5분 주기 자동 갱신)")

        # 상단 요약 지표
        cols = st.columns(len(df))
        for i, row in df.iterrows():
            with cols[i]:
                color = "normal" if row['signal'] == "wait" else "inverse"
                st.metric(
                    label=row['name'],
                    value=f"{row['price']:,.0f}",
                    delta=f"AI: {row['score']:.2f}",
                    delta_color=color
                )
        
        st.markdown("---")
        
        # 메인 분석 섹션
        col_list, col_chart = st.columns([1, 2])
        
        with col_list:
            st.subheader("📋 실시간 포지션 현황")
            display_df = df[['name', 'score', 'signal', 'potential_profit']].copy()
            display_df.columns = ['종목명', 'AI Score', '상태', '기대수익(%)']
            
            def color_signal(val):
                color = '#00ff88' if val == 'buy' else '#ffbc00'
                return f'color: {color}'
            
            st.dataframe(
                display_df.style.applymap(color_signal, subset=['상태'])
                .format({'AI Score': '{:.2f}', '기대수익(%)': '{:.1f}%'}),
                use_container_width=True,
                height=400
            )

        with col_chart:
            # [Step 1] 종목 선택 및 데이터 직접 호출
            selected_asset = st.selectbox("📊 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            # [Step 2] 데이터 확보 (기교 없이 순수하게)
            @st.cache_data(ttl=60)
            def fetch_plain_price(t):
                try:
                    # yfinance에서 최근 5년치 일간 종가만 원본 그대로 가져옴
                    raw = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    # MultiIndex 구조라면 최하단 가격 정보만 취득
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    
                    # 모든 컬럼명을 소문자로 통일하고 'Close'만 추출
                    raw.columns = [str(c).lower() for c in raw.columns]
                    if 'close' not in raw.columns: return pd.DataFrame()
                    
                    return raw[['close']].astype(float)
                except: return pd.DataFrame()

            price_df = fetch_plain_price(ticker)
            
            if not price_df.empty:
                st.subheader(f"🏛️ {selected_asset} 실데이터 기반 주가 검증")
                
                # [Step 3] 데이터 실명제: 그래프를 그리는 근거 수치 노출
                st.write("📈 **그래프 생성을 위한 로우 데이터 (최신 7일)**")
                verify_table = price_df.tail(7).copy()
                verify_table.index = verify_table.index.strftime('%Y-%m-%d')
                verify_table.columns = ['종가 (Y축 값)']
                st.table(verify_table.T)
                
                # [Step 4] "X=시간, Y=가격" 원칙에 따른 선긋기
                # 기교 없는 st.line_chart를 사용하여 데이터 포인트간의 직선 연결만 허용
                st.line_chart(price_df['close'], use_container_width=True)
                
                # 결과 보고
                last_price = price_df['close'].iloc[-1]
                st.info(f"✅ 위 차트는 **{ticker}**의 실제 데이터 포인트들을 선형(Linear) 눈금 위에 정직하게 연결한 결과입니다. (최종가: {last_price:,.0f} KRW)")
            else:
                st.error("데이터 서버 응답 대기 중이거나 티커명이 올바르지 않습니다.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
