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
            # [Step 1] 종목 및 데이터 원본 로드
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            @st.cache_data(ttl=60) # 유효시간 단축하여 실시간성 확보
            def fetch_real_data(t):
                try:
                    # [v6.0] 가장 정직한 download 방식 사용 (auto_adjust=True)
                    data = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
                    if data.empty: return pd.DataFrame()
                    
                    # 컬럼 구조 단순화 (MultiIndex 파괴)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.columns = [str(c).lower().strip() for c in data.columns]
                    
                    # 'close'만 추출하여 날짜별 정렬
                    clean_df = data[['close']].copy()
                    clean_df.index = pd.to_datetime(clean_df.index)
                    return clean_df.sort_index()
                except: return pd.DataFrame()

            chart_df = fetch_real_data(ticker)
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 실제 시장 주가 (Raw Graph)")
                
                # [Step 2] 데이터 조작 없는 '순정 라인 차트'
                # st.line_chart는 데이터가 가진 굴곡을 그대로 표현하는 가장 정직한 도구입니다.
                st.line_chart(chart_df['close'], use_container_width=True)
                
                # [Step 3] 데이터 검증 테이블 (사용자 확신용)
                with st.expander("🏛️ 수치 데이터 직접 검증 (최근 5거래일)"):
                    st.write(f"현재 선택된 티커: **{ticker}**")
                    # 날짜 형식을 보기 좋게 변경하여 수치 공개
                    verify_df = chart_df.tail(5).copy()
                    verify_df.index = verify_df.index.strftime('%Y-%m-%d')
                    st.table(verify_df)
                    st.caption("※ 위 수치가 실제 시장가와 일치함을 확인해 주세요. 그래프는 이 숫자를 그대로 선으로 연결한 결과입니다.")
                
                # 핵심 수치
                curr_p = chart_df['close'].iloc[-1]
                st.info(f"✅ **실시간 검증**: {selected_asset}의 최종 종가 데이터는 **{curr_p:,.0f}**원이며, 차트는 이 값을 종점(Right End)으로 찍고 있습니다.")
            else:
                st.error("데이터 서버와 통신 중 문제가 발생했습니다. (Ticker 설정 오류 가능성)")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
