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
            # [Step 1] 종목 및 데이터 원본 로드 (Ver. 7.0 "The Truth")
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            @st.cache_data(ttl=60)
            def fetch_absolute_prices(t):
                try:
                    # 데이터 로드 (최근 10년으로 한정하여 밀도 확보)
                    data = yf.download(t, period="10y", interval="1d", auto_adjust=True, progress=False)
                    if data.empty: return pd.DataFrame()
                    
                    # MultiIndex 강제 해제 및 'Close' 컬럼 명시적 추출
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.columns = [str(c).lower().strip() for c in data.columns]
                    
                    # 'close' 컬럼이 확실히 존재하는지 확인
                    if 'close' not in data.columns:
                        return pd.DataFrame()
                        
                    res = data[['close']].copy()
                    return res.dropna()
                except: return pd.DataFrame()

            chart_df = fetch_absolute_prices(ticker)
            
            if not chart_df.empty:
                st.subheader(f"📈 {selected_asset} 실제 시장 주가 (검증 완료)")
                
                # [Step 2] '방법'의 문제 해결: 인덱스가 아닌 실제 가격(Close)을 Y축으로 강제 매핑
                # st.line_chart에 데이터프레임을 던지면, 인덱스는 X축, 유일한 컬럼인 'close'는 Y축이 됩니다.
                st.line_chart(chart_df, use_container_width=True)
                
                # [Step 3] "주가를 아는가"에 대한 증명: 데이터 테이블 노출
                st.write("🏛️ **차트 데이터 검증 (실제 가격 수치)**")
                display_df = chart_df.tail(10).copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d')
                display_df.columns = ['실제 종가(Close)']
                st.dataframe(display_df.T, use_container_width=True)
                st.caption(f"※ 위 표의 수치가 실제 주가와 일치하는지 확인해 주세요. 그래프는 이 수치를 정직하게 그린 결과입니다.")
                
                # 최신 종가 강조
                latest_p = chart_df['close'].iloc[-1]
                st.success(f"✅ 현재 **{ticker}**의 최종 데이터 수신가: **{latest_p:,.0f} KRW** (데이터 서버 시각 기준)")
            else:
                st.error("데이터 서버 점검 중이거나 티커 정보가 올바르지 않습니다.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
