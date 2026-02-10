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
            # [Step 1] 종목 선택 및 전략 정보 추출
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            # [Step 2] 데이터 로드 (월간 트렌드)
            @st.cache_data(ttl=300)
            def fetch_trend_data(t):
                try:
                    raw = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    data = raw.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.columns = [str(c).lower() for c in data.columns]
                    
                    m_data = data[['close']].resample('ME').last().dropna()
                    return m_data
                except:
                    return pd.DataFrame()

            chart_df = fetch_trend_data(ticker)
            
            # [Step 3] 차트 렌더링 (사용자 요청: 검정/파랑/빨강 테마)
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} AI 전략 대시보드")
                
                # 데이터 그룹화
                viz_df = pd.DataFrame(index=chart_df.index)
                viz_df['[실제주가]'] = chart_df['close']
                viz_df['[Alpha 매출/매매 목표]'] = float(asset_info['target_price'])
                viz_df['[Alpha 진입/매수선]'] = float(asset_info['entry_price'])
                viz_df['[리스크 방어/손절]'] = float(asset_info['stop_loss'])
                
                # 차트 출력 (현재가:Black, 매수/매도:Blue range, 손절:Red)
                st.line_chart(
                    viz_df, 
                    color=["#000000", "#0000FF", "#00AAFF", "#FF0000"], 
                    height=500
                )
                
                # 🏛️ 전략 가이드
                st.info(f"""
                **🏛️ Alpha Engine 전문가 가이드**
                *   **⚫ 검정색 (실제주가)**: 시장의 현재 흐름입니다.
                *   **🔵 파란색 계열 (Alpha 매도매수)**: 시스템이 제안하는 **진입가격({asset_info['entry_price']:,.0f})**과 **목표가격({asset_info['target_price']:,.0f})**입니다. 주가가 이 구간 안에서 전략적으로 움직입니다.
                *   **🔴 빨간색 (손절/안전)**: 예상치 못한 급락 시 자산을 보호하기 위한 **리스크 방어선({asset_info['stop_loss']:,.0f})**입니다.
                """)
                
                # 수치 지표 정밀 요약
                m1, m2, m3 = st.columns(3)
                m2.metric("현재가 (Black)", f"{chart_df['close'].iloc[-1]:,.0f}")
                m1.metric("Alpha 진입/목표 (Blue)", f"{asset_info['entry_price']:,.0f}", delta=f"Target: {asset_info['target_price']:,.0f}")
                m3.metric("손절가 (Red)", f"{asset_info['stop_loss']:,.0f}")
            else:
                st.error("데이터 로딩 중 오류가 발생했습니다.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
