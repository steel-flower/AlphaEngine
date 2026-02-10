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
            # [Step 1] 종목 선택 및 정보 추출
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            # [Step 2] 데이터 로드 (가장 단순하고 확실한 방식)
            @st.cache_data(ttl=300)
            def fetch_simple_data(t):
                try:
                    # 일간 데이터를 가져와서 강제로 평탄화
                    raw = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    # 컬럼 정리 (MultiIndex 방어)
                    data = raw.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.columns = [str(c).lower() for c in data.columns]
                    
                    # 월간 리샘플링
                    m_data = data[['close']].resample('ME').last().dropna()
                    return m_data
                except:
                    return pd.DataFrame()

            chart_df = fetch_simple_data(ticker)
            
            # [Step 3] 차트 렌더링 (가장 시인성 높은 기본형으로 회귀)
            if not chart_df.empty:
                st.subheader(f"📈 {selected_asset} 실시간 추세 분석")
                
                # Plotly 엔진의 복잡한 설정을 버리고 Streamlit의 순정 차트로 승부
                # 시인성 확보를 위해 목표가/손절가를 차트 데이터에 합쳐서 전송
                viz_df = chart_df.copy()
                viz_df['Target'] = float(asset_info['target_price'])
                viz_df['StopLoss'] = float(asset_info['stop_loss'])
                
                # 사용자 요청 색상 반영 (흰색/파란색/빨간색)
                st.line_chart(
                    viz_df, 
                    color=["#FFFFFF", "#0088ff", "#ff4b4b"], # 현재가(White), 목표(Blue), 손절(Red)
                    height=450
                )
                
                # 하단에 큰 수치로 정보 보강
                c1, c2, c3 = st.columns(3)
                c2.metric("현재가 (🏳️White)", f"{chart_df['close'].iloc[-1]:,.0f}")
                c1.metric("목표가 (🎯Blue)", f"{asset_info['target_price']:,.0f}")
                c3.metric("손절가 (🛑Red)", f"{asset_info['stop_loss']:,.0f}")
                
                st.caption("※ 그래프가 너무 뭉쳐 보일 경우, 마우스 휠로 확대/축소하거나 우측 상단 메뉴에서 전체 화면으로 보실 수 있습니다.")
            else:
                st.error("차트 데이터를 불러오는 데 실패했습니다. 종목 코드를 확인하세요.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
