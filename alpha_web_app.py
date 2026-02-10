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
            selected_asset = st.sidebar.selectbox("상세 차트 분석 선택", df['name'].tolist()) if 'df' in locals() else st.selectbox("상세 차트 분석 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            
            @st.cache_data(ttl=300)
            def get_chart_data(ticker):
                try:
                    temp = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
                    if temp.empty: return pd.DataFrame()
                    if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.get_level_values(0)
                    temp.columns = [c.lower() for c in temp.columns]
                    return temp.astype(float)
                except Exception: return pd.DataFrame()
            
            chart_df = get_chart_data(asset_info['ticker'])
            
            if not chart_df.empty:
                st.subheader(f"📈 {selected_asset} 전략 캔버스")
                
                # 시인성 극대화 차트 (Native Line Chart with Target/Stop)
                # 사용자 요청 색상 적용: 주가(Black), 목표(Blue), 손절(Red)
                # 데이터 준비
                p_df = chart_df[['close']].copy()
                p_df.columns = ['현재가']
                p_df['목표가'] = asset_info['target_price']
                p_df['손절가'] = asset_info['stop_loss']
                
                # [v3.4 Premium Dynamic Scaling]
                y_min = min(p_df['현재가'].min(), asset_info['stop_loss']) * 0.98
                y_max = max(p_df['현재가'].max(), asset_info['target_price']) * 1.02
                
                # 스트림릿 차트는 색상 지정이 제한적이므로 Plotly로 고도화된 색상 적용
                fig = go.Figure()
                # 주가 (검정)
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['현재가'], name="현재가", line=dict(color='white', width=3))) 
                # 목표가 (파랑)
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['목표가'], name="목표가", line=dict(color='#0088ff', dash='dash')))
                # 손절가 (빨강)
                fig.add_trace(go.Scatter(x=p_df.index, y=p_df['손절가'], name="손절가", line=dict(color='#ff4b4b', dash='dash')))
                
                fig.update_layout(
                    template="plotly_dark", height=550,
                    yaxis=dict(range=[y_min, y_max], gridcolor='#333', title="Price (KRW)"),
                    xaxis=dict(gridcolor='#333'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 정보 요약
                c1, c2, c3 = st.columns(3)
                c1.metric("Target (Blue)", f"{asset_info['target_price']:,.0f}")
                c2.metric("Current (White)", f"{p_df['현재가'].iloc[-1]:,.0f}")
                c3.metric("Stop-Loss (Red)", f"{asset_info['stop_loss']:,.0f}")
            else:
                st.error("데이터 로드 중...")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
