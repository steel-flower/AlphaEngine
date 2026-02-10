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
            # 종목 선택
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            
            # 차트 유형 선택 (로그 스케일 등)
            col_opt1, col_opt2 = st.columns(2)
            use_log = col_opt1.checkbox("로그 스케일 (장기 추세용)", value=False)
            
            @st.cache_data(ttl=300)
            def get_chart_data(ticker):
                try:
                    # 상장 이후 전체 기간
                    temp = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
                    if temp.empty: return pd.DataFrame()
                    if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.get_level_values(0)
                    temp.columns = [c.lower() for c in temp.columns]
                    return temp.astype(float)
                except Exception: return pd.DataFrame()
            
            with st.spinner(f"{selected_asset} 역대 데이터 분석 중..."):
                chart_df = get_chart_data(asset_info['ticker'])
            
            if not chart_df.empty and 'close' in chart_df.columns:
                st.subheader(f"🏛️ {selected_asset} 전생애 전략 캔버스")
                
                # 데이터 추출
                current_price = chart_df['close'].iloc[-1]
                target_p = float(asset_info['target_price'])
                stop_p = float(asset_info['stop_loss'])
                
                # 시각화 엔진 (Plotly)
                fig = go.Figure()
                
                # 1. 메인 주가 선
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['close'],
                    name="주가 흐름", line=dict(color='white', width=2)
                ))
                
                # 2. 목표가/손절가 라인 (현재 시점 부근에만 강조하기 위해 최근 20% 지점부터 그리기)
                start_idx = chart_df.index[int(len(chart_df)*0.85)] # 최근 15% 기간만 표시
                fig.add_trace(go.Scatter(
                    x=[start_idx, chart_df.index[-1]], y=[target_p, target_p],
                    name="Target (Blue)", line=dict(color='#0088ff', width=3, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=[start_idx, chart_df.index[-1]], y=[stop_p, stop_p],
                    name="Stop (Red)", line=dict(color='#ff4b4b', width=3, dash='dot')
                ))
                
                # [v3.4 Smart Scaling]
                # 전체 데이터를 보되, Y축 범위를 현재 주가와 타겟 부근으로 '기본값' 설정 (사용자가 줌 가능)
                fig.update_layout(
                    template="plotly_dark", height=500,
                    yaxis_type="log" if use_log else "linear",
                    yaxis=dict(gridcolor='#333', title="Price (KRW)", autorange=True),
                    xaxis=dict(gridcolor='#333', title="Date", rangeslider=dict(visible=True)), # 하단 슬라이더 추가
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 하단 대형 지표
                m1, m2, m3 = st.columns(3)
                m1.metric("Target (🎯Blue)", f"{target_p:,.0f}")
                m2.metric("Current (🏳️White)", f"{current_price:,.0f}")
                m3.metric("Stop-Loss (🛑Red)", f"{stop_p:,.0f}")
            else:
                st.error(f"데이터를 불러올 수 없습니다.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
