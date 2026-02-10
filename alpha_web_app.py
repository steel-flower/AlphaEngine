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
            # [Step 1] 종목 선택 및 데이터 호출
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            
            @st.cache_data(ttl=300)
            def fetch_raw_history(t):
                try:
                    # 데이터 왜곡을 막기 위해 '일간' 원본 데이터를 그대로 가져옴
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                    raw.columns = [str(c).lower() for c in raw.columns]
                    return raw[['close']].dropna()
                except: return pd.DataFrame()

            chart_df = fetch_raw_history(asset_info['ticker'])
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 전생애 주가 흐름 및 전략")
                
                # 수치 데이터
                curr = chart_df['close'].iloc[-1]
                target = float(asset_info['target_price'])
                stop = float(asset_info['stop_loss'])
                
                # [Step 2] 변동성 복원 엔진 (Plotly)
                fig = go.Figure()
                
                # 메인 주가 (검정색 굵은 선)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['close'],
                    name="실제 주가 (Price)", line=dict(color='black', width=2)
                ))
                
                # 전략 가이드선 (Plotly 전용 h-line 사용 - 스케일 방해 최소화)
                fig.add_hline(y=target, line_dash="dash", line_color="blue", annotation_text=f"Sell: {target:,.0f}")
                fig.add_hline(y=stop, line_dash="dot", line_color="red", annotation_text=f"Stop: {stop:,.0f}")
                
                # [v3.7 핵심: 로그 스케일 및 자동 스케일링]
                # 십수년치를 볼 때 수평선으로 보이지 않게 하는 유일한 방법
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=600,
                    yaxis_type="log", # 로그 스케일로 전생애 굴곡 복원
                    yaxis=dict(
                        gridcolor='#eee', 
                        autorange=True, # 시스템이 개입하지 않고 엔진이 굴곡을 찾음
                        title="Price (KRW, Log Scale)",
                        side="right" # 눈금을 우측으로 옮겨 시인성 확보
                    ),
                    xaxis=dict(gridcolor='#eee', title="Date", rangeslider=dict(visible=True)),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=50, t=50, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 실제 데이터 값 PROVE (표기)
                col1, col2, col3 = st.columns(3)
                col1.info(f"🎯 목표(Sell): {target:,.0f}")
                col2.success(f"💎 현재(Live): {curr:,.0f}")
                col3.error(f"🛑 손절(Risk): {stop:,.0f}")
            else:
                st.error("차트 데이터를 불러오는 중입니다...")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
