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
            # [Step 1] 종목 및 기초 데이터 로드
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker_symbol = asset_info['ticker']
            
            @st.cache_data(ttl=300)
            def fetch_robust_history(t):
                try:
                    # [v5.0] 가장 원시적이고 확실한 Ticker.history() 방식 사용
                    obj = yf.Ticker(t)
                    data = obj.history(period="max")
                    if data.empty:
                        # 'max' 실패 시 대안 기간 시도
                        data = obj.history(period="10y")
                    
                    if data.empty: return pd.DataFrame()
                    
                    # 컬럼 정규화 (history()는 보통 MultiIndex가 아님)
                    data.columns = [str(c).lower().strip() for c in data.columns]
                    
                    # 'close' 컬럼 확보
                    if 'close' not in data.columns:
                        return pd.DataFrame()
                    
                    return data[['close']].astype(float).dropna()
                except: return pd.DataFrame()

            chart_df = fetch_robust_history(ticker_symbol)
            
            if not chart_df.empty:
                st.subheader(f"📈 {selected_asset} 실제 거래 가격 기록")
                
                # [Step 2] 시각화 (순수 정량 데이터 차트)
                fig = go.Figure()
                
                # 블랙 라인 (주가 변동 실시간 재현)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['close'],
                    name="주가 흐름",
                    line=dict(color='#000000', width=1.5)
                ))
                
                # [Step 3] 가독성 레이아웃 (White & Sharp)
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=600,
                    yaxis_type="log", # 역사적 파동 보존을 위한 로그 스케일
                    yaxis=dict(
                        gridcolor='#f0f0f0', 
                        autorange=True, 
                        title="Price (KRW, Log Scale)",
                        side="right", tickformat=',.0f'
                    ),
                    xaxis=dict(
                        gridcolor='#f0f0f0', 
                        title="Timeline",
                        rangeslider=dict(visible=True)
                    ),
                    margin=dict(l=10, r=40, t=30, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 데이터 유효성 증명 레이블
                curr_val = chart_df['close'].iloc[-1]
                st.success(f"🏛️ **데이터 검증 완료**: {selected_asset}의 최근 종가는 {curr_val:,.0f}원 입니다.")
            else:
                st.error(f"❌ '{selected_asset}' 종목의 데이터를 서버에서 가져오지 못했습니다. 시장 데이터 제공사의 일시적 오류일 수 있으니 잠 잠시 후 다시 시도해 주세요.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
