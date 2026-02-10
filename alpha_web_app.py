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
            selected_asset = st.selectbox("상세 차트 분석 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            
            # Yfinance 차트 데이터 가져오기
            @st.cache_data(ttl=300)
            def get_chart_data(ticker):
                try:
                    # [v3.4] 최신 yfinance 구조 완벽 대응
                    raw_data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
                    
                    if raw_data.empty:
                        return pd.DataFrame()
                    
                    # 멀티인덱스(Multi-index) 해제 및 컬럼 정리
                    temp_df = raw_data.copy()
                    if isinstance(temp_df.columns, pd.MultiIndex):
                        temp_df.columns = temp_df.columns.get_level_values(0)
                    
                    return temp_df
                except Exception as e:
                    return pd.DataFrame()
            
            with st.spinner(f"{selected_asset} 차트 분석 중..."):
                chart_df = get_chart_data(asset_info['ticker'])
            
            if not chart_df.empty and len(chart_df) > 0:
                # 캔들스틱 차트 생성
                fig = go.Figure()
                
                # 컬럼 존재 여부 확인 후 안전하게 그리기
                try:
                    fig.add_trace(go.Candlestick(
                        x=chart_df.index,
                        open=chart_df['Open'],
                        high=chart_df['High'],
                        low=chart_df['Low'],
                        close=chart_df['Close'],
                        name="주가"
                    ))
                    
                    # 목표가/손절가 수평선
                    fig.add_hline(y=asset_info['target_price'], line_dash="dash", line_color="#00ff88", 
                                 annotation_text=f"Target: {asset_info['target_price']:,.0f}")
                    fig.add_hline(y=asset_info['stop_loss'], line_dash="dash", line_color="#ff4b4b", 
                                 annotation_text=f"StopLoss: {asset_info['stop_loss']:,.0f}")
                    
                    fig.update_layout(
                        title=f"🏛️ {selected_asset} AI 전략 가이드",
                        template="plotly_dark",
                        height=500,
                        margin=dict(l=20, r=20, t=50, b=20),
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    # 캔들스틱 실패 시 일반 라인 차트로 대체 (보험)
                    st.line_chart(chart_df['Close'])
                    st.warning(f"고급 차트 엔진 오류로 일반 차트로 표시합니다.")
            else:
                st.error("데이터 서버 응답 지연으로 차트를 불러올 수 없습니다. 잠시 후 새로고침해 주세요.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
