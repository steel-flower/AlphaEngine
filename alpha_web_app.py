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
                    # 최신 yfinance 구조 대응 (auto_adjust=True로 데이터 표준화)
                    raw = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    # 멀티인덱스 및 컬럼명 정리
                    temp = raw.copy()
                    if isinstance(temp.columns, pd.MultiIndex):
                        temp.columns = temp.columns.get_level_values(0)
                    
                    # 모든 컬럼명을 소문자로 통일하여 검색 (유연한 대응)
                    temp.columns = [c.lower() for c in temp.columns]
                    return temp
                except Exception:
                    return pd.DataFrame()
            
            with st.spinner(f"{selected_asset} 리서치 중..."):
                chart_df = get_chart_data(asset_info['ticker'])
            
            if not chart_df.empty:
                try:
                    # OHLC 컬럼을 이름이 아닌 순서나 키워드로 추출 (매우 중요)
                    o = chart_df.get('open', pd.Series())
                    h = chart_df.get('high', pd.Series())
                    l = chart_df.get('low', pd.Series())
                    c = chart_df.get('close', pd.Series())
                    
                    if not c.empty:
                        fig = go.Figure()
                        # 캔들스택 데이터가 완전할 때만 실행
                        if not o.empty and not h.empty:
                            fig.add_trace(go.Candlestick(
                                x=chart_df.index, open=o, high=h, low=l, close=c, name="주가"
                            ))
                        else:
                            fig.add_trace(go.Scatter(x=chart_df.index, y=c, mode='lines', line=dict(color='#00ff88')))
                        
                        # 목표가/손절가 수평선
                        fig.add_hline(y=asset_info['target_price'], line_dash="dash", line_color="#00ff88")
                        fig.add_hline(y=asset_info['stop_loss'], line_dash="dash", line_color="#ff4b4b")
                        
                        fig.update_layout(
                            title=f"🏛️ {selected_asset} AI 전략 대시보드",
                            template="plotly_dark", height=500, xaxis_rangeslider_visible=False,
                            margin=dict(l=10, r=10, t=50, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("차트 데이터를 렌더링할 수 없습니다.")
                except Exception as e:
                    st.line_chart(chart_df) # 최후의 수단
            
            # [Fact Check] 데이터 존재 여부 확인용 테이블 (하단 배치)
            with st.expander("📊 수신된 데이터 로우(Raw) 확인"):
                st.dataframe(chart_df.tail(10))

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
