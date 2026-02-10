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
            
            # [Step 2] 데이터 로드 (전체 역사)
            @st.cache_data(ttl=300)
            def fetch_trend_data(t):
                try:
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    data = raw.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data.columns = [str(c).lower() for c in data.columns]
                    
                    # 월간 정제
                    m_data = data[['close']].resample('ME').last().dropna()
                    return m_data
                except:
                    return pd.DataFrame()

            chart_df = fetch_trend_data(ticker)
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 전략 캔버스")
                
                # [Step 3] Plotly를 이용한 정밀 시각화
                # 주가(Black), 매수/매도(Blue), 손절(Red)
                fig = go.Figure()
                
                # 1. 실제 주가 흐름 (Black)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['close'],
                    name="실제 주가", line=dict(color='white', width=2) # 다크모드 가독성을 위해 흰색 테두리
                ))
                
                # 2. 전략 수평선 (현재 시점 부근에만 강조)
                # 전체 데이터를 그리면 과거 데이터가 뭉쳐 보이므로 수평선은 현재 가격 기준으로 적절히 배치
                target_p = float(asset_info['target_price'])
                entry_p = float(asset_info['entry_price'])
                stop_p = float(asset_info['stop_loss'])
                
                # 수평선 추가 (전체 기간이 아닌 최근 영역에만 표시하여 과거 왜곡 방지)
                line_start = chart_df.index[int(len(chart_df)*0.9)] # 최근 10% 지점
                
                fig.add_trace(go.Scatter(
                    x=[line_start, chart_df.index[-1]], y=[target_p, target_p],
                    name="Blue: 전략 매도가", line=dict(color='#0088ff', width=3, dash='dash'),
                    mode='lines+text', text=["", f"Goal: {target_p:,.0f}"], textposition="top left"
                ))
                fig.add_trace(go.Scatter(
                    x=[line_start, chart_df.index[-1]], y=[entry_p, entry_p],
                    name="Blue: 전략 매수가", line=dict(color='#00AAFF', width=3, dash='dot'),
                    mode='lines+text', text=["", f"Entry: {entry_p:,.0f}"], textposition="bottom left"
                ))
                fig.add_trace(go.Scatter(
                    x=[line_start, chart_df.index[-1]], y=[stop_p, stop_p],
                    name="Red: 손절 방어선", line=dict(color='#FF4B4B', width=3, dash='longdash'),
                    mode='lines+text', text=["", f"Stop: {stop_p:,.0f}"], textposition="bottom left"
                ))
                
                # 가독성을 위한 레이아웃 조정
                fig.update_layout(
                    template="plotly_dark", height=550,
                    xaxis=dict(gridcolor='#333', rangeslider=dict(visible=True)),
                    yaxis=dict(gridcolor='#333', autorange=True, fixedrange=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=40, t=40, b=10)
                )
                
                # Y축 범위 자동 최적화 (현재 주가와 전략 라인이 잘 보이도록)
                # 사용자가 줌 가능하도록 설정
                st.plotly_chart(fig, use_container_width=True)
                
                # 🏛️ 하단 섹션 - 수치 정보
                st.info(f"💡 현재 **{selected_asset}**은 Alpha 시스템 분석 결과 **{asset_info['signal'].upper()}** 신호가 유지되고 있습니다.")
                m1, m2, m3 = st.columns(3)
                m2.metric("현재 주가", f"{chart_df['close'].iloc[-1]:,.0f}")
                m1.metric("Alpha 매도목표", f"{target_p:,.0f}")
                m3.metric("Alpha 안전바", f"{stop_p:,.0f}")
            else:
                st.error("데이터 로딩 중...")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
