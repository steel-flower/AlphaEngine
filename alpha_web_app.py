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
            # [Step 1] 종목 선택 및 데이터 로드
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            
            @st.cache_data(ttl=300)
            def fetch_full_history(t):
                try:
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                    raw.columns = [str(c).lower() for c in raw.columns]
                    return raw[['close']].resample('ME').last().dropna()
                except: return pd.DataFrame()

            chart_df = fetch_full_history(asset_info['ticker'])
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 전략 캔버스 (마스터 뷰)")
                
                # 수치 데이터
                prices = chart_df['close']
                curr = prices.iloc[-1]
                target = float(asset_info['target_price'])
                entry = float(asset_info['entry_price'])
                stop = float(asset_info['stop_loss'])
                
                # [Step 2] Plotly 고성능 엔진
                fig = go.Figure()
                
                # 메인 주가 (검정/화이트 대비)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=prices,
                    name="주가 흐름", line=dict(color='white', width=2)
                ))
                
                # 전략 가이드선 (차트 전체를 가로지르는 h-line 사용)
                fig.add_hline(y=target, line_dash="dash", line_color="#0088ff", annotation_text=f"Sell: {target:,.0f}", annotation_position="top right")
                fig.add_hline(y=entry, line_dash="dot", line_color="#00AAFF", annotation_text=f"Buy: {entry:,.0f}", annotation_position="bottom right")
                fig.add_hline(y=stop, line_dash="longdash", line_color="#FF4B4B", annotation_text=f"Stop: {stop:,.0f}", annotation_position="bottom right")
                
                # [v3.5 Smart Focus]
                # 최근 3년 데이터와 전략 라인이 모두 보기에 가장 좋은 범위를 계산
                recent_p = prices.tail(36)
                y_min = min(recent_p.min(), stop, entry) * 0.95
                y_max = max(recent_p.max(), target) * 1.05
                
                fig.update_layout(
                    template="plotly_dark", height=550,
                    xaxis=dict(gridcolor='#333', rangeslider=dict(visible=True)),
                    yaxis=dict(gridcolor='#333', range=[y_min, y_max], autorange=False, title="Price (KRW)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=60, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 지표 카드
                m1, m2, m3 = st.columns(3)
                m2.metric("현재가", f"{curr:,.0f}")
                m1.metric("Alpha 매도목표", f"{target:,.0f}", f"{(target/curr-1)*100:+.1f}%")
                m3.metric("Alpha 손절안전", f"{stop:,.0f}", f"{(stop/curr-1)*100:+.1f}%")
            else:
                st.error("데이터 서버 응답 대기 중...")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
