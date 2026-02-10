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
            # [Step 1] 종목 선택 및 데이터 정제
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            
            @st.cache_data(ttl=300)
            def fetch_clean_data(t):
                try:
                    # 상장 이후 전체 (월간)
                    raw = yf.download(t, period="max", interval="1mo", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                    raw.columns = [str(c).lower() for c in raw.columns]
                    # 시계열 인덱스 명확화
                    raw.index = pd.to_datetime(raw.index)
                    return raw[['close']].dropna()
                except: return pd.DataFrame()

            chart_df = fetch_clean_data(asset_info['ticker'])
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 리서치 리포트")
                
                # 수치 정보
                curr = chart_df['close'].iloc[-1]
                target = float(asset_info['target_price'])
                entry = float(asset_info['entry_price'])
                stop = float(asset_info['stop_loss'])
                
                # [Step 2] 시인성 극대화 차트 (White Theme + Black/Blue/Red)
                fig = go.Figure()
                
                # 메인 주가 (검정색 굵은 선 - 화이트 배경에서 최강의 시인성)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['close'],
                    name="실제 주가", line=dict(color='#000000', width=3)
                ))
                
                # 전략 라인 (최신 기간에만 짧게 표시하여 '긴 수평선' 문제 해결)
                # 최근 20% 지점 혹은 최근 12개월 중 짧은 쪽 선택
                line_len = min(12, int(len(chart_df)*0.2))
                line_x = chart_df.index[-line_len:]
                
                # 익절 목표가 (Blue)
                fig.add_trace(go.Scatter(
                    x=line_x, y=[target]*len(line_x),
                    name="Blue: 전략 매도가", line=dict(color='blue', width=3, dash='dash')
                ))
                # 안전 손절가 (Red)
                fig.add_trace(go.Scatter(
                    x=line_x, y=[stop]*len(line_x),
                    name="Red: 손절 방어선", line=dict(color='red', width=3, dash='dot')
                ))
                
                # [v3.6 Dynamic Focusing]
                # 최근 데이터 범위로 초기 시야 고정 (과거 데이터로 인한 뭉침 방지)
                recent_data = chart_df['close'].tail(line_len * 2) if len(chart_df) > line_len*2 else chart_df['close']
                y_min = min(recent_data.min(), stop) * 0.95
                y_max = max(recent_data.max(), target) * 1.05
                
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=500,
                    xaxis=dict(gridcolor='#eee', rangeslider=dict(visible=True)),
                    yaxis=dict(gridcolor='#eee', range=[y_min, y_max], autorange=False, title="Price (KRW)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=40, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 하단 수치 가이드
                c1, c2, c3 = st.columns(3)
                c2.metric("현재가 (⚫Black)", f"{curr:,.0f}")
                c1.metric("매도목표 (🔵Blue)", f"{target:,.0f}", f"{(target/curr-1)*100:+.1f}%")
                c3.metric("손절선 (🔴Red)", f"{stop:,.0f}", f"{(stop/curr-1)*100:+.1f}%")
            else:
                st.error("데이터 통신 중입니다. 잠시만 기다려주세요.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
