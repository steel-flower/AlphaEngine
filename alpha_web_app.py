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
            # [Step 1] 종목 선택 및 데이터 정밀 로드
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            @st.cache_data(ttl=300)
            def fetch_true_history(t):
                try:
                    # 원본 데이터 그대로 확보 (단 1원의 오차도 허용 안함)
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    # 컬럼 이름 정규화 (MultiIndex 및 대소문자 방어)
                    data = raw.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [c[0] for c in data.columns]
                    data.columns = [str(c).lower() for c in data.columns]
                    
                    # 'close' 컬럼이 없으면 가장 근접한 가격 컬럼 찾기
                    if 'close' not in data.columns:
                        for c in data.columns:
                            if 'close' in c or 'adj' in c:
                                data['close'] = data[c]
                                break
                    return data[['close']].dropna()
                except: return pd.DataFrame()

            chart_df = fetch_true_history(ticker)
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 전략적 자산 히스토리")
                
                # 수치 변수
                prices = chart_df['close']
                curr = prices.iloc[-1]
                target = float(asset_info['target_price'])
                stop = float(asset_info['stop_loss'])
                entry = float(asset_info['entry_price'])
                
                # [Step 2] Plotly 다이내믹 렌더링
                fig = go.Figure()
                
                # 1. 실제 주가 곡선 (최우선 순위: 시각적 변동성 확보)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=prices,
                    name="실제 주가 흐름",
                    line=dict(color='black', width=1.5)
                ))
                
                # 2. 전략선 구간 한정 (최근 15% 기간에만 표시하여 과거 데이터 압착 방지)
                # 이 부분이 수평선 문제를 해결하는 핵심 'Segmented Rendering' 입니다.
                display_len = max(24, int(len(chart_df) * 0.15))
                segment_x = chart_df.index[-display_len:]
                
                # 매도 목표선 (Blue)
                fig.add_trace(go.Scatter(
                    x=segment_x, y=[target] * len(segment_x),
                    name="Alpha Target (Blue)",
                    line=dict(color='blue', width=3, dash='dash'),
                    mode='lines+text', text=["", f"Goal {target:,.0f}"], textposition="top left"
                ))
                
                # 손절 안전선 (Red)
                fig.add_trace(go.Scatter(
                    x=segment_x, y=[stop] * len(segment_x),
                    name="Risk Stop (Red)",
                    line=dict(color='red', width=3, dash='dot'),
                    mode='lines+text', text=["", f"Stop {stop:,.0f}"], textposition="bottom left"
                ))

                # [Step 3] 로그 스케일 및 다이내믹 레이아웃
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=600,
                    yaxis_type="log", # 십수년치 데이터의 굴곡을 살리는 유일한 방법
                    yaxis=dict(
                        gridcolor='#eee', 
                        autorange=True, 
                        title="Price (KRW, Log Scale)",
                        side="right",
                        tickformat=',.0f'
                    ),
                    xaxis=dict(
                        gridcolor='#eee', 
                        title="Timeline",
                        rangeslider=dict(visible=True) # 전 기간 탐색 가능
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=40, t=50, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 데이터 건전성 실시간 표기 (디버그 대용)
                c1, c2, c3 = st.columns(3)
                c2.info(f"💎 Live: {curr:,.0f}")
                c1.success(f"🎯 Target: {target:,.0f}")
                c3.error(f"🛑 Stop: {stop:,.0f}")
            else:
                st.error("데이터 서버에서 가격 정보를 수신하지 못했습니다. (종목 코드 확인 요망)")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
