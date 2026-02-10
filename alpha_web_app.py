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
            # [Step 1] 종목 선택 및 데이터 로드 (일간 차트)
            selected_asset = st.selectbox("📊 상세 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            @st.cache_data(ttl=300)
            def fetch_daily_history(t):
                try:
                    # [v4.0] 사용자 요청: 정밀 일간 데이터 원본 로드
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    data = raw.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [c[0] for c in data.columns]
                    data.columns = [str(c).lower() for c in data.columns]
                    
                    if 'close' not in data.columns:
                        for c in data.columns:
                            if 'close' in c or 'adj' in c:
                                data['close'] = data[c]
                                break
                    return data[['close']].dropna()
                except: return pd.DataFrame()

            chart_df = fetch_daily_history(ticker)
            
            if not chart_df.empty:
                st.subheader(f"🏛️ {selected_asset} 실시간 일간 분석 캔버스")
                
                # 수치 변수
                prices = chart_df['close']
                curr = prices.iloc[-1]
                target = float(asset_info['target_price'])
                stop = float(asset_info['stop_loss'])
                
                # [Step 2] 고해상도 일간 렌더링
                fig = go.Figure()
                
                # 메인 주가 (일간 데일리 라인)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=prices,
                    name="일간 주가 흐름",
                    line=dict(color='black', width=1) # 데일리는 선을 조금 얇게 하여 디테일 살림
                ))
                
                # 전략선 구간 한정 (최근 60일 혹은 전체 10% 중 긴 쪽 선택)
                display_len = max(60, int(len(chart_df) * 0.1))
                segment_x = chart_df.index[-display_len:]
                
                fig.add_trace(go.Scatter(
                    x=segment_x, y=[target] * len(segment_x),
                    name="Alpha Goal (Blue)",
                    line=dict(color='blue', width=2, dash='dash'),
                    mode='lines+text', text=["", f"Goal {target:,.0f}"], textposition="top left"
                ))
                
                fig.add_trace(go.Scatter(
                    x=segment_x, y=[stop] * len(segment_x),
                    name="Risk Floor (Red)",
                    line=dict(color='red', width=2, dash='dot'),
                    mode='lines+text', text=["", f"Stop {stop:,.0f}"], textposition="bottom left"
                ))

                # [Step 3] 시야 최적화 (최근 1년치 집중 조명)
                # 초기 범위를 최근 1년으로 설정하여 일간 변동성이 즉시 보이게 함
                one_year_ago = chart_df.index[-1] - pd.Timedelta(days=365)
                recent_p = chart_df.loc[chart_df.index >= one_year_ago, 'close']
                if recent_p.empty: recent_p = prices.tail(100)
                
                y_min = min(recent_p.min(), stop) * 0.98
                y_max = max(recent_p.max(), target) * 1.02
                
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=600,
                    yaxis_type="log", # 장기 데이터의 선형 뭉침 방지
                    yaxis=dict(
                        gridcolor='#eee', 
                        autorange=False, 
                        range=[np.log10(y_min), np.log10(y_max)] if y_min > 0 else None,
                        title="Price (KRW, Daily Log Scale)",
                        side="right", tickformat=',.0f'
                    ),
                    xaxis=dict(
                        gridcolor='#eee', 
                        title="Timeline",
                        range=[one_year_ago, chart_df.index[-1]], # 초기 시야: 최근 1년
                        rangeslider=dict(visible=True) # 전 기간 수동 탐색 가능
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=40, t=50, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 수치 브리핑
                c1, c2, c3 = st.columns(3)
                c2.info(f"💎 현재가: {curr:,.0f}")
                c1.success(f"🎯 목표가: {target:,.0f}")
                c3.error(f"🛑 손절가: {stop:,.0f}")
            else:
                st.error("데이터 서버 로딩 실패 (종목 시스템 점검 중)")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
