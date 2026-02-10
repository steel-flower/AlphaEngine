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
            
            # Yfinance 데이터 로드
            @st.cache_data(ttl=300)
            def get_chart_data(ticker):
                try:
                    # 데이터 표준화 호출
                    temp = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
                    if temp.empty: return pd.DataFrame()
                    # 컬럼 평탄화 및 소문자화
                    if isinstance(temp.columns, pd.MultiIndex):
                        temp.columns = temp.columns.get_level_values(0)
                    temp.columns = [c.lower() for c in temp.columns]
                    return temp.astype(float)
                except Exception:
                    return pd.DataFrame()
            
            chart_df = get_chart_data(asset_info['ticker'])
            
            if not chart_df.empty:
                st.subheader(f"📈 {selected_asset} 추세 분석")
                
                # [Engine 1] 무조건 그려지는 Native Line Chart
                # 종가(Close)와 목표가/손절가를 한 번에 표시
                plot_data = chart_df[['close']].copy()
                plot_data['Target'] = asset_info['target_price']
                plot_data['StopLoss'] = asset_info['stop_loss']
                
                st.line_chart(plot_data, color=["#00ff88", "#00ff88", "#ff4b4b"])
                st.caption(f"초록 점선: 목표가 ({asset_info['target_price']:,.0f}) | 빨간 점선: 손절가 ({asset_info['stop_loss']:,.0f})")

                # [Engine 2] 프리미엄 캔들스틱 (선택 사항)
                with st.expander("🕯️ 프리미엄 캔들스택 차트 보기 (Plotly)"):
                    try:
                        o, h, l, c = chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close']
                        fig = go.Figure(data=[go.Candlestick(
                            x=chart_df.index, open=o, high=h, low=l, close=c, name="Candle"
                        )])
                        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.warning("캔들스택 엔진에 일시적 오류가 있습니다. 위 라인 차트를 참조하세요.")
            else:
                st.error("데이터 수신 오류. 잠시 후 새로고침해 주세요.")
            
            with st.expander("📊 수신 데이터 상세 확인 (Debug)"):
                st.write(f"Column Names: {list(chart_df.columns)}")
                st.write(chart_df.tail(3))

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
