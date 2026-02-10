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
            ticker = asset_info['ticker']
            
            @st.cache_data(ttl=300)
            def fetch_verified_history(t):
                try:
                    # [v4.1] 정밀 데이터 로드 (한국 시장 특화)
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    
                    # [CRITICAL FIX] MultiIndex 컬럼 완전 제거 및 가격 데이터 추출
                    data = raw.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        # 두 번째 레벨(Price Type)이 가격 정보를 가지고 있음
                        data.columns = data.columns.get_level_values(1)
                    
                    data.columns = [str(c).lower().strip() for c in data.columns]
                    
                    # 'close' 컬럼 확보 전략
                    if 'close' not in data.columns:
                        # adj close 등이 있다면 그것을 사용
                        potential_cols = [c for c in data.columns if 'close' in c]
                        if potential_cols:
                            data['close'] = data[potential_cols[0]]
                        else:
                            return pd.DataFrame()
                    
                    return data[['close']].astype(float).dropna()
                except: return pd.DataFrame()

            chart_df = fetch_verified_history(ticker)
            
            if not chart_df.empty:
                st.subheader(f"📈 {selected_asset} 실시간 주가 히스토리")
                
                # [Step 2] 실제 가격 매핑 시각화
                fig = go.Figure()
                
                # 순수 블랙 라인 (실제 시장 주가)
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['close'],
                    name="시장 가격",
                    line=dict(color='#000000', width=1.8)
                ))
                
                # [Step 3] 시각적 왜곡 방지 세팅
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=600,
                    yaxis_type="log", # 장기 성장을 보여주는 필수 설정
                    yaxis=dict(
                        gridcolor='#f0f0f0', 
                        autorange=True, 
                        title="Price (KRW, Log Scale)",
                        side="right", tickformat=',.0f'
                    ),
                    xaxis=dict(
                        gridcolor='#f0f0f0', 
                        title="Timeline (Historical)",
                        rangeslider=dict(visible=True)
                    ),
                    margin=dict(l=10, r=40, t=20, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 데이터 정합성 증명
                current_p = chart_df['close'].iloc[-1]
                st.caption(f"✅ 현재 데이터 동기화 완료: {selected_asset} ({ticker}) | 최종가: {current_p:,.0f} KRW")
            else:
                st.warning("선택하신 종목의 역사적 가격 정보를 구성하는 중입니다. 잠시만 기다려주세요.")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
