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
            # [Step 1] 종목 및 전구간 데이터 호출
            selected_asset = st.selectbox("📊 분석 종목 선택", df['name'].tolist())
            asset_info = df[df['name'] == selected_asset].iloc[0]
            ticker = asset_info['ticker']
            
            @st.cache_data(ttl=60)
            def fetch_full_history(t):
                try:
                    raw = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
                    if raw.empty: return pd.DataFrame()
                    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                    raw.columns = [str(c).lower().strip() for c in raw.columns]
                    return raw[['close']].astype(float).sort_index()
                except: return pd.DataFrame()

            price_df = fetch_full_history(ticker)
            
            if not price_df.empty:
                st.subheader(f"🏛️ {selected_asset} 전구간 궤적 및 Alpha 전략")
                
                # 시각화 엔진 (데이터 정직성 + 전략 가독성 최적화)
                fig = go.Figure()
                
                # 1. 실제 주가 (Solid Black) - 전구간 관통
                fig.add_trace(go.Scatter(
                    x=price_df.index, y=price_df['close'],
                    name="실제 주가",
                    line=dict(color='black', width=1.5),
                    hovertemplate="날짜: %{x}<br>주가: %{y:,.0f} KRW<extra></extra>"
                ))
                
                # [AlphaEngine 전략 지점] 
                # 역사적 스케일을 보호하기 위해 '최근 구간'에만 가독성 좋게 표시
                target = float(asset_info['target_price'])
                entry = float(asset_info['entry_price'])
                
                # 최근 10% 기간 계산 (전략선이 역사를 가리지 않게 함)
                total_len = len(price_df)
                start_idx = price_df.index[max(0, total_len - int(total_len * 0.1))] # 최근 10% 지점
                end_idx = price_df.index[-1]
                
                # 2. Alpha 매도 목표 (Dashed Black - 최근 구간 매핑)
                fig.add_trace(go.Scatter(
                    x=[start_idx, end_idx], y=[target, target],
                    name="Alpha 매도목표 (Dash)",
                    line=dict(color='black', width=2, dash='dash'),
                    hovertemplate=f"Alpha 매도 목표: {target:,.0f} KRW<extra></extra>"
                ))
                
                # 3. Alpha 매수 진입 (Dotted Black - 최근 구간 매핑)
                fig.add_trace(go.Scatter(
                    x=[start_idx, end_idx], y=[entry, entry],
                    name="Alpha 매수진입 (Dot)",
                    line=dict(color='black', width=2, dash='dot'),
                    hovertemplate=f"Alpha 매수 진입: {entry:,.0f} KRW<extra></extra>"
                ))
                
                fig.update_layout(
                    paper_bgcolor='white', plot_bgcolor='white',
                    height=600,
                    yaxis=dict(
                        gridcolor='#f5f5f5', autorange=True,
                        title="Price (KRW)", side="right", tickformat=',.0f'
                    ),
                    xaxis=dict(
                        gridcolor='#f5f5f5', title="Timeline",
                        autorange=True, rangeslider=dict(visible=True)
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    margin=dict(l=10, r=40, t=50, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 하단 수치 가이드 (직관적 보조)
                st.markdown(f"🏛️ **{selected_asset} 전략 정보**: 현재가 대비 매도 목표까지 **{((target/price_df['close'].iloc[-1])-1)*100:+.1f}%** 여유가 있습니다.")
            else:
                st.error("데이터 로딩 실패")

        # 하단 상세 정보
        with st.expander("🏛️ v.3.4 마스터 전략 가이드 상세 보기"):
            st.write(df)

    # 하단 정보
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Alpha Engine Sigma v3.4 Master Precision Dashboard | Created by Antigravity</p>", unsafe_allow_html=True)
    
    # 5분마다 자동 새로고침 (Streamlit 기본 기능 활용)
    if st.button("🔄 수동 새로고침"):
        st.rerun()
