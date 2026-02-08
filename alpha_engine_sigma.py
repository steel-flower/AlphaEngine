import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import itertools
import warnings
from datetime import datetime
import os
import random
import json
import sys

# Windows 인코딩 대응
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 결과 재현성
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ==============================================================================
# 1. CORE AI COMPONENTS (기초 코드 원본 - Transformer)
# ==============================================================================

class TransformerPredictor(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=2, num_heads=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==============================================================================
# 2. ALPHA ENGINE SIGMA v7.0 (SACRED RESTORATION - 기초 코드 완전 복원)
# ==============================================================================

class AlphaEngineSigma:
    def __init__(self, ticker="226490.KS", name="KODEX 코스피"):
        self.ticker = ticker
        self.name = name
        self.df = None
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.features = []
        self.vix_df = None
        self.best_params = {'lrs_period': 45, 'entry_threshold': 0.50,
                            'tp_atr': 5.0, 'sl_atr': 1.8, 'rsi_period': 14}
        self.transaction_cost = 0.002
        self.blueprint_file = "ticker_blueprints.json"
        self.log_dir = "execution_logs"
        
        # 로그 디렉토리 생성
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.load_blueprint()

    def load_blueprint(self):
        if os.path.exists(self.blueprint_file):
            try:
                with open(self.blueprint_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    if self.ticker in data:
                        self.best_params.update(data[self.ticker].get('best_params', {}))
                        print(f" [계승] {self.ticker} 파라미터 로드 완료")
            except: pass

    def save_blueprint(self, metrics):
        blueprints = {}
        if os.path.exists(self.blueprint_file):
            try:
                with open(self.blueprint_file, "r", encoding='utf-8') as f: blueprints = json.load(f)
            except: pass
        
        # JSON 직렬화를 위한 타입 변환
        def convert_value(v):
            if isinstance(v, (np.float32, np.float64)): return float(v)
            if isinstance(v, (np.int32, np.int64)): return int(v)
            if isinstance(v, dict): return {str(k): convert_value(vv) for k, vv in v.items()}
            return v
        
        clean_m = {k: convert_value(v) for k, v in metrics.items()}
        blueprints[self.ticker] = {"name": self.name, "best_params": self.best_params, "metrics": clean_m, "updated": str(datetime.now())}
        with open(self.blueprint_file, "w", encoding='utf-8') as f: json.dump(blueprints, f, ensure_ascii=False, indent=4)

    def fetch_data(self):
        print(f"\n [데이터] {self.name} ({self.ticker}) 수집 중...")
        data = yf.download(self.ticker, period="max", auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if isinstance(data['Close'], pd.DataFrame):
            c = data['Close'].iloc[:, 0]; v = data['Volume'].iloc[:, 0]
            self.df = pd.DataFrame({'Close': c, 'Volume': v}, index=data.index)
        else:
            self.df = data[['Close', 'Volume']].copy()
        self.df.dropna(inplace=True)
        print(f"  - 기간: {self.df.index[0].date()} ~ {self.df.index[-1].date()} ({len(self.df)} 거래일)")

        print(" [데이터] VIX 지수 수집 중...")
        vix = yf.download("^VIX", period="max", auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        self.vix_df = vix[['Close']].rename(columns={'Close': 'VIX'})
        self.df = self.df.join(self.vix_df).fillna(method='ffill')

        print(" [데이터] USD/KRW 수집 중...")
        usd_krw = yf.download("KRW=X", period="max", auto_adjust=True)
        if isinstance(usd_krw.columns, pd.MultiIndex): usd_krw.columns = usd_krw.columns.get_level_values(0)
        self.df = self.df.join(usd_krw[['Close']].rename(columns={'Close': 'USD_KRW'})).fillna(method='ffill')

    def add_indicators(self, df, lrs_period=45, rsi_period=14):
        """기초 코드 원본 지표 - 17개 피처 완전 복원"""
        df = df.copy()
        close = df['Close']; volume = df['Volume']
        df['Log_Ret'] = np.log(close / close.shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(20, min_periods=1).std()
        df['Vol_60'] = df['Log_Ret'].rolling(60, min_periods=1).std()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period, min_periods=1).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        df['MA20'] = close.rolling(20, min_periods=1).mean()
        df['MA200'] = close.rolling(200, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std()
        df['BB_Width'] = (4 * std20) / (df['MA20'] + 1e-9)
        df['BB_Squeeze'] = df['BB_Width'].rolling(20, min_periods=1).min() / (df['BB_Width'] + 1e-9)

        df['Disp_20'] = close / (df['MA20'] + 1e-9)
        df['Disp_200'] = close / (df['MA200'] + 1e-9)

        def calc_adx(c_ser, p=14):
            up = c_ser.diff(); down = -c_ser.diff()
            plus_dm = up.where((up > down) & (up > 0), 0)
            minus_dm = down.where((down > up) & (down > 0), 0)
            atr = up.abs().rolling(p, min_periods=1).mean()
            plus_di = 100 * (plus_dm.rolling(p, min_periods=1).mean() / (atr + 1e-9))
            minus_di = 100 * (minus_dm.rolling(p, min_periods=1).mean() / (atr + 1e-9))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
            adx = dx.rolling(p, min_periods=1).mean()
            return adx, adx.diff(), plus_di, minus_di

        df['ADX'], df['ADX_Slope'], df['Plus_DI'], df['Minus_DI'] = calc_adx(close)
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV_Slope'] = df['OBV'].diff(5) / (df['OBV'].rolling(15, min_periods=1).std() + 1e-9)

        def linreg_slope(series, period):
            return series.rolling(period, min_periods=2).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (x[-1] + 1e-9), raw=True)

        df['Price_Slope'] = linreg_slope(close, lrs_period)
        df['ATR'] = close.diff().abs().rolling(14, min_periods=1).mean()

        df['VIX_Rel'] = df['VIX'].rolling(252, min_periods=1).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-9), raw=True)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        low_14 = close.rolling(14, min_periods=1).min()
        high_14 = close.rolling(14, min_periods=1).max()
        df['Stoch_K'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-9)
        df['Stoch_D'] = df['Stoch_K'].rolling(3, min_periods=1).mean()

        df['USD_KRW_Change'] = df['USD_KRW'].pct_change()
        df['Target'] = df['Log_Ret'].rolling(5).sum().shift(-5)
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        return df

    def prepare_sequences(self, df, seq_len=60):
        """기초 코드 원본 - 17개 피처"""
        self.features = ['Log_Ret', 'RSI', 'BB_Width', 'Price_Slope', 'Vol_20', 'ADX', 'Plus_DI', 'Minus_DI',
                         'Disp_200', 'Disp_20', 'OBV_Slope', 'BB_Squeeze', 'MACD', 'MACD_Signal',
                         'Stoch_K', 'Stoch_D', 'USD_KRW_Change']
        X_scaled = self.scaler_X.fit_transform(df[self.features])
        y_scaled = self.scaler_y.fit_transform(df[['Target']])
        X_seq = [X_scaled[i-seq_len:i] for i in range(seq_len, len(X_scaled))]
        y_seq = [y_scaled[i] for i in range(seq_len, len(X_scaled))]
        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train, y_train, epochs=10):
        model = TransformerPredictor(input_size=len(self.features))
        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=False)  # 재현성 보장
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        return model

    def evaluate_strategy(self, df_test, params=None):
        """기초 코드 원본 - Tech_Score 4중 합산 + 완전한 트레이딩 로직"""
        if self.model is None: raise ValueError("모델이 학습되지 않음")
        self.model.eval()
        p = params if params else self.best_params
        tp_atr_base = p.get('tp_atr', 3.2); sl_atr_base = p.get('sl_atr', 1.5); entry_base = p.get('entry_threshold', 0.35)

        seq_len = 60
        X_scaled = self.scaler_X.transform(df_test[self.features])
        if len(X_scaled) < seq_len:
            return df_test, {'total_return': 0, 'sharpe': 0, 'mdd': 0}
        X_tensor = torch.FloatTensor(np.array([X_scaled[i-seq_len:i] for i in range(seq_len, len(X_scaled))]))
        with torch.no_grad(): pred_scaled = self.model(X_tensor).numpy()
        pred_return = self.scaler_y.inverse_transform(pred_scaled).flatten()

        res_df = df_test.iloc[seq_len:].copy()
        res_df['Pred_Ret'] = pred_return
        
        # 기초 코드 핵심: AI_Score + Tech_Score (4중 합산)
        res_df['AI_Score'] = res_df['Pred_Ret'] * 55
        res_df['Tech_Score'] = np.where(res_df['Plus_DI'] > res_df['Minus_DI'], 0.5, -0.45) \
                             + np.where(res_df['OBV_Slope'] > 1.0, 0.5, -0.25) \
                             + np.where(res_df['MACD'] > res_df['MACD_Signal'], 0.3, -0.3) \
                             + np.where(res_df['Stoch_K'] > res_df['Stoch_D'], 0.2, -0.2)
        res_df['Total_Score'] = res_df['Tech_Score'] + res_df['AI_Score']

        res_df['Strategy_Return'] = 0.0; res_df['Year'] = res_df.index.year
        signals = np.zeros(len(res_df)); strat_returns = np.zeros(len(res_df))
        prices = res_df['Close'].values; atrs = res_df['ATR'].values; mk_rets = res_df['Close'].pct_change().fillna(0).values
        vol_median = res_df['Vol_60'].median()
        
        pos = 0; entry_price = 0; highest_price = 0; holding_days = 0; ytd_return = 0.0
        trades_per_year = {y: 0 for y in res_df['Year'].unique()}
        trade_results = {y: [] for y in res_df['Year'].unique()}
        holding_days_per_year = {y: [] for y in res_df['Year'].unique()}
        winning_trades = []  # 수익 거래
        losing_trades = []   # 손실 거래
        pos_size = 0.5
        current_year = res_df['Year'].iloc[0]

        for i in range(1, len(res_df)):
            year = res_df['Year'].iloc[i]
            if year != current_year:
                current_year = year; ytd_return = 0.0

            vol_ratio = res_df['Vol_60'].iloc[i] / (vol_median + 1e-9)
            is_bear = prices[i-1] < res_df['MA200'].iloc[i-1]

            # 기초 코드 핵심: YTD 성과에 따른 동적 조정
            if ytd_return < 0:
                defensive_mult = 0.7; entry_threshold_adj = entry_base * 1.5
                sl_atr = sl_atr_base * 0.7; tp_atr = tp_atr_base * 1.2
            elif ytd_return > 0.15:
                defensive_mult = 0.8; entry_threshold_adj = entry_base * 1.1
                sl_atr = sl_atr_base * 0.6; tp_atr = tp_atr_base * 0.8
            else:
                defensive_mult = 1.0; entry_threshold_adj = entry_base
                sl_atr = sl_atr_base; tp_atr = tp_atr_base

            # 기초 코드 핵심: VIX 리바운드 감지
            vix_rel = res_df['VIX_Rel'].iloc[i]
            vix_change = res_df['VIX'].iloc[i] - res_df['VIX'].iloc[i-1]
            is_vix_rebound = (vix_rel > 0.7) and (vix_change < 0)

            # 기초 코드 핵심: 켈리 공식 포지션 사이징
            expected_ret = res_df['Pred_Ret'].iloc[i-1]
            expected_vol = res_df['Vol_20'].iloc[i-1]
            kelly_size = expected_ret / (expected_vol ** 2 + 1e-9) if expected_vol > 0 else 0
            pos_size = min(max(defensive_mult * (1.5 if is_vix_rebound else 1.0) * kelly_size, 0), 1.0)

            final_threshold = entry_threshold_adj if not is_vix_rebound else entry_threshold_adj * 0.85
            is_deep_oversold = (res_df['Disp_20'].iloc[i-1] < 0.94) or (res_df['RSI'].iloc[i-1] < 24)

            trade_cost = 0.0
            score = res_df['Total_Score'].iloc[i-1]
            
            if pos == 0:
                # v7.7: 수동 트레이딩 모드 - AI와 기술 점수 동시 만족
                tech_score = res_df['Tech_Score'].iloc[i-1]
                if (score > final_threshold and tech_score > 0.3) or (score > 0 and tech_score > 0.5 and is_deep_oversold):
                    pos = 1; entry_price = prices[i]; highest_price = prices[i]; holding_days = 0; signals[i] = pos_size
                    trade_cost = pos_size * self.transaction_cost
                    trades_per_year[year] += 1
            else:
                highest_price = max(highest_price, prices[i]); holding_days += 1
                curr_tp = entry_price + tp_atr * atrs[i]
                curr_sl = highest_price - sl_atr * atrs[i]
                
                # v7.7: 수동 트레이딩 모드 - 60일 보유 제한
                if prices[i] < curr_sl or prices[i] > curr_tp or score < -0.4 or (holding_days > 60 and prices[i] < entry_price * 1.01):
                    pos = 0; signals[i] = -1
                    trade_cost = pos_size * self.transaction_cost
                    # 거래 수익률 계산
                    trade_return = (prices[i]/entry_price - 1) - self.transaction_cost
                    trade_results[year].append(trade_return > 0)
                    
                    # 수익/손실 분류
                    if trade_return > 0:
                        winning_trades.append(trade_return)
                    else:
                        losing_trades.append(trade_return)
                    
                    holding_days_per_year[year].append(holding_days)

            active_size = signals[i] if pos == 1 and signals[i] > 0 else (pos_size if pos == 1 else 0)
            strat_returns[i] = active_size * mk_rets[i] - trade_cost
            ytd_return = (1 + ytd_return) * (1 + strat_returns[i]) - 1

        res_df['Signal'] = signals; res_df['Strategy_Return'] = strat_returns
        cum_s = (1 + res_df['Strategy_Return']).cumprod()
        
        win_rates = {y: (sum(r)/len(r) if r else 0) for y, r in trade_results.items()}
        avg_holding = {y: (np.mean(d) if d else 0) for y, d in holding_days_per_year.items()}
        
        # CAGR 계산 (연평균 수익률)
        total_days = (res_df.index[-1] - res_df.index[0]).days
        years = total_days / 365.25
        cagr = (cum_s.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
        
        # 손익비 계산 (Win/Loss Ratio)
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 총 거래 횟수
        total_trades = sum(trades_per_year.values())
        
        return res_df, {'total_return': cum_s.iloc[-1]-1,
                        'sharpe': np.sqrt(252)*res_df['Strategy_Return'].mean()/(res_df['Strategy_Return'].std()+1e-9),
                        'mdd': ((cum_s.cummax()-cum_s)/(cum_s.cummax()+1e-9)).max(),
                        'trades_per_year': trades_per_year,
                        'win_rates': win_rates,
                        'avg_holding': avg_holding,
                        'cagr': cagr,
                        'win_loss_ratio': win_loss_ratio,
                        'total_trades': total_trades}

    def generate_report(self, res_df, met):
        print("\n" + "="*105)
        print(f" [최종 리포트] {self.name} ({self.ticker}) - Alpha Engine Sigma v7.7 (Manual Trading)")
        print("="*105)
        print(f" - 거래 비용: 0.2% (세금, 수수료 포함)")
        print(f" - 누적 수익률:           {met['total_return']:>10.2%}")
        print(f" - 연평균 수익률 (CAGR):  {met['cagr']:>10.2%}")
        print(f" - Sharpe Ratio:          {met['sharpe']:>10.2f}")
        print(f" - 최대 낙폭(MDD):        {met['mdd']:>10.2%}")
        print(f" - 총 거래 횟수:          {met['total_trades']:>10}회")
        print(f" - 손익비 (Win/Loss):   {met['win_loss_ratio']:>10.2f}")

        res_df['Cum_Return'] = (1 + res_df['Strategy_Return']).cumprod()
        print("\n [연도별 상세 성과]")
        print("-" * 105)
        print(f"   {'연도':^6}   |   {'수익률':^12}   |   {'거래횟수':^10}   |   {'적중률':^10}   |   {'평균보유일':^12}   |   {'상태':^6}")
        print("-" * 105)
        
        all_pass = True
        for year, group in res_df.groupby('Year'):
            first_v = group['Cum_Return'].iloc[0] / (1 + group['Strategy_Return'].iloc[0])
            y_ret = (group['Cum_Return'].iloc[-1] / first_v) - 1
            t_cnt = met['trades_per_year'].get(year, 0)
            wr = met['win_rates'].get(year, 0)
            avg_h = met['avg_holding'].get(year, 0)
            
            if y_ret < 0.10: all_pass = False
            status = "PASS" if y_ret >= 0.10 else ("OK" if y_ret >= 0 else "FAIL")
            print(f"     {year}     |   {y_ret:>12.2%}   |   {t_cnt:^10}   |   {wr:>10.1%}   |   {avg_h:>10.1f}일     |   {status:^6}")
        
        print("-" * 105)
        if all_pass:
            print(" [✓] 목표 달성: 모든 연도 10% 이상 수익")
        else:
            print(" [!] 경고: 10% 미달 연도 존재")
        print("="*105)
        
        # 매매 전략 제안
        self.generate_trading_recommendation(res_df)
        
        # 웹용 정밀 데이터 저장 (새로 추가)
        self.save_web_data(res_df, met)
        
        # 실행 로그 저장
        self.save_execution_log(res_df, met)
        
        self.save_blueprint(met)

    def save_web_data(self, res_df, met):
        """웹 앱(Streamlit)에서 표시할 정밀 성과 데이터를 JSON으로 저장"""
        try:
            # 0. 누적 수익률 계산 (누락 방지)
            if 'Cum_Return' not in res_df.columns:
                res_df['Cum_Return'] = (1 + res_df['Strategy_Return']).cumprod()
            
            # 1. 월간 수익률 계산
            monthly_ret = res_df.resample('M')['Strategy_Return'].apply(lambda x: (1 + x).prod() - 1)
            monthly_data = []
            for date, val in monthly_ret.items():
                monthly_data.append({
                    "month": date.strftime('%Y-%m'),
                    "return": float(val)
                })

            # 2. 연도별 수익률 (정밀)
            yearly_data = []
            for year, group in res_df.groupby('Year'):
                first_v = group['Cum_Return'].iloc[0] / (1 + group['Strategy_Return'].iloc[0])
                y_ret = (group['Cum_Return'].iloc[-1] / first_v) - 1
                yearly_data.append({
                    "year": int(year),
                    "return": float(y_ret),
                    "trades": int(met['trades_per_year'].get(year, 0)),
                    "win_rate": float(met['win_rates'].get(year, 0)),
                    "avg_hold": float(met['avg_holding'].get(year, 0))
                })

            # 3. 최근 거래 내역 (10건)
            trades = res_df[res_df['Signal'] != 0].tail(10)
            trade_history = []
            for idx, row in trades.iterrows():
                trade_history.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "price": float(row['Close']),
                    "signal": "BUY" if row['Signal'] > 0 else "SELL",
                    "ai_score": float(row['AI_Score']),
                    "tech_score": float(row['Tech_Score'])
                })

            # 4. 차트용 일별 누적 수익률 (상장 이후 전체 기간 포함)
            all_perf_df = res_df.copy()
            # 전체 시장 누적 수익률 (최초 시점 대비)
            all_perf_df['Market_Cum'] = (1 + all_perf_df['Close'].pct_change().fillna(0)).cumprod()
            daily_performance = []
            for idx, row in all_perf_df.iterrows():
                daily_performance.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "strategy": float(row['Cum_Return']),
                    "market": float(row['Market_Cum']),
                    "close": float(row['Close']),
                    "signal": float(row['Signal'])
                })

            # 최종 데이터 패키징
            web_package = {
                "ticker": self.ticker,
                "name": self.name,
                "updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "summary": {
                    "total_return": float(met['total_return']),
                    "cagr": float(met['cagr']),
                    "sharpe": float(met['sharpe']),
                    "mdd": float(met['mdd']),
                    "total_trades": int(met['total_trades']),
                    "win_loss_ratio": float(met['win_loss_ratio']),
                    "current_price": float(res_df['Close'].iloc[-1])
                },
                "latest_signal": {
                    "ai_score": float(res_df['AI_Score'].iloc[-1]),
                    "tech_score": float(res_df['Tech_Score'].iloc[-1]),
                    "entry_threshold": float(self.best_params['entry_threshold'])
                },
                "yearly_performance": yearly_data,
                "monthly_performance": monthly_data,
                "trade_history": trade_history,
                "daily_performance": daily_performance
            }

            # 파일 저장
            file_path = f"web_data_{self.ticker.replace('.KS', '')}.json"
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(web_package, f, ensure_ascii=False, indent=4)
            print(f" [시스템] 웹 연동 데이터 업데이트 완료: {file_path}")

        except Exception as e:
            print(f" [오류] 웹 데이터 저장 실패: {e}")

    def generate_trading_recommendation(self, res_df):
        """차기 거래일 매매 전략 제안"""
        print("\n [차기 거래일 매매 전략 제안]")
        print("-" * 105)
        
        # 최신 데이터
        latest = res_df.iloc[-1]
        current_price = latest['Close']
        atr = latest['ATR']
        score = latest['AI_Score']
        tech_score = latest['Tech_Score']
        
        # 진입 조건 확인
        entry_signal = score > self.best_params['entry_threshold'] and tech_score > 0
        
        if entry_signal:
            entry_price = current_price
            target_price = current_price * (1 + self.best_params['tp_atr'] * atr / current_price)
            stop_loss = current_price * (1 - self.best_params['sl_atr'] * atr / current_price)
            
            print(f" 전략: 매수 진입 권장")
            print(f" - 진입가:  {entry_price:>10,.0f}원")
            print(f" - 목표가:  {target_price:>10,.0f}원 (+{(target_price/entry_price-1)*100:.1f}%)")
            print(f" - 손절가:  {stop_loss:>10,.0f}원 (-{(1-stop_loss/entry_price)*100:.1f}%)")
            print(f" - AI 점수: {score:>10.2f} (기준: {self.best_params['entry_threshold']:.2f})")
            print(f" - 기술점수: {tech_score:>10.2f}")
        else:
            print(f" 전략: 관망")
            print(f" - 현재가:  {current_price:>10,.0f}원")
            print(f" - AI 점수: {score:>10.2f} (기준: {self.best_params['entry_threshold']:.2f}) - 진입 조건 미달")
            print(f" - 기술점수: {tech_score:>10.2f}")
        
        print("-" * 105)
    
    def save_execution_log(self, res_df, met):
        """실행 로그 저장 (종목당 최신 2개만 유지)"""
        from datetime import datetime
        import glob
        
        # 로그 파일명 (날짜_티커 형식)
        timestamp = datetime.now().strftime("%Y%m%d")
        ticker_code = self.ticker.replace('.KS', '')
        log_filename = f"{self.log_dir}/{timestamp}_{ticker_code}.txt"
        
        # 로그 내용 작성
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write("="*105 + "\n")
            f.write(f" Alpha Engine Sigma v7.7 (Manual Trading) - 실행 로그\n")
            f.write(f" 종목: {self.name} ({self.ticker})\n")
            f.write(f" 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*105 + "\n\n")
            
            f.write("[최적화 파라미터]\n")
            for key, val in self.best_params.items():
                f.write(f" - {key}: {val}\n")
            
            f.write(f"\n[성과 지표]\n")
            f.write(f" - 누적 수익률: {met['total_return']:.2%}\n")
            f.write(f" - 연평균 수익률 (CAGR): {met['cagr']:.2%}\n")
            f.write(f" - Sharpe Ratio: {met['sharpe']:.2f}\n")
            f.write(f" - 최대 낙폭(MDD): {met['mdd']:.2%}\n")
            f.write(f" - 총 거래 횟수: {met['total_trades']}회\n")
            f.write(f" - 손익비 (Win/Loss): {met['win_loss_ratio']:.2f}\n")
            
            f.write(f"\n[연도별 성과]\n")
            for year in sorted(met['trades_per_year'].keys()):
                y_ret = res_df[res_df['Year']==year]['Strategy_Return']
                y_ret_total = (1 + y_ret).prod() - 1
                f.write(f" {year}: {y_ret_total:>8.2%} (거래: {met['trades_per_year'][year]}회, ")
                f.write(f"적중률: {met['win_rates'][year]:.1%})\n")
        
        # 오래된 로그 삭제 (종목당 최신 2개만 유지)
        ticker_code = self.ticker.replace('.KS', '')
        log_files = sorted(glob.glob(f"{self.log_dir}/*_{ticker_code}.txt"), reverse=True)
        
        for old_log in log_files[2:]:  # 3번째 파일부터 삭제
            try:
                os.remove(old_log)
                print(f" [로그] 오래된 로그 삭제: {os.path.basename(old_log)}")
            except:
                pass
        
        print(f" [로그] 실행 로그 저장: {os.path.basename(log_filename)}")


    def run_optimization(self, df_orig):
        """v7.7 수동 트레이딩 모드 - 적중률 60%+ 목표"""
        print(" [최적화] Alpha Engine Sigma v7.7: 수동 트레이딩 모드 (적중률 60%+ 목표)")
        
        # 재현성 보장: 최적화 시작 전 시드 재설정
        set_seed(42)
        
        # v7.7: 수동 트레이딩 최적화 그리드
        param_grid = {
            'lrs_period': [45, 60],
            'entry_threshold': [0.45, 0.50, 0.55],  # 높은 문턱
            'tp_atr': [4.0, 5.0, 6.0],  # 큰 수익 목표
            'sl_atr': [1.5, 1.8, 2.0],  # 여유 있는 손절
            'rsi_period': [14]
        }
        keys = param_grid.keys()
        combos = list(itertools.product(*param_grid.values()))
        
        # 재현성 보장: 랜덤 샘플링 대신 정렬된 조합 사용
        combos.sort()  # 항상 같은 순서
        sampled_combos = combos[:20]  # 처음 20개 (매번 동일)
        
        best_sigma_score = -np.inf
        best_model_st = None
        split_idx = int(len(df_orig) * 0.8)

        for c in tqdm(sampled_combos, desc="Win-Rate Priority Search (Deterministic)"):
            ps = dict(zip(keys, c))
            try:
                # 각 조합마다 시드 재설정 (재현성)
                set_seed(42)
                
                df = self.add_indicators(df_orig, lrs_period=ps['lrs_period'], rsi_period=ps['rsi_period'])
                Xt, yt = self.prepare_sequences(df.iloc[:split_idx])
                mod = self.train(Xt[:int(len(Xt)*0.9)], yt[:int(len(Xt)*0.9)], epochs=20)
                self.model = mod
                res_df, met = self.evaluate_strategy(df, params=ps)

                yr_rets = res_df.groupby(res_df.index.year)['Strategy_Return'].apply(lambda x: (1+x).cumprod().iloc[-1]-1)
                neg_years = (yr_rets < 0).sum()
                under_10_years = (yr_rets < 0.10).sum()
                # v7.7: 수동 트레이딩 모드 - 적중률 최우선
                avg_win_rate = np.mean([met['win_rates'].get(y, 0) for y in met['win_rates'].keys()])

                # 적중률 60% 목표 - 최고 가중치
                win_rate_bonus = (avg_win_rate - 0.6) * 20000  # 60% 이상이면 큰 보너스
                stability_penalty = (neg_years * 8000) + (under_10_years * 5000) + (met['mdd'] * 2000)
                sigma_score = (met['total_return'] * 100) + (met['sharpe'] * 400) + win_rate_bonus - stability_penalty

                if sigma_score > best_sigma_score:
                    best_sigma_score = sigma_score
                    self.best_params = ps
                    best_model_st = mod.state_dict()
                    print(f"  [발견] WinRate: {avg_win_rate:.1%}, NegYrs: {neg_years}, Under10%: {under_10_years}, MDD: {met['mdd']:.2%}, Tot: {met['total_return']:.1%}")
            except Exception as e:
                print(f"Error in combo {ps}: {e}")
                continue

        if best_model_st:
            self.model = TransformerPredictor(input_size=len(self.features))
            self.model.load_state_dict(best_model_st)
        else:
            print(" [경고] 유효한 파라미터를 찾지 못함. 기본값 사용")
            df = self.add_indicators(df_orig, lrs_period=self.best_params['lrs_period'],
                                     rsi_period=self.best_params['rsi_period'])
            Xt, yt = self.prepare_sequences(df.iloc[:split_idx])
            self.model = self.train(Xt[:int(len(Xt)*0.9)], yt[:int(len(Xt)*0.9)], epochs=10)

    def run_engine(self):
        """기초 코드 원본 실행 흐름 (무거운 학습 포함)"""
        self.run_optimization(self.df)
        
        f_df = self.add_indicators(self.df,
                                   lrs_period=self.best_params['lrs_period'],
                                   rsi_period=self.best_params['rsi_period'])
        res_df, met = self.evaluate_strategy(f_df)
        self.generate_report(res_df, met)

    def run_quick_update(self):
        """[모니터링 모드] 저장된 최적 파라미터로 즉시 분석 및 웹 데이터 갱신"""
        print(f" [모니터] {self.name} - 로컬 최적값 호출 및 고속 분석 중...")
        
        # 1. 지표 계산 (저장된 파라미터 사용)
        f_df = self.add_indicators(self.df, 
                                   lrs_period=self.best_params['lrs_period'], 
                                   rsi_period=self.best_params['rsi_period'])
        
        # 2. 고속 학습 (epochs 5)
        Xt, yt = self.prepare_sequences(f_df)
        if len(Xt) > 0:
            self.model = self.train(Xt, yt, epochs=5)
            
            # 3. 결과 도출
            res_df, met = self.evaluate_strategy(f_df)
            
            # 4. 웹 데이터 저장 및 리포트 공시
            self.save_web_data(res_df, met)
            # 수동 트레이딩 제안만 출력 (화면 가독성)
            self.generate_trading_recommendation(res_df) 
            print(f" ✅ {self.name} 모니터링 업데이트 완료")
        else:
            print(f" ❌ {self.name} 데이터 부족으로 업데이트 실패")

# ==============================================================================
# 메인 메뉴
# ==============================================================================

def main():
    if not os.path.exists("assets.json"):
        default = [{"ticker": "226490.KS", "name": "KODEX 코스피"},
                   {"ticker": "114800.KS", "name": "KODEX 인버스"},
                   {"ticker": "122630.KS", "name": "KODEX 레버리지"}]
        with open("assets.json", "w", encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=4)
    
    with open("assets.json", "r", encoding='utf-8') as f:
        assets = json.load(f)

    while True:
        print("\n" + "="*40)
        print(" ALPHA ENGINE SIGMA v7.7")
        print(" (MANUAL TRADING MODE)")
        print("="*40)
        for i, a in enumerate(assets):
            print(f" {i+1}. {a['name']} ({a['ticker']})")
        
        choice = input("\n 종목 선택 (Q: 종료): ").upper()
        if choice == 'Q':
            break
        
        try:
            target = assets[int(choice)-1]
            engine = AlphaEngineSigma(target['ticker'], target['name'])
            engine.fetch_data()
            engine.run_engine()
            input("\n 엔터를 누르면 메뉴로 돌아갑니다...")
        except Exception as e:
            print(f" [에러] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
