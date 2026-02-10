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
import glob
import json

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. DETERMINISTIC CONTROL (완벽 통제)
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ==============================================================================
# 1. CORE AI COMPONENTS (v1 Original Core Logic)
# ==============================================================================

class TransformerPredictor(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=2, num_heads=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                   dim_feedforward=256, dropout=0.2)
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
# 2. ALPHA ENGINE SIGMA v1 (Infinite Discovery & Persistence)
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
        # Original Baseline Params
        self.best_params = {'lrs_period': 45, 'entry_threshold': 0.35,
                            'tp_atr': 3.2, 'sl_atr': 1.5, 'rsi_period': 14}
        self.transaction_cost = 0.003
        self.log_dir = "Log_box"
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

    def fetch_data(self):
        print(f" [Data] Fetching {self.name} ({self.ticker})...")
        data = yf.download(self.ticker, period="max", auto_adjust=True)
        if hasattr(data, 'columns') and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        self.df = data[['Close', 'Volume']].copy()
        self.df.dropna(inplace=True)

        vix = yf.download("^VIX", period="max", auto_adjust=True)
        if hasattr(vix, 'columns') and isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        self.vix_df = vix[['Close']].rename(columns={'Close': 'VIX'})
        self.df = self.df.join(self.vix_df).fillna(method='ffill')

        usd_krw = yf.download("KRW=X", period="max", auto_adjust=True)
        if hasattr(usd_krw, 'columns') and isinstance(usd_krw.columns, pd.MultiIndex):
            usd_krw.columns = usd_krw.columns.get_level_values(0)
        self.df = self.df.join(usd_krw[['Close']].rename(columns={'Close': 'USD_KRW'})).fillna(method='ffill')

    def add_indicators(self, df, lrs_period=45, rsi_period=14):
        df = df.copy()
        close = df['Close']; volume = df['Volume']
        df['Log_Ret'] = np.log(close / close.shift(1))
        df['Vol_20'] = df['Log_Ret'].rolling(20).std(); df['Vol_60'] = df['Log_Ret'].rolling(60).std()
        delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean(); loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['MA20'] = close.rolling(20).mean(); df['MA200'] = close.rolling(200).mean()
        std20 = close.rolling(20).std(); df['BB_Width'] = (4 * std20) / (df['MA20'] + 1e-9)
        df['BB_Squeeze'] = df[' BB_Width'].rolling(20).min() / (df['BB_Width'] + 1e-9) if ' BB_Width' in df else df['BB_Width'].rolling(20).min() / (df['BB_Width'] + 1e-9)
        df['Disp_20'] = close / (df['MA20'] + 1e-9); df['Disp_200'] = close / (df['MA200'] + 1e-9)

        def calc_adx(c_ser, p=14):
            up = c_ser.diff(); down = -c_ser.diff()
            plus_dm = up.where((up > down) & (up > 0), 0); minus_dm = down.where((down > up) & (down > 0), 0)
            atr = up.abs().rolling(p).mean()
            plus_di = 100 * (plus_dm.rolling(p).mean() / (atr + 1e-9)); minus_di = 100 * (minus_dm.rolling(p).mean() / (atr + 1e-9))
            adx = (100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))).rolling(p).mean()
            return adx, adx.diff(), plus_di, minus_di

        df['ADX'], df['ADX_Slope'], df['Plus_DI'], df['Minus_DI'] = calc_adx(close)
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['OBV_Slope'] = df['OBV'].diff(5) / (df['OBV'].rolling(15).std() + 1e-9)
        df['Price_Slope'] = close.rolling(lrs_period).apply(lambda x: np.polyfit(np.arange(lrs_period), x, 1)[0] / (x[-1] + 1e-9), raw=True)
        df['ATR'] = close.diff().abs().rolling(14).mean()
        df['VIX_Rel'] = df['VIX'].rolling(252).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-9), raw=True)
        ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26; df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        low_14 = close.rolling(14).min(); high_14 = close.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-9); df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        df['USD_KRW_Change'] = df['USD_KRW'].pct_change()
        df['Target'] = df['Log_Ret'].rolling(5).sum().shift(-5)
        return df.dropna()

    def prepare_sequences(self, df, seq_len=60):
        self.features = ['Log_Ret', 'RSI', 'BB_Width', 'Price_Slope', 'Vol_20', 'ADX', 'Plus_DI', 'Minus_DI',
                         'Disp_200', 'Disp_20', 'OBV_Slope', 'BB_Squeeze', 'MACD', 'MACD_Signal',
                         'Stoch_K', 'Stoch_D', 'USD_KRW_Change']
        X_scaled = self.scaler_X.fit_transform(df[self.features])
        y_scaled = self.scaler_y.fit_transform(df[['Target']])
        X_seq = np.array([X_scaled[i-seq_len:i] for i in range(seq_len, len(X_scaled))])
        y_seq = np.array([y_scaled[i] for i in range(seq_len, len(X_scaled))])
        return X_seq, y_seq

    def train(self, X_train, y_train, current_seed=42):
        set_seed(current_seed)
        model = TransformerPredictor(input_size=len(self.features))
        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for _ in range(10):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad(); loss = criterion(model(bx), by); loss.backward(); optimizer.step()
        return model

    def evaluate_strategy(self, df_test, params=None):
        if self.model is None: raise ValueError("Model not trained")
        self.model.eval()
        p = params if params else self.best_params
        tp_atr_base = p.get('tp_atr', 3.2); sl_atr_base = p.get('sl_atr', 1.5); entry_base = p.get('entry_threshold', 0.35)
        seq_len = 60
        X_scaled = self.scaler_X.transform(df_test[self.features])
        X_tensor = torch.FloatTensor([X_scaled[i-seq_len:i] for i in range(seq_len, len(X_scaled))])
        with torch.no_grad(): pred_scaled = self.model(X_tensor).numpy()
        pred_return = self.scaler_y.inverse_transform(pred_scaled).flatten()

        res_df = df_test.iloc[seq_len:].copy()
        res_df['Pred_Ret'] = pred_return
        res_df['AI_Score'] = res_df['Pred_Ret'] * 55
        res_df['Tech_Score'] = np.where(res_df['Plus_DI'] > res_df['Minus_DI'], 0.5, -0.45) \
                             + np.where(res_df['OBV_Slope'] > 1.0, 0.5, -0.25) \
                             + np.where(res_df['MACD'] > res_df['MACD_Signal'], 0.3, -0.3) \
                             + np.where(res_df['Stoch_K'] > res_df['Stoch_D'], 0.2, -0.2)
        res_df['Total_Score'] = res_df['Tech_Score'] + res_df['AI_Score']
        res_df['Strategy_Return'] = 0.0; res_df['Year'] = res_df.index.year
        pos = 0; entry_price = 0; highest_price = 0; ytd_return = 0.0; current_year = res_df['Year'].iloc[0]; holding_days = 0
        signals = np.zeros(len(res_df)); strat_returns = np.zeros(len(res_df))
        prices = res_df['Close'].values; atrs = res_df['ATR'].values; mk_rets = res_df['Close'].pct_change().fillna(0).values

        for i in range(1, len(res_df)):
            if res_df['Year'].iloc[i] != current_year: current_year = res_df['Year'].iloc[i]; ytd_return = 0.0
            if ytd_return < 0:
                defensive_mult = 0.7; entry_threshold_adj = entry_base * 1.5; sl_atr = sl_atr_base * 0.7; tp_atr = tp_atr_base * 1.2
            elif ytd_return > 0.15:
                defensive_mult = 0.8; entry_threshold_adj = entry_base * 1.1; sl_atr = sl_atr_base * 0.6; tp_atr = tp_atr_base * 0.8
            else:
                defensive_mult = 1.0; entry_threshold_adj = entry_base; sl_atr = sl_atr_base; tp_atr = tp_atr_base

            vix_rel = res_df['VIX_Rel'].iloc[i]; vix_change = res_df['VIX'].iloc[i] - res_df['VIX'].iloc[i-1]
            is_vix_rebound = (vix_rel > 0.7) and (vix_change < 0)
            final_threshold = entry_threshold_adj if not is_vix_rebound else entry_threshold_adj * 0.85
            is_deep_oversold = (res_df['Disp_20'].iloc[i-1] < 0.94) or (res_df['RSI'].iloc[i-1] < 24)
            pos_size = min(max(defensive_mult * (1.5 if is_vix_rebound else 1.0) * (res_df['Pred_Ret'].iloc[i-1] / (res_df['Vol_20'].iloc[i-1] ** 2 + 1e-9)), 0), 1.0)
            trade_cost = 0.0
            if pos == 0:
                score = res_df['Total_Score'].iloc[i-1]
                if score > final_threshold or (score > 0 and is_deep_oversold):
                    pos = 1; entry_price = prices[i]; highest_price = prices[i]; holding_days = 0; signals[i] = pos_size
            else:
                highest_price = max(highest_price, prices[i]); holding_days += 1
                curr_tp = entry_price + tp_atr * atrs[i]; curr_sl = highest_price - sl_atr * atrs[i]
                if prices[i] < curr_sl or prices[i] > curr_tp or res_df['Total_Score'].iloc[i-1] < -0.4 or (holding_days > 30 and prices[i] < entry_price * 1.01):
                    pos = 0; signals[i] = -1; trade_cost = pos_size * 0.003
            active_size = signals[i] if pos == 1 and signals[i] > 0 else (pos_size if pos == 1 else 0)
            strat_returns[i] = active_size * mk_rets[i] - trade_cost
            ytd_return = (1 + ytd_return) * (1 + strat_returns[i]) - 1

        res_df['Strategy_Return'] = strat_returns; cum_s = (1 + res_df['Strategy_Return']).cumprod()
        met = {'total_return': cum_s.iloc[-1]-1, 'sharpe': (strat_returns.mean()/(strat_returns.std()+1e-9))*np.sqrt(252),
               'mdd': ((cum_s.cummax()-cum_s)/(cum_s.cummax()+1e-9)).max()}
        return res_df, met

    def generate_report(self, res_df, met):
        print("\n" + "="*60 + f"\n ALPHA ENGINE SIGMA v2.0 RE-EQUILIBRATED REPORT ({self.name})\n" + "="*60)
        print(f" - Cumulative Return: {met['total_return']:>8.2%}\n - Sharpe: {met['sharpe']:>8.2f}\n - Max MDD: {met['mdd']:>8.2%}")
        yr_s = res_df.groupby(res_df.index.year)['Strategy_Return'].apply(lambda x: (1+x).cumprod().iloc[-1]-1)
        print("\n [Annual Performance Verification]")
        all_positive = True
        for y, r in yr_s.items():
            mark = "[OK]" if r >= 0 else "[!!]"
            if r < 0: all_positive = False
            print(f"  * Year {y}: {r:>7.2%} | {mark}")
        print("="*60)
        if all_positive: print(" [*] STABILITY ACHIEVED: NO LOSS YEARS.")
        else: print(" [!] Warning: Negative years detected.")

    def run_persistence_evolution(self, num_rounds=1):
        # 1. Persistence Load
        pattern = os.path.join(self.log_dir, f"{self.name}_{self.ticker}_*.txt")
        files = sorted(glob.glob(pattern), reverse=True)
        prev_params = None; baseline_score = -np.inf
        
        split_idx = int(len(self.df) * 0.8); df_train = self.add_indicators(self.df.iloc[:split_idx]); df_test = self.add_indicators(self.df)
        Xt, yt = self.prepare_sequences(df_train)

        if files:
            last_file = files[0]
            try:
                with open(last_file, 'r', encoding='utf-8') as f:
                    meta = json.loads(f.read().split("### METADATA ###")[1].strip())
                    prev_params = meta['params']
                    # Use a stable seed for baseline evaluation
                    set_seed(42)
                    temp_model = self.train(Xt[:int(len(Xt)*0.9)], yt[:int(len(Xt)*0.9)], 42)
                    self.model = temp_model
                    _, b_met = self.evaluate_strategy(df_test, prev_params)
                    b_yr = (1 + _['Strategy_Return']).groupby(_.index.year).prod() - 1
                    baseline_score = (b_met['total_return']*100 + b_met['sharpe']*300 - (b_yr < 0).sum()*5000)
                    print(f" [Persistence] '{os.path.basename(last_file)}' 복원 및 현재 데이터 재평가 완료.")
            except: pass

        # 2. Multi-Round Master Discovery
        best_overall_score = -np.inf
        best_overall_params = None
        best_overall_met = None
        best_overall_df = None

        print(f" [Evolution] Analyzing {self.name} ({self.ticker}) for {num_rounds} round(s)...")
        for r in range(num_rounds):
            current_seed = 42 + r
            if num_rounds > 1: print(f"  * Round {r+1}/{num_rounds} (Seed: {current_seed})")
            
            # Model Training with unique seed
            self.model = self.train(Xt[:int(len(Xt)*0.9)], yt[:int(len(Xt)*0.9)], current_seed)

            # Grid Search
            param_grid = {'lrs_period': [45], 'entry_threshold': [0.3, 0.35, 0.4], 'tp_atr': [3.0, 3.2, 3.5], 'sl_atr': [1.2, 1.5]}
            keys = param_grid.keys(); combos = list(itertools.product(*param_grid.values()))
            
            for c in tqdm(combos, desc=" [Scan] Grid Search", leave=False):
                ps = dict(zip(keys, c)); ps['rsi_period'] = 14
                s_df, s_met = self.evaluate_strategy(df_test, ps)
                s_yr = (1 + s_df['Strategy_Return']).groupby(s_df.index.year).prod() - 1
                score = (s_met['total_return']*100 + s_met['sharpe']*300 - (s_yr < 0).sum()*5000)
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_params = ps
                    best_overall_met = s_met
                    best_overall_df = s_df

        # 3. Inheritance vs New Best
        if prev_params and baseline_score >= best_overall_score:
            print(f"  [Result] 과거 최정예 지능 유지 (기존 파라미터 점수: {baseline_score:.1f} >= 현재 최고: {best_overall_score:.1f})")
            self.best_params = prev_params
            final_score = baseline_score
        else:
            msg = f"새로운 알파 발견" if num_rounds == 1 else f"10회 검증 중 최강 지능 추출 완료"
            print(f"  [Result] {msg} (최고 점수: {best_overall_score:.1f})")
            self.best_params = best_overall_params
            final_score = best_overall_score

        # 4. Final Finalize & Save
        final_df, final_met = self.evaluate_strategy(df_test, self.best_params)
        self.generate_report(final_df, final_met)
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.name}_{self.ticker}_{dt}.txt"; fp = os.path.join(self.log_dir, fn)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(f"=== ALPHA ENGINE REPORT v2.0: {self.name} ({self.ticker}) ===\n Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if num_rounds > 1: f.write(f" [Mode] INFINITE DISCOVERY (10-Round Best)\n")
            f.write(f"\n[1. Optimized Strategy Parameters]\n{json.dumps(self.best_params, indent=2)}\n")
            f.write(f"\n[2. Performance Metrics]\n - Return: {final_met['total_return']:.2%}\n - Sharpe: {final_met['sharpe']:.2f}\n - MDD: {final_met['mdd']:.2%}\n\n[3. Annual]\n")
            yr_s = final_df.groupby(final_df.index.year)['Strategy_Return'].apply(lambda x: (1+x).cumprod().iloc[-1]-1)
            for y, r in yr_s.items(): f.write(f" - {y}: {r:+.2%}\n")
            f.write(f"\n### METADATA ###\n{json.dumps({'params': self.best_params, 'score': final_score})}\n")
        l_files = sorted(glob.glob(os.path.join(self.log_dir, f"{self.name}_{self.ticker}_*.txt")), reverse=True)
        for old in l_files[5:]: os.remove(old)

def main():
    assets = [("KODEX 코스피", "226490.KS"), ("KODEX 인버스", "114800.KS"), ("ACE KRX 금현물", "411060.KS"), ("KODEX 은선물(H)", "144600.KS"), ("RISE 글로벌 자산배분 액티브", "461490.KS"), ("RISE 글로벌농업경제", "437370.KS"), ("KODEX WTI 원유선물(H)", "261220.KS")]
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "═"*60 + "\n    ALPHA ENGINE SIGMA v2.0\n" + "═"*60)
        for i, (n, t) in enumerate(assets, 1): print(f" {i:2d}. {n:25s} ({t})")
        print(f"  8. [RUN ALL] 모든 종목 일괄 분석 및 순차 진화\n  9. [RUN ALL x10] 모든 종목 10회 검증 및 최강 지능 추출\n  Q. [EXIT] 종료\n" + "═"*60)
        choice = input("\n [Choice]: ").upper()
        if choice == 'Q': break
        rounds = 10 if choice == '9' else 1
        to_run = assets if choice in ['8', '9'] else ([assets[int(choice)-1]] if choice.isdigit() and 1 <= int(choice) <= 7 else [])
        for n, t in to_run:
            try: e = AlphaEngineSigma(t, n); e.fetch_data(); e.run_persistence_evolution(num_rounds=rounds)
            except Exception as ex: print(f" Error: {ex}")
        input("\n [!] 완료. 초기 화면 복귀하려면 엔터...")

if __name__ == "__main__":
    main()
