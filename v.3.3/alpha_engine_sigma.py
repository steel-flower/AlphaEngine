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
# 0. DETERMINISTIC CONTROL
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

# ==============================================================================
# 1. CORE AI COMPONENTS (v1 Original Core)
# ==============================================================================
class TransformerPredictor(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=2, num_heads=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=256, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.proj(x); out = self.transformer(x)
        return self.fc(out[:, -1, :])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==============================================================================
# 2. ALPHA ENGINE SIGMA v3.2 (Master Precision Evolution)
# ==============================================================================
class AlphaEngineSigma:
    def __init__(self, ticker="226490.KS", name="KODEX 코스피"):
        self.ticker = ticker; self.name = name; self.df = None; self.model = None
        self.scaler_X = MinMaxScaler(); self.scaler_y = MinMaxScaler()
        self.features = []; self.vix_df = None; self.is_downtrend_mode = False
        self.maintenance_reason = ""; self.legacy_data = None
        self.best_params = {'lrs_period': 45, 'entry_threshold': 0.35, 'tp_atr': 3.2, 'sl_atr': 1.5, 'rsi_period': 14}
        self.log_dir = "Log_box"
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

    def fetch_data(self):
        print(f" [Data] Fetching {self.name} ({self.ticker})...")
        data = yf.download(self.ticker, period="max", auto_adjust=True)
        if hasattr(data, 'columns') and isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        self.df = data[['Close', 'Volume']].copy().dropna()
        n = len(self.df); long_w = min(200, max(20, n // 5)); mid_w = min(60, max(10, n // 15))
        ma_l = self.df['Close'].rolling(long_w, min_periods=1).mean(); ma_m = self.df['Close'].rolling(mid_w, min_periods=1).mean()
        slope_l = ma_l.diff(min(20, long_w)).iloc[-1] / (ma_l.iloc[-1] + 1e-9)
        is_bear = (self.df['Close'].iloc[-1] < ma_l.iloc[-1]) and (ma_m.iloc[-1] < ma_l.iloc[-1]) and (slope_l < -0.005)
        is_inv = any(kw in self.name for kw in ["인버스", "Inverse", "Short"])
        self.is_downtrend_mode = is_inv or is_bear
        if self.is_downtrend_mode: print(f" [Strategy] '{self.name}' [하락 특제] 적응형 분석 모드 가동.")
        
        vix = yf.download("^VIX", period="max", auto_adjust=True)
        if hasattr(vix, 'columns') and isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        self.vix_df = vix[['Close']].rename(columns={'Close': 'VIX'})
        self.df = self.df.join(self.vix_df).fillna(method='ffill')
        usd = yf.download("KRW=X", period="max", auto_adjust=True)
        if hasattr(usd, 'columns') and isinstance(usd.columns, pd.MultiIndex): usd.columns = usd.columns.get_level_values(0)
        self.df = self.df.join(usd[['Close']].rename(columns={'Close': 'USD_KRW'})).fillna(method='ffill')
        if self.is_downtrend_mode and self.ticker != "226490.KS":
            kospi = yf.download("226490.KS", period="max", auto_adjust=True)
            if hasattr(kospi, 'columns') and isinstance(kospi.columns, pd.MultiIndex): kospi.columns = kospi.columns.get_level_values(0)
            self.df['Mkt_Log_Ret'] = np.log(kospi['Close'] / kospi['Close'].shift(1)).reindex(self.df.index).fillna(0)

    def add_indicators(self, df):
        df = df.copy(); close = df['Close']; vol = df['Volume']
        df['Log_Ret'] = np.log(close / (close.shift(1) + 1e-9))
        df['Vol_20'] = df['Log_Ret'].rolling(20, min_periods=1).std()
        df['Vol_60'] = df['Log_Ret'].rolling(60, min_periods=1).std()
        delta = close.diff(); g = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean(); l = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        df['RSI'] = 100 - (100 / (1 + (g / (l + 1e-9))))
        df['MA20'] = close.rolling(20, min_periods=1).mean(); df['MA200'] = close.rolling(200, min_periods=1).mean()
        s20 = close.rolling(20, min_periods=1).std(); df['BB_Width'] = (4 * s20) / (df['MA20'] + 1e-9)
        df['BB_Squeeze'] = df['BB_Width'].rolling(20, min_periods=1).min() / (df['BB_Width'] + 1e-9)
        df['Disp_20'] = close / (df['MA20'] + 1e-9); df['Disp_200'] = close / (df['MA200'] + 1e-9)
        up = close.diff(); dn = -close.diff(); atr_i = up.abs().rolling(14, min_periods=1).mean()
        pdi = 100 * (up.where((up > dn) & (up > 0), 0).rolling(14, min_periods=1).mean() / (atr_i + 1e-9)); mdi = 100 * (dn.where((dn > up) & (dn > 0), 0).rolling(14, min_periods=1).mean() / (atr_i + 1e-9))
        df['Plus_DI'] = pdi; df['Minus_DI'] = mdi; df['ADX'] = (100 * (abs(pdi - mdi) / (pdi + mdi + 1e-9))).rolling(14, min_periods=1).mean()
        df['OBV'] = (np.sign(close.diff()) * vol).fillna(0).cumsum(); df['OBV_Slope'] = df['OBV'].diff(5) / (df['OBV'].rolling(15, min_periods=1).std() + 1e-9)
        df['Price_Slope'] = close.rolling(min(len(df)//2, 45)).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (x[-1] + 1e-9) if len(x)>1 else 0, raw=True)
        df['ATR'] = atr_i; df['VIX_Rel'] = df['VIX'].rolling(min(252, len(df)), min_periods=1).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-9) if len(x)>1 else 0.5, raw=True)
        ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean(); df['MACD'] = ema12 - ema26; df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        l14 = close.rolling(14, min_periods=1).min(); h14 = close.rolling(14, min_periods=1).max(); df['Stoch_K'] = 100 * (close - l14) / (h14 - l14 + 1e-9); df['Stoch_D'] = df['Stoch_K'].rolling(3, min_periods=1).mean()
        df['USD_KRW_Change'] = df['USD_KRW'].pct_change()
        if 'Mkt_Log_Ret' not in df: df['Mkt_Log_Ret'] = 0.0
        df['Mkt_Ref_5d'] = df['Mkt_Log_Ret'].rolling(5, min_periods=1).sum(); df['Target'] = df['Log_Ret'].rolling(5, min_periods=1).sum().shift(-5)
        # Only drop rows that have NaNs in input features (e.g. initial rolling windows), leave 'Target' NaNs for inference
        return df.dropna(subset=[c for c in df.columns if c != 'Target'])

    def prepare_sequences(self, df):
        # [CRITICAL] Training requires Target. Filter NaNs only for sequence preparation.
        df_train = df.dropna(subset=['Target'])
        seq = max(5, min(60, len(df_train) // 3))
        self.features = ['Log_Ret', 'RSI', 'BB_Width', 'Price_Slope', 'Vol_20', 'ADX', 'Plus_DI', 'Minus_DI', 'Disp_200', 'Disp_20', 'OBV_Slope', 'BB_Squeeze', 'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D', 'USD_KRW_Change', 'Mkt_Ref_5d']
        X = self.scaler_X.fit_transform(df_train[self.features]); y = self.scaler_y.fit_transform(df_train[['Target']])
        return np.array([X[i-seq:i] for i in range(seq, len(X))]), np.array([y[i] for i in range(seq, len(X))])

    def train(self, Xt, yt, s=42):
        set_seed(s); m = TransformerPredictor(input_size=len(self.features))
        ld = DataLoader(TimeSeriesDataset(Xt, yt), batch_size=32, shuffle=False)
        opt = torch.optim.AdamW(m.parameters(), lr=0.001); crit = nn.MSELoss()
        for _ in range(10):
            m.train()
            for bx, by in ld: opt.zero_grad(); loss = crit(m(bx), by); loss.backward(); opt.step()
        return m

    def evaluate_strategy(self, dft, params=None):
        if self.model is None: return None, {'total_return':-1}
        self.model.eval(); p = params if params else self.best_params
        tp_b = p.get('tp_atr', 3.2); sl_b = p.get('sl_atr', 1.5); eth_b = p.get('entry_threshold', 0.35)
        seq = max(5, min(60, len(dft)//3)); Xs = self.scaler_X.transform(dft[self.features])
        X_tns = torch.FloatTensor([Xs[i-seq:i] for i in range(seq, len(Xs))])
        with torch.no_grad(): pred = self.scaler_y.inverse_transform(self.model(X_tns).numpy()).flatten()
        res = dft.iloc[seq:].copy(); res['Pred_Ret'] = pred; res['Strategy_Return'] = 0.0; res['Year'] = res.index.year
        # Strictly v3.1 Scoring Logic
        res['Total_Score'] = (np.where(res['Plus_DI']>res['Minus_DI'], 0.5, -0.45) + np.where(res['OBV_Slope']>1.0, 0.5, -0.25) + np.where(res['MACD']>res['MACD_Signal'], 0.3, -0.3) + np.where(res['Stoch_K']>res['Stoch_D'], 0.2, -0.2)) + (res['Pred_Ret']*55)
        
        pos = 0; ent = 0; hi = 0; ytd = 0.0; yr = res['Year'].iloc[0]; days = 0; sigs = np.zeros(len(res)); rets = np.zeros(len(res))
        px = res['Close'].values; atrs = res['ATR'].values; mk = res['Close'].pct_change().fillna(0).values
        for i in range(1, len(res)):
            if res['Year'].iloc[i] != yr: yr = res['Year'].iloc[i]; ytd = 0.0
            if ytd < 0: def_m = 0.7; eth = eth_b * 1.5; sl = sl_b * 0.7; tp = tp_b * 1.2
            elif ytd > 0.15: def_m = 0.8; eth = eth_b * 1.1; sl = sl_b * 0.6; tp = tp_b * 0.8
            else: def_m = 1.0; eth = eth_b; sl = sl_b; tp = tp_b
            if self.is_downtrend_mode: tp *= 0.6; sl *= 1.2
            v_re = (res['VIX_Rel'].iloc[i]>0.7) and (res['VIX'].iloc[i]<res['VIX'].iloc[i-1]); final_th = eth if not v_re else eth * 0.85
            sz = min(max(def_m * (res['Pred_Ret'].iloc[i-1]/(res['Vol_20'].iloc[i-1]**2+1e-9)), 0), 1.0)
            cost = 0.0
            if pos == 0:
                if res['Total_Score'].iloc[i-1] > final_th or (res['Total_Score'].iloc[i-1] > 0 and (res['Disp_20'].iloc[i-1]<0.94 or res['RSI'].iloc[i-1]<24)):
                    # [5% PROFIT GUARD]
                    potential_profit = (tp * atrs[i]) / (px[i] + 1e-9)
                    if potential_profit >= 0.05:
                        pos = 1; ent = px[i]; hi = px[i]; days = 0; sigs[i] = sz
            else:
                hi = max(hi, px[i]); days += 1; tp_p = ent + tp * atrs[i]; sl_p = hi - sl * atrs[i]
                tx = (self.is_downtrend_mode and days>=5) or (days>30)
                if px[i] < sl_p or px[i] > tp_p or res['Total_Score'].iloc[i-1] < -0.4 or tx: pos = 0; sigs[i] = -1; cost = sz * 0.003
            rets[i] = (sigs[i] if pos==1 and sigs[i]>0 else (sz if pos==1 else 0)) * mk[i] - cost; ytd = (1 + ytd) * (1 + rets[i]) - 1
        res['Strategy_Return'] = rets; cum = (1 + rets).cumprod(); tr = []
        if len(res) > 0:
            st = -1
            for i in range(len(res)):
                if sigs[i] > 0 and st == -1: st = i
                elif sigs[i] == -1 and st != -1: tr.append({'year': res['Year'].iloc[i], 'pnl': (1+rets[st:i+1]).prod()-1}); st = -1
        mdd = np.max((np.maximum.accumulate(cum) - cum) / (np.maximum.accumulate(cum) + 1e-9))
        return res, {'total_return': cum[-1]-1, 'sharpe': (rets.mean()/(rets.std()+1e-9))*np.sqrt(252), 'mdd': mdd, 'trades': tr}

    def generate_report(self, res, met):
        def get_w(s): return sum(2 if ord(c)>127 else 1 for c in str(s))
        def pad(s, w, align='center'):
            s = str(s); d = w - get_w(s); l = d // 2 if align == 'center' else 0; r = d - l
            return " " * l + s + " " * r
        print("\n"+"="*85+f"\n ALPHA ENGINE SIGMA v3.2 MASTER PRECISION ({self.name})\n"+"="*85)
        if self.maintenance_reason: print(f" [!] {self.maintenance_reason}")
        if self.legacy_data and self.maintenance_reason:
            print(f" [Legacy] 역대 최고 성적 복원 (Score: {self.legacy_data.get('score', 0):.1f})")
            ret_disp = self.legacy_data.get('total_return', met['total_return'])
            sharp_disp = self.legacy_data.get('sharpe', met['sharpe']); mdd_disp = self.legacy_data.get('mdd', met['mdd'])
            yr_data = self.legacy_data.get('annual'); trades = self.legacy_data.get('trades_summary')
            actual_trades = met['trades']
        else:
            ret_disp = met['total_return']; sharp_disp = met['sharpe']; mdd_disp = met['mdd']
            yr_rets = res.groupby(res.index.year)['Strategy_Return'].apply(lambda x: (1+x).prod()-1)
            yr_data = yr_rets.to_dict(); tr_df = pd.DataFrame(met['trades']); trades = []
            for y, r in yr_data.items():
                yt = tr_df[tr_df['year'] == y] if not tr_df.empty else pd.DataFrame()
                if not yt.empty:
                    cnt=len(yt); wr=f"{len(yt[yt['pnl']>0])/cnt:.2%}"
                    # Robust AW/AL calculation to prevent nan
                    wins = yt[yt['pnl']>0]['pnl']
                    losses = yt[yt['pnl']<=0]['pnl']
                    aw = wins.mean() if not wins.empty else 0
                    al = abs(losses.mean()) if not losses.empty else 0
                    pls = f"{aw/(al+1e-9):.2f}" if al > 0 else "손실없음"
                else: cnt=0; wr="0.00%"; pls="0.00"
                trades.append({'year':y, 'ret':r, 'pls':pls, 'wr':wr, 'cnt':cnt})
            actual_trades = met['trades']

        # Calculate Overall Average Profit-Loss Ratio
        if actual_trades:
            tr_df_total = pd.DataFrame(actual_trades)
            avg_win = tr_df_total[tr_df_total['pnl'] > 0]['pnl'].mean() if not tr_df_total[tr_df_total['pnl'] > 0].empty else 0
            avg_loss = abs(tr_df_total[tr_df_total['pnl'] <= 0]['pnl'].mean()) if not tr_df_total[tr_df_total['pnl'] <= 0].empty else 0
            total_pls_disp = f"{avg_win/avg_loss:.2f}" if avg_loss > 0 else "손실없음"
        else:
            total_pls_disp = "데이터없음"

        print(f" [종합 실적] 누적 수익률: {ret_disp:>8.2%} | 샤프: {sharp_disp:>5.2f} | MDD: {mdd_disp:>6.2%} | 평균손익비: {total_pls_disp}")
        print("\n [연도별 세부 실적 일람표]"); div = "-"*85; print(div)
        print(" "+" | ".join(pad(h, w) for h,w in zip(["해당연도","수익률","평균손익비","적중율","평균거래건수"],[12,16,18,14,16]))); print(div)
        for t in trades: print(" "+" | ".join(pad(d, w) for d,w in zip([f"{t['year']}년", f"{t['ret']:.2%}", t['pls'], t['wr'], str(t['cnt'])],[12,16,18,14,16])))
        print(div)
        
        # --- Trading Strategy Suggestion (매매전략 제안) ---
        last = res.iloc[-1]; lp = last['Close']; latr = last['ATR']; score = last['Total_Score']
        eth = self.best_params.get('entry_threshold', 0.35)
        tp_m = self.best_params.get('tp_atr', 3.2); sl_m = self.best_params.get('sl_atr', 1.5)
        if self.is_downtrend_mode: tp_m *= 0.6; sl_m *= 1.2
        print("\n [실시간 매매전략 제안]")
        if score > eth or (score > 0 and (last['Disp_20']<0.94 or last['RSI']<24)):
            # Special 5% logic check for suggestion
            potential_profit = (tp_m * latr) / (lp + 1e-9)
            if potential_profit >= 0.05:
                print(f"  ▶ 현재 포지션: [진입 권장] (AI Score: {score:.2f}, 기대이익: {potential_profit:.1%})")
                print(f"  ▶ 권장 진입가: {lp:,.0f}원 이하 | 목표가: {lp+tp_m*latr:,.0f}원 | 손절가: {lp-sl_m*latr:,.0f}원")
            else:
                print(f"  ▶ 현재 포지션: [관망 권장] (AI Score: {score:.2f}, 기대이익 {potential_profit:.1%}로 기준치 5% 미달)")
        else:
            print(f"  ▶ 현재 포지션: [관망 권장] (AI Score: {score:.2f})")
        print("="*85)

    def run_persistence_evolution(self, rounds=1):
        pattern = os.path.join(self.log_dir, f"{self.name}_{self.ticker}_*.txt"); all_logs = glob.glob(pattern)
        all_logs.sort(key=os.path.getmtime, reverse=True); g_best_score = -np.inf; best_meta = None
        for fp in all_logs:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    m = json.loads(f.read().split("### METADATA ###")[1].strip())
                    if m.get('score', -1) > g_best_score: g_best_score = m['score']; best_meta = m
            except: pass
        if best_meta: print(f" [Persistence] 역대 최고 성과 탐지 (Score: {g_best_score:.1f})")
        split = int(len(self.df)*0.8); df_tr = self.add_indicators(self.df.iloc[:split]); df_te = self.add_indicators(self.df)
        Xt, yt = self.prepare_sequences(df_tr); c_best_score = -np.inf; c_best_params = None; c_best_model = None
        for r in range(rounds):
            m = self.train(Xt[:int(len(Xt)*0.9)], yt[:int(len(Xt)*0.9)], 42+r); self.model = m
            grid = {'entry_threshold':[0.3, 0.35, 0.4], 'tp_atr':[3.0, 3.2, 3.5], 'sl_atr':[1.2, 1.5]}
            keys = list(grid.keys()); combos = list(itertools.product(*grid.values()))
            for c in tqdm(combos, desc=f" [Scan] Seed {42+r}", leave=False):
                ps = dict(zip(keys, c)); ps.update({'lrs_period':45, 'rsi_period':14})
                _, mt = self.evaluate_strategy(df_te, ps); yp = (1+_['Strategy_Return']).groupby(_.index.year).prod()-1
                sc = (mt['total_return']*100 + mt['sharpe']*300 - (yp<0).sum()*5000)
                if sc > c_best_score: c_best_score = sc; c_best_params = ps; c_best_model = m
        if best_meta and g_best_score >= 0 and g_best_score >= c_best_score: self.maintenance_reason = "전회보다 당회의 누적수익률이 좋지 못해 전회의 실적유산과 환경을 그대로 승계유지"; self.best_params = best_meta['params']; self.legacy_data = best_meta
        else:
            self.best_params = c_best_params; self.legacy_data = None; self.maintenance_reason = ""
            if best_meta and g_best_score < 0: print(f" [!] 과거 기록이 손실 상태이므로 승계를 거부하고 금회 성과를 채택합니다.")
        self.model = c_best_model; f_df, f_met = self.evaluate_strategy(df_te, self.best_params); self.generate_report(f_df, f_met)
        yr_rets = f_df.groupby(f_df.index.year)['Strategy_Return'].apply(lambda x: (1+x).prod()-1); tr_df = pd.DataFrame(f_met['trades']); trades_summary = []
        for y, r in yr_rets.items():
            yt = tr_df[tr_df['year'] == y] if not tr_df.empty else pd.DataFrame()
            if not yt.empty:
                cnt=len(yt); wr=f"{len(yt[yt['pnl']>0])/cnt:.2%}"
                wins = yt[yt['pnl']>0]['pnl']; losses = yt[yt['pnl']<=0]['pnl']
                aw = wins.mean() if not wins.empty else 0
                al = abs(losses.mean()) if not losses.empty else 0
                pls = f"{aw/(al+1e-9):.2f}" if al > 0 else "손실없음"
            else: cnt=0; wr="0.00%"; pls="0.00"
            trades_summary.append({'year':y, 'ret':r, 'pls':pls, 'wr':wr, 'cnt':cnt})
        # Prepare Metadata - If inheriting, preserve legacy metrics to avoid poisoning
        if self.legacy_data and self.maintenance_reason:
            save_meta = self.legacy_data
        else:
            save_meta = {
                'params': self.best_params,
                'score': c_best_score,
                'total_return': f_met['total_return'],
                'sharpe': f_met['sharpe'],
                'mdd': f_met['mdd'],
                'annual': yr_rets.to_dict(),
                'trades_summary': trades_summary
            }
        
        dt = datetime.now().strftime("%Y%m%d_%H%M%S"); fn = f"{self.name}_{self.ticker}_{dt}.txt"; fp = os.path.join(self.log_dir, fn)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(f"=== ALPHA ENGINE REPORT v3.2: {self.name} ({self.ticker}) ===\n Date: {dt}\n")
            if self.maintenance_reason: f.write(f" [!] {self.maintenance_reason}\n")
            f.write(f" ### METADATA ###\n{json.dumps(save_meta)}\n")
        lf = sorted(glob.glob(os.path.join(self.log_dir, f"{self.name}_{self.ticker}_*.txt")), reverse=True)[5:]; [os.remove(old) for old in lf]

def main():
    assets = [("KODEX 코스피", "226490.KS"), ("KODEX 인버스", "114800.KS"), ("ACE KRX 금현물", "411060.KS"), ("KODEX 은선물(H)", "144600.KS"), ("RISE 글로벌 자산배분 액티브", "461490.KS"), ("RISE 글로벌농업경제", "437370.KS"), ("KODEX WTI 원유선물(H)", "261220.KS"), ("삼성전자", "005930.KS"), ("SK 하이닉스", "000660.KS")]
    while True:
        os.system('cls' if os.name == 'nt' else 'clear'); print("\n"+"═"*70+"\n    ALPHA ENGINE SIGMA v3.2 MASTER PRECISION\n"+"═"*70)
        for i, (n, t) in enumerate(assets, 1): print(f"  {i:2d}. {n} ({t})")
        print(f"  10. [RUN ALL] 모든 종목 일괄 분석 및 순차 진화\n  11. [RUN ALL x10] 모든 종목 10회 검증 및 최강 지능 추출\n  12. [SELECT x10] 해당 종목만 10회 검증 및 최강 지능 추출\n  Q.  [EXIT] 종료\n"+"═"*70)
        choice = input("\n [Choice]: ").upper()
        if choice == 'Q': break
        if choice == '12':
            sub = input(" > 10회 검증할 종목 번호를 선택하세요 (1-9): ")
            if sub.isdigit() and 1 <= int(sub) <= 9: n, t = assets[int(sub)-1]; e = AlphaEngineSigma(t, n); e.fetch_data(); e.run_persistence_evolution(rounds=10)
            else: print(" [!] 잘못된 번호입니다.")
        else:
            rnds = 10 if choice == '11' else 1
            to_run = assets if choice in ['10', '11'] else ([assets[int(choice)-1]] if choice.isdigit() and 1 <= int(choice) <= 9 else [])
            for n, t in to_run:
                try: e = AlphaEngineSigma(t, n); e.fetch_data(); e.run_persistence_evolution(rounds=rnds)
                except Exception as ex: print(f" Error: {ex}")
        input("\n [!] 완료. 엔터...")
if __name__ == "__main__": main()
