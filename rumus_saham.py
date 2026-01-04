import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. ALAT BANTU HITUNG (MATH INDICATORS)
# ==========================================

def hitung_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def hitung_bollinger(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + (std * 2), sma - (std * 2)

def hitung_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def hitung_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def hitung_stochastic(high, low, close, k_window=14, d_window=3):
    low_min = low.rolling(window=k_window).min()
    high_max = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def hitung_vwap(df):
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).cumsum() / v.cumsum()

# ==========================================
# 2. FUNGSI AMBIL BERITA (ANTI CRASH)
# ==========================================
def ambil_berita_saham(ticker):
    try:
        kode_bersih = ticker.replace(".JK", "")
        if not ticker.endswith(".JK"): ticker += ".JK"
        stock = yf.Ticker(kode_bersih)
        raw_news = stock.news
        berita_bersih = []
        if raw_news:
            for n in raw_news[:5]: 
                link = n.get('link', '')
                if not link: link = f"https://finance.yahoo.com/quote/{kode_bersih}"
                ts = n.get('providerPublishTime', 0)
                try: tgl = datetime.fromtimestamp(ts).strftime('%d %b %H:%M')
                except: tgl = "Terkini"
                berita_bersih.append({
                    "title": n.get('title', 'Update Pasar'),
                    "publisher": n.get('publisher', 'Yahoo Finance'),
                    "link": link,
                    "date": tgl
                })
        if not berita_bersih:
            berita_bersih.append({"title": f"Info {kode_bersih}", "publisher": "System", "link": f"https://finance.yahoo.com/quote/{kode_bersih}", "date": "Now"})
        return berita_bersih
    except: return []

# ==========================================
# 3. OTAK UTAMA: ANALISA MULTI-STRATEGY PRO
# ==========================================
def analisa_multistrategy(ticker):
    try:
        if not ticker.endswith(".JK"): ticker += ".JK"
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y") 
        info = stock.info 
        
        if df.empty or len(df) < 50: 
            return {"verdict": "SKIP", "reason": "Data Kurang", "score": 0, "type": "UNKNOWN", "last_price": 0, "change_pct": 0, "support": 0}

        # --- VARIABEL DATA MENTAH ---
        last = df.iloc[-1]
        last_price = int(last['Close'])
        prev_close = df.iloc[-2]['Close']
        change_pct = (last['Close'] - prev_close) / prev_close
        
        open_price = last['Open']
        close_price = last['Close']
        high_price = last['High']
        low_price = last['Low']
        volume_now = last['Volume']

        # --- HITUNG INDIKATOR TEKNIKAL ---
        sma_5 = df['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) > 200 else 0
        vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        rsi = hitung_rsi(df['Close']).iloc[-1]
        bb_upper, bb_lower = hitung_bollinger(df['Close'])
        last_bb_lower = bb_lower.iloc[-1]
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma_20
        
        macd, macd_signal = hitung_macd(df['Close'])
        last_macd = macd.iloc[-1]
        last_signal = macd_signal.iloc[-1]
        
        obv = hitung_obv(df['Close'], df['Volume'])
        last_obv = obv.iloc[-1]
        prev_obv = obv.iloc[-5] 

        stoch_k, stoch_d = hitung_stochastic(df['High'], df['Low'], df['Close'])
        last_k = stoch_k.iloc[-1]
        last_d = stoch_d.iloc[-1]

        vwap_series = hitung_vwap(df)
        last_vwap = vwap_series.iloc[-1]

        pe = info.get('trailingPE', 100)
        pbv = info.get('priceToBook', 10)
        volatility = (high_price - low_price) / low_price if low_price > 0 else 0

        # --- [FITUR BARU] FILTER DOWNTREND KERAS ---
        is_downtrend = last_price < sma_50 and last_price < sma_200

        # ==========================================
        # SISTEM SKORING (18 VARIABEL TETAP LENGKAP)
        # ==========================================
        scores = {"BSJP": 0, "BPJS": 0, "SCALPING": 0, "SWING": 0, "ARA": 0, "INVEST": 0}
        reasons = []

        # 1. ARA HUNTER
        if change_pct > 0.10: scores["ARA"] += 40
        if volume_now > (vol_avg * 3): scores["ARA"] += 30
        if close_price >= (high_price * 0.98): scores["ARA"] += 20
        if last_obv > prev_obv: scores["ARA"] += 10 

        # 2. SCALPING
        if volatility > 0.03: scores["SCALPING"] += 30
        if volume_now > (vol_avg * 1.5): scores["SCALPING"] += 20
        if close_price > last_vwap: scores["SCALPING"] += 30
        if rsi > 55: scores["SCALPING"] += 20

        # 3. SWING
        if sma_20 > sma_50: scores["SWING"] += 20
        if close_price > sma_20: scores["SWING"] += 20
        if last_k > last_d and last_k < 80: scores["SWING"] += 30
        if last_macd > last_signal: scores["SWING"] += 20 
        if bb_width < 0.15: scores["SWING"] += 10 

        # 4. BSJP
        if close_price > open_price and close_price > sma_5: scores["BSJP"] += 40
        if close_price >= (high_price * 0.99): scores["BSJP"] += 30
        if volume_now > vol_avg: scores["BSJP"] += 30

        # 5. BPJS
        if rsi < 35: scores["BPJS"] += 40 
        if close_price < last_bb_lower: scores["BPJS"] += 30 
        if last_k < 20 and last_k > last_d: scores["BPJS"] += 30 

        # 6. INVEST
        if pe < 15 and pe > 0: scores["INVEST"] += 30
        if pbv < 1.5: scores["INVEST"] += 30
        if close_price > sma_200: scores["INVEST"] += 20 
        if sma_50 > sma_200: scores["INVEST"] += 20 

        # ==========================================
        # KEPUTUSAN FINAL & LOGIKA SUPPORT SMART
        # ==========================================
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # [FITUR PERBAIKAN] MENENTUKAN ENTRY (WAJIB DI BAWAH)
        harga_support = last_price 
        
        if best_type == "BSJP":
            harga_support = max(sma_5, last_vwap)
        elif best_type == "BPJS":
            harga_support = last_bb_lower
        elif best_type == "SWING":
            harga_support = sma_20
        elif best_type == "SCALPING":
            harga_support = last_vwap
        elif best_type == "INVEST":
            harga_support = sma_50
        elif best_type == "ARA":
            pivot = (high_price + low_price + close_price) / 3
            harga_support = pivot

        # --- PROTEKSI 1: ANTI ANTRI DI ATAS HARGA PASAR ---
        if best_type != "ARA": 
            if harga_support > last_price:
                # Jika hitungan support di atas harga (karena longsor), paksa ke Low Hari ini
                harga_support = low_price
                reasons.append("Support Jebol (Adjusted to Low)")

        # --- PROTEKSI 2: VERDICT DOWNTREND ---
        verdict = "WAIT"
        if is_downtrend and best_type != "INVEST" and best_type != "BPJS":
            verdict = "AVOID / SELL ❌"
            reasons.append("⚠️ Downtrend Parah (Dibawah MA50/200)")
            best_score = 30 # Turunkan skor agar tidak memicu STRONG BUY
        elif best_score >= 80: verdict = "STRONG BUY 🔥"
        elif best_score >= 60: verdict = "BUY ✅"
        elif best_score <= 40: verdict = "SELL"
        
        # Alasan Dinamis
        if best_type == "ARA": reasons.append("Volume Spike")
        elif best_type == "SCALPING": reasons.append("Harga > VWAP")
        elif best_type == "SWING": reasons.append("MA Uptrend")
        elif best_type == "BSJP": reasons.append("Closing Kuat")
        
        if last_macd > last_signal: reasons.append("✅ MACD (+)")
        if last_obv > prev_obv: reasons.append("✅ Akumulasi")

        return {
            "score": int(best_score),
            "verdict": verdict,
            "type": best_type,
            "reason": " | ".join(reasons),
            "last_price": last_price,
            "change_pct": round(change_pct * 100, 2),
            "support": int(harga_support)
        }

    except Exception as e:
        return {
            "score": 0, "verdict": "ERROR", "type": "ERROR", 
            "reason": str(e), "last_price": 0, "change_pct": 0, "support": 0
        }