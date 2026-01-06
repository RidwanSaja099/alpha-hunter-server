import os
import time
import concurrent.futures
import yfinance as yf
import pandas as pd
import numpy as np
import math
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# --- IMPORT LIBRARY AI & SEARCH ---
try:
    from duckduckgo_search import DDGS 
except ImportError:
    DDGS = None
    print("⚠️ Warning: duckduckgo_search tidak ditemukan.")

from groq import Groq 
from openai import OpenAI 

try:
    from google import genai
except ImportError:
    genai = None

load_dotenv()

# Pastikan file rumus_saham.py ada di folder yang sama
from rumus_saham import analisa_multistrategy, ambil_berita_saham 

app = Flask(__name__)

# ==========================================
# 0. KONFIGURASI AI CLIENT (ANTI-CRASH)
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client_groq = None
if GROQ_API_KEY:
    try: client_groq = Groq(api_key=GROQ_API_KEY)
    except: print("⚠️ Gagal Init Groq")

client_deepseek = None
if DEEPSEEK_API_KEY:
    try: client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except: print("⚠️ Gagal Init DeepSeek")

client_gemini = None
if GEMINI_API_KEY and genai:
    try: client_gemini = genai.Client(api_key=GEMINI_API_KEY)
    except: print("⚠️ Gagal Init Gemini")

# ==========================================
# 1. FITUR V14: MESIN 13 INDIKATOR (GOD MODE)
# ==========================================
def hitung_indikator_lengkap(ticker_lengkap):
    try:
        df = yf.Ticker(ticker_lengkap).history(period="1y")
        if len(df) < 100: return "Data Historis Tidak Cukup."

        # Data Dasar
        close = df['Close'].iloc[-1]
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]

        # --- GROUP 1: MOMENTUM (4 Indikator) ---
        # 1. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # 2. Stochastic
        low14 = df['Low'].rolling(window=14).min().iloc[-1]
        high14 = df['High'].rolling(window=14).max().iloc[-1]
        stoch_k = 100 * ((close - low14) / (high14 - low14))

        # 3. MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12.iloc[-1] - exp26.iloc[-1]
        signal_line = (exp12 - exp26).ewm(span=9, adjust=False).mean().iloc[-1]
        macd_hist = macd_line - signal_line

        # 4. OBV (Smart Money)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_trend = "NAIK (Akumulasi)" if obv.iloc[-1] > obv.iloc[-5] else "TURUN (Distribusi)"

        # --- GROUP 2: VOLATILITAS & TREN (4 Indikator) ---
        # 5. Bollinger Bands
        ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        std = df['Close'].rolling(window=20).std().iloc[-1]
        upper_bb = ma20 + (2 * std)
        lower_bb = ma20 - (2 * std)
        
        # 6. ATR (Napas Saham)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().iloc[-1]

        # 7. MA Trend
        ma5 = df['Close'].rolling(window=5).mean().iloc[-1]
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) > 200 else ma20
        trend_long = "BULLISH" if close > ma200 else "BEARISH"

        # 8. Volume Ratio
        vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
        vol_ratio = df['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 0

        # --- GROUP 3: STRUKTUR & PREDIKSI (NEW V14 - 5 Indikator) ---
        
        # 9. Ichimoku Cloud (Simplified)
        # Tenkan-sen (Conversion Line) - 9 periods
        high9 = df['High'].rolling(window=9).max().iloc[-1]
        low9 = df['Low'].rolling(window=9).min().iloc[-1]
        tenkan = (high9 + low9) / 2
        # Kijun-sen (Base Line) - 26 periods
        high26 = df['High'].rolling(window=26).max().iloc[-1]
        low26 = df['Low'].rolling(window=26).min().iloc[-1]
        kijun = (high26 + low26) / 2
        # Span A & B (Cloud Future)
        span_a = (tenkan + kijun) / 2
        high52 = df['High'].rolling(window=52).max().iloc[-1]
        low52 = df['Low'].rolling(window=52).min().iloc[-1]
        span_b = (high52 + low52) / 2
        
        ichi_status = "DI ATAS CLOUD (Strong)" if close > span_a and close > span_b else \
                      "DI DALAM CLOUD (Consolidation)" if (close > span_a and close < span_b) or (close < span_a and close > span_b) else \
                      "DI BAWAH CLOUD (Weak)"

        # 10. Fibonacci Retracement (Auto Swing High/Low 3 Bulan)
        last_3m = df[-60:]
        swing_high = last_3m['High'].max()
        swing_low = last_3m['Low'].min()
        fibo_618 = swing_high - ((swing_high - swing_low) * 0.618) # Golden Ratio Support
        fibo_382 = swing_high - ((swing_high - swing_low) * 0.382) # Resistance

        # 11. TTM Squeeze (Manual Logic)
        # Jika Bollinger Band masuk ke dalam Keltner Channel (disini kita pakai logika penyempitan STD)
        bb_width = (upper_bb - lower_bb) / ma20
        avg_width = bb_width # Simplified snapshot
        squeeze_status = "SIAP MELEDAK (Squeeze)" if std < (df['Close'].rolling(20).std().mean()) else "Normal"

        # 12. Pivot Points
        pp = (df['High'].iloc[-2] + df['Low'].iloc[-2] + df['Close'].iloc[-2]) / 3
        r1 = (2 * pp) - df['Low'].iloc[-2]
        s1 = (2 * pp) - df['High'].iloc[-2]

        return f"""
        [DATA TEKNIKAL 13 INDIKATOR - GOD MODE]
        A. MOMENTUM:
        1. RSI: {rsi:.2f}
        2. Stoch %K: {stoch_k:.2f}
        3. MACD Hist: {macd_hist:.2f}
        4. OBV Trend: {obv_trend}

        B. STRUKTUR & TREN:
        5. Tren MA200: {trend_long}
        6. Ichimoku: {ichi_status} (Tenkan: {tenkan:.0f}, Kijun: {kijun:.0f})
        7. Fibonacci Golden (Sup): {fibo_618:.0f} | Res: {fibo_382:.0f}
        8. Pivot Point: S1={s1:.0f} | R1={r1:.0f}

        C. VOLATILITAS & BANDAR:
        9.  Bollinger Pos: {(close - lower_bb)/(upper_bb-lower_bb):.2f} (0=Bawah, 1=Atas)
        10. TTM Squeeze: {squeeze_status}
        11. ATR (Risk): {atr:.0f}
        12. Volume Ratio: {vol_ratio:.2f}x
        13. Posisi Harga: {close:.0f}
        """
    except Exception as e: return f"Error Hitung: {e}"

# ==========================================
# 2. FITUR V7: AGEN PENCARI BERITA (HYBRID + SEKTORAL)
# ==========================================
def dapatkan_keywords_cerdas(ticker, sektor):
    sektor = sektor.upper() if sektor else "GENERAL"
    query = f"berita saham {ticker} indonesia terbaru hari ini sentimen"
    
    # Menambah konteks pencarian agar lebih pintar
    if any(x in sektor for x in ["GOLD", "MINING", "METAL"]): query += " + harga komoditas emas nikel dunia"
    elif any(x in sektor for x in ["OIL", "ENERGY"]): query += " + harga minyak brent crude oil"
    elif "COAL" in sektor or ticker in ["ADRO", "PTBA", "ITMG"]: query += " + harga batubara newcastle"
    elif "BANK" in sektor: query += " + suku bunga BI rate rupiah"
    elif "TECH" in sektor: query += " + saham teknologi nasdaq goto"
    elif "CPO" in sektor or "PLANTATION" in sektor: query += " + harga CPO malaysia"
    return query

def agen_pencari_berita_robust(ticker, sektor, berita_yahoo_backup):
    laporan_mentah = ""
    sumber_data = "YAHOO (BACKUP)"

    if DDGS:
        try:
            query = dapatkan_keywords_cerdas(ticker, sektor)
            print(f"🌍 Searching: {query}")
            results = DDGS().text(query, max_results=5)
            if results:
                ddg_text = [f"- {r['title']}: {r['body']}" for r in results]
                laporan_mentah = "\n".join(ddg_text)
                sumber_data = "INTERNET (REAL-TIME)"
        except Exception as e: print(f"⚠️ DDG Error: {e}")

    if not laporan_mentah or len(laporan_mentah) < 50:
        yahoo_text = [f"- {b.get('title', '')}" for b in berita_yahoo_backup]
        laporan_mentah = "\n".join(yahoo_text) if yahoo_text else "Tidak ada berita spesifik."

    # Gunakan Groq untuk merangkum agar rapi (jika ada)
    if client_groq: 
        try:
            prompt_wartawan = f"""
            Kamu adalah Reporter Pasar Modal.
            DATA MENTAH ({sumber_data}):
            {laporan_mentah}
            
            TUGAS:
            1. Ambil inti berita yang relevan dengan harga saham.
            2. Jika ada sentimen komoditas/global, masukkan.
            3. Buat ringkasan padat 3 poin.
            """
            chat = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt_wartawan}],
                model="llama-3.3-70b-versatile",
            )
            return f"**SUMBER: {sumber_data}**\n" + chat.choices[0].message.content.strip()
        except: pass

    return f"[{sumber_data}] {laporan_mentah}"

# ==========================================
# 3. FITUR V9: AGEN KEPALA ANALIS (ANTI-OFFLINE / FAILOVER)
# ==========================================
def agen_analis_utama(data_context):
    """
    Prompt diperbarui untuk membaca 13 Indikator (Ichimoku, Fibo, TTM Squeeze, dll)
    dan memberikan Action Plan Presisi.
    """
    prompt_analis = f"""
    Kamu adalah Elite Fund Manager & Ahli Strategi Saham (Quantitative Expert).
    
    TUGAS UTAMA:
    Analisa saham ini berdasarkan **13 DATA INDIKATOR TEKNIKAL (GOD MODE)** yang disediakan di bawah.
    
    DATA LENGKAP:
    {data_context}
    
    ⚠️ **SOP ANALISIS (WAJIB PATUH AGAR SEJALAN DENGAN SCANNER):**
    1. **Momentum (RSI & Stoch):** Jika RSI > 60 dan Stochastic naik, itu **MOMENTUM KUAT** (Bukan Overbought). Sarankan BUY/FOLLOW TREND.
    2. **Struktur (Ichimoku & MA):** Jika Harga > Awan Ichimoku & > MA200 = **SUPER BULLISH**. Jika di dalam Awan = Hati-hati (Konsolidasi).
    3. **Bandar (OBV & Volume):** Jika OBV naik dan Volume > 1.2x Rata-rata, konfirmasi **AKUMULASI BANDAR**.
    4. **Volatilitas (TTM Squeeze):** Jika status "SIAP MELEDAK", bersiap untuk **Buy on Breakout**.
    5. **Area Penting (Fibo & Pivot):** Gunakan Fibonacci 0.618 sebagai Support Emas, dan Pivot R1/R2 sebagai Target.
    6. **Fundamental:** Jika Teknikal "Perfect" tapi Fundamental jelek, tetap berikan rekomendasi **TRADING CEPAT (Hit & Run)**.

    JAWAB 6 POIN INI SECARA TEGAS & DATA-DRIVEN:
    
    1. 🌍 **Korelasi Berita & Makro** (Apakah sentimen sektoral mendukung data teknikal?)
       
    2. 🕵️‍♂️ **Bandarmologi & Smart Money** (Analisis OBV & Volume Ratio. Apakah Bandar sedang Akumulasi, Mark-Up, atau Distribusi?)
       
    3. 📊 **Valuasi & Fundamental** (Review PER/PBV. Apakah murah atau mahal? Layak invest atau cuma trading?)
       
    4. ⏱️ **Kekuatan Tren & Struktur (13 Indikator)** (Sintesa dari Ichimoku, MA200, MACD, dan TTM Squeeze. Apakah Trend Valid?)

    5. 🎯 **ACTION PLAN PRESISI (WAJIB ISI ANGKA)**
       - **STRATEGI:** (Pilih: SCALPING / SWING / INVEST / BPJS / BSJP / HINDARI / CALON ARA).
       - **TIMING MASUK:** (HAKA Pagi / Tunggu Koreksi Sesi 1 / Buy on Breakout Sore).
       - **AREA ENTRY:** Tentukan harga beli (Gunakan Fibonacci Support atau Pivot S1).
       - **TARGET PROFIT (TP):** Berikan TP1, TP2, dan TP3 (Gunakan Fibo Resistance/Pivot R1).
       - **STOP LOSS (SL):** Titik cut loss aman (Gunakan ATR atau Pivot S2).

    6. ⚖️ **VERDICT FINAL** (STRONG BUY / BUY / WAIT / SELL).
    
    Jawab tegas, gunakan angka dari data indikator di atas sebagai bukti analisamu.
    """
    # --- OPSI 1: DEEPSEEK (PRIORITAS) ---
    if client_deepseek:
        try:
            print("🤖 Mencoba DeepSeek...")
            res = client_deepseek.chat.completions.create(
                model="deepseek-chat", messages=[{"role": "user", "content": prompt_analis}]
            )
            return res.choices[0].message.content.strip()
        except Exception as e: print(f"⚠️ DeepSeek Gagal: {e}")

    # --- OPSI 2: GROQ (CADANGAN PERTAMA) ---
    if client_groq:
        try:
            print("⚡ Switch ke Groq...")
            chat = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt_analis}],
                model="llama-3.3-70b-versatile"
            )
            return chat.choices[0].message.content.strip()
        except Exception as e: print(f"⚠️ Groq Gagal: {e}")

    # --- OPSI 3: GEMINI (CADANGAN TERAKHIR) ---
    if client_gemini:
        try:
            print("🌟 Switch ke Gemini...")
            return client_gemini.models.generate_content(
                model='gemini-1.5-flash', contents=prompt_analis
            ).text.strip()
        except Exception as e: print(f"⚠️ Gemini Gagal: {e}")
            
    return "⚠️ SYSTEM ERROR: Semua AI (DeepSeek, Groq, Gemini) tidak merespons. Cek kuota API/Koneksi."

# ==========================================
# 4. DATABASE & UTILS
# ==========================================
CACHE_DATA = {}
CACHE_TIMEOUT = 300 
MARKET_STATUS = {"condition": "NORMAL", "last_check": 0}

DATABASE_SYARIAH = ["ADRO", "ANTM", "ASII", "BRIS", "GOTO", "TLKM", "UNTR", "ICBP", "INDF", "PGAS", "PTBA", "MDKA", "ACES", "ELSA"]
MARKET_UNIVERSE = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "GOTO", "ANTM", "ADRO", "BREN", "AMMN", "MDKA"]
WATCHLIST = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "GOTO", "ANTM", "ADRO", "UNTR"]

def validasi_histori_panjang(ticker_lengkap, data_short):
    try:
        hist = yf.Ticker(ticker_lengkap).history(period="1y")
        if hist.empty: return 0, {} 
        current_price = data_short['last_price']
        price_1y_ago = hist['Close'].iloc[0]
        max_1y = hist['High'].max(); min_1y = hist['Low'].min()
        avg_vol = hist['Volume'].mean()

        penalty = 0; reasons = []
        if current_price < price_1y_ago: penalty += 20; reasons.append("Downtrend 1Y")
        if current_price < 60: penalty += 30; reasons.append("Saham Gocap")
        if avg_vol < 50000: penalty += 25; reasons.append("Tidak Likuid")
        
        final_score = max(0, data_short['score'] - penalty)
        hist_data = {"max_1y": max_1y, "min_1y": min_1y, "avg_volume": avg_vol, "note": ", ".join(reasons) if reasons else "Valid"}
        return final_score, hist_data
    except: return data_short['score'], {}

def get_cached_analysis(ticker):
    now = time.time()
    if ticker in CACHE_DATA:
        item = CACHE_DATA[ticker]
        if now - item['timestamp'] < CACHE_TIMEOUT: return item['data']
    data = analisa_multistrategy(ticker)
    if data['last_price'] > 0:
        new_score, hist_data = validasi_histori_panjang(ticker, data)
        data['score'] = int(new_score); data['hist_data'] = hist_data
        CACHE_DATA[ticker] = {'data': data, 'timestamp': now}
    return data

# [FITUR TAMBAHAN] Ambil Data Fundamental Live
def ambil_data_fundamental_live(ticker_lengkap):
    try:
        stock = yf.Ticker(ticker_lengkap)
        info = stock.info
        return {
            "sektor": info.get('sector', 'General'),
            "per": info.get('trailingPE', 0),
            "pbv": info.get('priceToBook', 0),
            "market_cap": info.get('marketCap', 0),
            "roe": info.get('returnOnEquity', 0),
            "text_summary": f"Sektor: {info.get('sector')} | PER: {info.get('trailingPE', 0):.2f}x | PBV: {info.get('priceToBook', 0):.2f}x | ROE: {info.get('returnOnEquity', 0):.2f}"
        }
    except: return {"sektor": "General", "per": 0, "pbv": 0, "market_cap": 0, "roe": 0, "text_summary": "Data Fundamental N/A"}

def ambil_data_live_lengkap(ticker_lengkap):
    try:
        stock = yf.Ticker(ticker_lengkap)
        info = stock.info
        day_open = info.get('open', 0); day_high = info.get('dayHigh', 0); day_low = info.get('dayLow', 0)
        curr_price = info.get('currentPrice', day_open); volume = info.get('volume', 0)
        candle_stat = "🟢 BULLISH" if curr_price > day_open else "🔴 BEARISH"
        return f"- LIVE: Open {day_open} | High {day_high} | Low {day_low} | Last {curr_price} | {candle_stat} | Vol: {volume}"
    except: return "Data Live Tidak Tersedia."

def cek_kondisi_market():
    now = time.time()
    if now - MARKET_STATUS['last_check'] < 900: return MARKET_STATUS['condition']
    try:
        ihsg = yf.Ticker("^JKSE").history(period="2d")
        if len(ihsg) >= 2:
            change = (ihsg['Close'].iloc[-1] - ihsg['Close'].iloc[-2]) / ihsg['Close'].iloc[-2]
            MARKET_STATUS['condition'] = "CRASH" if change < -0.008 else "NORMAL"
        MARKET_STATUS['last_check'] = now
    except: MARKET_STATUS['condition'] = "NORMAL"
    return MARKET_STATUS['condition']

# ==========================================
# 5. LOGIKA PLAN SAKTI (PERHITUNGAN ANGKA)
# ==========================================
def get_tick_size(harga):
    if harga < 200: return 1
    elif harga < 500: return 2
    elif harga < 2000: return 5
    elif harga < 5000: return 10
    else: return 25

def bulatkan_ke_tick(harga):
    if harga <= 0: return 0
    tick = get_tick_size(harga)
    return int(round(harga / tick) * tick)

def get_psychological_step(harga):
    if harga < 200: return 10
    elif harga < 1000: return 50
    elif harga < 5000: return 100
    else: return 250

def format_angka(nilai):
    return "{:,}".format(int(nilai)).replace(",", ".")

def hitung_plan_sakti(data_analisa, ticker_fibo=None):
    harga_sekarang = data_analisa.get('last_price', 0)
    hist_data = data_analisa.get('hist_data', {})
    support_short = data_analisa.get('support', 0)
    if support_short == 0: support_short = int(harga_sekarang * 0.96)
    tipe_trading = data_analisa.get('type', 'UNKNOWN')

    if harga_sekarang <= 0: return "-", 0, "-"
    
    base_support = support_short
    tick_size = get_tick_size(base_support)
    buy_low = bulatkan_ke_tick(base_support + (2 * tick_size))
    buy_high = bulatkan_ke_tick(buy_low + (3 * tick_size))
    
    status_entry = ""
    if harga_sekarang > (buy_high * 1.03): status_entry = "\n⚠️ Harga Lari"
    elif harga_sekarang < buy_low: buy_low = harga_sekarang

    entry_str = f"{format_angka(buy_low)} - {format_angka(buy_high)}{status_entry}"

    sl_raw = base_support - (6 * get_tick_size(base_support))
    sl = bulatkan_ke_tick(sl_raw)

    if tipe_trading == "ARA": 
        tp_str = "HOLD SAMPAI ARA 🚀"
        sl = bulatkan_ke_tick(harga_sekarang * 0.92)
    elif tipe_trading == "INVEST":
        tp_str = "HOLD JANGKA PANJANG"
        sl = bulatkan_ke_tick(harga_sekarang * 0.85)
    else:
        tp1 = bulatkan_ke_tick(buy_low * 1.04)
        max_1y = hist_data.get('max_1y', 0)
        if max_1y > buy_low and max_1y < (buy_low * 1.5): tp2_raw = max_1y
        else: tp2_raw = buy_low * 1.08

        if ticker_fibo:
            try:
                hist = yf.Ticker(ticker_fibo).history(period="1mo")
                if not hist.empty:
                    swing_high = hist['High'].max()
                    swing_low = hist['Low'].min()
                    swing_range = swing_high - swing_low
                    tp_fibo = swing_low + (swing_range * 1.618)
                    if tp_fibo < tp2_raw and tp_fibo > buy_low: tp2_raw = tp_fibo
            except: pass
        
        tp2 = bulatkan_ke_tick(tp2_raw)
        step = get_psychological_step(tp2)
        tp3_raw = (int(tp2 / step) + 1) * step
        if tp3_raw <= tp2: tp3_raw += step
        tp3 = bulatkan_ke_tick(tp3_raw)

        tp_str = (f"🎯 TP1: {format_angka(tp1)}\n"
                  f"🚀 TP2: {format_angka(tp2)}\n"
                  f"💎 TP3: {format_angka(tp3)}")

    return entry_str, sl, tp_str

# ==========================================
# 6. ENDPOINT DETAIL (AGGREGATOR V8 + FAILOVER V9)
# ==========================================
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    ticker_lengkap = ticker_polos + ".JK"
    data = get_cached_analysis(ticker_lengkap)
    if data['last_price'] == 0: return jsonify({"error": "Not Found"})
    
    # --- PENGUMPULAN DATA PENDUKUNG (SUPAYA AI PAHAM) ---
    
    # 1. Data Fundamental (PER/PBV/ROE)
    funda = ambil_data_fundamental_live(ticker_lengkap)
    
    # 2. Data Teknikal Mentah (7+ Indikator) [FITUR V8 - UTUH]
    teknikal_lengkap = hitung_indikator_lengkap(ticker_lengkap)
    
    # 3. Berita & Korelasi Sektoral [FITUR V7 - UTUH]
    list_berita = ambil_berita_saham(ticker_lengkap)
    laporan_berita = agen_pencari_berita_robust(ticker_polos, funda['sektor'], list_berita)

    # 4. Susun Context untuk AI
    data_context = f"""
    SAHAM: {ticker_polos}
    
    [HASIL SCANNER PYTHON]
    - Skor Akhir: {data['score']}/100 
    - Rekomendasi Mesin: {data['verdict']}
    - Catatan Scanner: {data.get('hist_data', {}).get('note', 'Valid')}
    
    {teknikal_lengkap}
    
    [DATA FUNDAMENTAL LIVE]
    {funda['text_summary']}
    - Market Cap: {funda['market_cap']:,}
    
    [BERITA & SENTIMEN]
    {laporan_berita}
    """
    
    # 5. Analisa AI (DENGAN FAILOVER V9)
    analisa_final = agen_analis_utama(data_context)
    entry, sl, tp = hitung_plan_sakti(data)

    # 6. Formatting Output
    reason_final = f"📊 **SKOR SCANNER: {data['score']} ({data['verdict']})**\n"
    if "Valid" not in data.get('hist_data', {}).get('note', 'Valid'):
        reason_final += f"⚠️ **Scanner Note:** {data['hist_data']['note']}\n"
        
    reason_final += f"\n📰 **SENTIMEN BERITA:**\n{laporan_berita}\n\n====================\n🧠 **ANALISA AI (DATA-DRIVEN):**\n{analisa_final}"

    pct = data.get('change_pct', 0)
    tanda = "+" if pct >= 0 else ""

    return jsonify({
        "ticker": ticker_polos,
        "company_name": f"Rp {format_angka(data['last_price'])} ({tanda}{pct:.2f}%)",
        "badges": { "syariah": ticker_polos in DATABASE_SYARIAH, "lq45": True },
        "analysis": {
            "score": int(data['score']),
            "verdict": data['verdict'],
            "reason": reason_final, 
            "type": data['type']
        },
        "plan": { "entry": entry, "stop_loss": sl, "take_profit": tp },
        "news": list_berita, 
        "is_watchlist": ticker_polos in WATCHLIST
    })

# ==========================================
# 7. SCANNER
# ==========================================
def process_single_stock(kode, target_strategy, min_score_needed):
    try:
        ticker = kode + ".JK"
        data = get_cached_analysis(ticker)
        if data['last_price'] == 0 or data['score'] < min_score_needed: return None
        
        tipe_ditemukan = data['type']
        if target_strategy != 'ALL' and target_strategy != 'WATCHLIST' and target_strategy != 'SYARIAH':
            if target_strategy not in tipe_ditemukan: return None

        entry, sl, tp = hitung_plan_sakti(data)
        return {
            "ticker": kode,
            "company_name": f"Rp {format_angka(data['last_price'])}",
            "badges": { "syariah": kode in DATABASE_SYARIAH, "lq45": True },
            "analysis": {
                "score": int(data['score']),
                "verdict": data['verdict'],
                "reason": data['reason'],
                "type": tipe_ditemukan
            },
            "plan": {"entry": entry, "stop_loss": sl, "take_profit": tp},
            "news": [], "is_watchlist": kode in WATCHLIST
        }
    except: return None

@app.route('/api/scan-results', methods=['GET'])
def get_scan_results():
    target_strategy = request.args.get('strategy', 'ALL') 
    kondisi_market = cek_kondisi_market()
    MIN_SCORE = 60 if kondisi_market == "NORMAL" else 80
    
    daftar_scan = DATABASE_SYARIAH if target_strategy == 'SYARIAH' else WATCHLIST if target_strategy in ['ALL', 'WATCHLIST'] else MARKET_UNIVERSE
    limit_scan = daftar_scan[:60]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_stock, kode, target_strategy, MIN_SCORE): kode for kode in limit_scan}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: results.append(res)
    results.sort(key=lambda x: x['analysis']['score'], reverse=True)
    return jsonify(results)

@app.route('/api/watchlist/add', methods=['POST'])
def add_watchlist():
    ticker = request.args.get('ticker')
    if ticker and ticker not in WATCHLIST: WATCHLIST.append(ticker)
    return jsonify({"message": "Success", "current_list": WATCHLIST})

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_watchlist():
    ticker = request.args.get('ticker')
    if ticker and ticker in WATCHLIST: WATCHLIST.remove(ticker)
    return jsonify({"message": "Success", "current_list": WATCHLIST})

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "Server Alpha Hunter V10 (Merged V8+V9) ONLINE 🚀"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Alpha Hunter V10 Server Running on Port: {port}")
    app.run(host='0.0.0.0', port=port)
