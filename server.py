import os
import time
import concurrent.futures
import yfinance as yf
import math
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# --- IMPORT LIBRARY AI ---
from google import genai
from google.genai import types # Untuk Google Search
from groq import Groq 
from openai import OpenAI 

load_dotenv()

# Pastikan file rumus_saham.py ada di folder yang sama (V5 Sniper)
from rumus_saham import analisa_multistrategy, ambil_berita_saham 

app = Flask(__name__)

# ==========================================
# 0. KONFIGURASI AI
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

# ==========================================
# 1. BAGIAN 1: GEMINI SEBAGAI "WARTAWAN" (PENCARI FAKTA)
# ==========================================
def agen_pencari_berita_gemini(ticker):
    """
    Tugas: Browsing internet cari Corporate Action & Isu Terkini.
    Output: Ringkasan Fakta (Bukan Analisa).
    """
    if not client_gemini:
        return "Gemini tidak aktif. Mengandalkan berita Yahoo."

    try:
        print(f"🌍 Gemini sedang browsing info tentang {ticker}...")
        prompt_news = f"""
        TUGAS KHUSUS: Cari berita TERBARU dan VALID mengenai saham {ticker} (Indonesia).
        Gunakan Google Search.
        
        FOKUS PADA:
        1. Corporate Action Terbaru/Akan Datang (Dividen, RUPS, Right Issue, Merger, Akuisisi, Buyback).
        2. Isu Vital (Kasus Hukum, PKPU, Laporan Keuangan Rilis, Proyek Besar).
        
        OUTPUT:
        Buatlah RINGKASAN PADAT (maksimal 3 poin penting). 
        Jika tidak ada berita penting dalam 1 minggu terakhir, katakan "NIL".
        JANGAN BERIKAN ANALISA/REKOMENDASI. HANYA FAKTA.
        """
        
        response = client_gemini.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt_news,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    google_search_retrieval=types.GoogleSearchRetrieval
                )]
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return "Gagal mengambil berita live (Limit/Error)."

# ==========================================
# 2. BAGIAN 2: GROQ/DEEPSEEK SEBAGAI "KEPALA ANALIS"
# ==========================================
def agen_analis_utama(data_context):
    """
    Tugas: Menerima data teknikal + Laporan Gemini, lalu memutuskan strategi.
    """
    prompt_analis = f"""
    Kamu adalah Fund Manager Senior / Kepala Analis Saham.
    
    BERIKUT ADALAH LAPORAN LENGKAP:
    {data_context}
    
    TUGASMU ADALAH MENGAMBIL KEPUTUSAN FINAL (DECISION MAKER):
    
    1. 🧠 **Sintesa Data**: Hubungkan Fakta Berita (dari Laporan Lapangan) dengan Data Teknikal & Bandar (Volume/Flow). Apakah beritanya mendukung kenaikan harga?
    2. 🛡️ **Cek Validitas**: Apakah kenaikan harga ini didukung Fundamental (Valuasi) atau cuma gorengan semata?
    3. 🎯 **Penentuan Strategi**: Tentukan gaya trading (BPJS/BSJP/SWING/INVEST/HINDARI).
    4. 🔢 **PLAN EKSEKUSI**: Tentukan Entry Price & Target Price yang realistis sesuai volatilitas hari ini.
    5. ⚖️ **VERDICT**: (STRONG BUY / BUY / WAIT / SELL).
    
    Jawab dengan gaya bahasa profesional tapi tajam.
    """

    # Prioritas 1: Groq (Llama 3 - Cepat & Logis)
    if client_groq:
        try:
            print("⚡ Groq sedang menganalisa data gabungan...")
            chat = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt_analis}],
                model="llama-3.3-70b-versatile",
            )
            return chat.choices[0].message.content.strip()
        except Exception as e: print(f"⚠️ Groq Error: {e}")

    # Prioritas 2: DeepSeek (Reasoning Kuat)
    if client_deepseek:
        try:
            print("🧠 DeepSeek sedang berpikir dalam...")
            res = client_deepseek.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": prompt_analis}]
            )
            return res.choices[0].message.content.strip()
        except: pass
    
    # Emergency Backup: Balik ke Gemini kalau 2 AI diatas mati (Jarang terjadi)
    if client_gemini:
        try:
            return client_gemini.models.generate_content(
                model='gemini-1.5-flash', contents=prompt_analis
            ).text.strip()
        except: pass

    return "Maaf, semua Analis AI sedang sibuk."

# ==========================================
# 3. DATABASE & CACHE
# ==========================================
CACHE_DATA = {}
CACHE_TIMEOUT = 300 
MARKET_STATUS = {"condition": "NORMAL", "last_check": 0}

DATABASE_SYARIAH = [
    "ADRO", "AKRA", "ANTM", "ASII", "BRIS", "BRPT", "BUKA", "CPIN", 
    "EMTK", "EXCL", "GOTO", "HRUM", "ICBP", "INCO", "INDF", "INKP", 
    "INTP", "ISAT", "ITMG", "JPFA", "KLBF", "MDKA", "MEDC", "PGAS", 
    "PTBA", "SCMA", "SIDO", "SMGR", "TPIA", "UNTR", "UNVR",
    "ACES", "ADHI", "ADMR", "AGII", "AMMN", "AMRT", "ASSA", "AUTO", 
    "AVIA", "BIRD", "BISI", "BLESS", "BMTR", "BTPS", "BUMI", "BYAN", 
    "CINT", "CLEO", "CMRY", "CTRA", "DEWAN", "DOID", "DRMA", "ELSA", 
    "ENRG", "ERAA", "ESSA", "FREN", "HEAL", "HOKI", "HMSP", "INDY", 
    "INRA", "JSMR", "KAEF", "KRYA", "LSIP", "MAPI", "MAPA", "MBMA", 
    "MCOL", "MIKA", "MNCN", "MPPA", "MTEL", "MYOR", "NCKL", "NICL", 
    "PANI", "PGEO", "PNBN", "PNLF", "PTPP", "PWON", "RAJA", "RALS", 
    "RMKE", "ROTI", "SAME", "SCCO", "SILO", "SIMP", "SMDR", "SMRA", 
    "SMSM", "SRTG", "SSIA", "STAA", "TAPG", "TBIG", "TINS", "TKIM", 
    "TLKM", "TOWR", "TRIN", "TSPC", "ULTJ", "WIKA", "WIIM", "WOOD"
]
MARKET_UNIVERSE = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "INDF", "GOTO", "MDKA", "ANTM", "INCO", "PGAS", "ADRO", "PTBA", "BRPT", "BREN", "AMMN"]
WATCHLIST = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "GOTO", "ANTM", "ADRO", "UNTR"]

# FUNGSI VALIDASI HISTORI (V5 SNIPER)
def validasi_histori_panjang(ticker_lengkap, data_short):
    try:
        hist = yf.Ticker(ticker_lengkap).history(period="1y")
        if hist.empty: return 0, {} 

        current_price = data_short['last_price']
        max_1y = hist['High'].max()
        min_1y = hist['Low'].min()
        avg_vol = hist['Volume'].mean()
        price_1y_ago = hist['Close'].iloc[0]

        penalty = 0
        alasan_penalti = []

        if current_price < price_1y_ago:
            penalty += 20
            alasan_penalti.append("Downtrend Jangka Panjang")
        if current_price < 60:
            penalty += 30
            alasan_penalti.append("Saham Gocap")
        if avg_vol < 50000:
            penalty += 25
            alasan_penalti.append("Tidak Likuid")
        
        range_1y = max_1y - min_1y
        posisi_thd_max = (current_price - min_1y) / range_1y if range_1y > 0 else 0
        if posisi_thd_max > 0.95:
            penalty += 10
            alasan_penalti.append("Resisten Tahunan")

        final_score = max(0, data_short['score'] - penalty)
        
        hist_data = {
            "max_1y": max_1y,
            "min_1y": min_1y,
            "trend_1y": "UP" if current_price > price_1y_ago else "DOWN",
            "avg_volume": avg_vol,
            "note": ", ".join(alasan_penalti) if alasan_penalti else "Valid"
        }
        return final_score, hist_data
    except: return data_short['score'], {}

def get_cached_analysis(ticker):
    now = time.time()
    if ticker in CACHE_DATA:
        item = CACHE_DATA[ticker]
        if now - item['timestamp'] < CACHE_TIMEOUT:
            return item['data']
    
    data = analisa_multistrategy(ticker)
    if data['last_price'] > 0:
        new_score, hist_data = validasi_histori_panjang(ticker, data)
        data['score'] = int(new_score)
        data['hist_data'] = hist_data
        CACHE_DATA[ticker] = {'data': data, 'timestamp': now}
    return data

def ambil_data_live_lengkap(ticker_lengkap):
    try:
        stock = yf.Ticker(ticker_lengkap)
        info = stock.info
        per = info.get('trailingPE', 0)
        pbv = info.get('priceToBook', 0)
        sector = info.get('sector', 'Unknown')
        day_open = info.get('open', 0)
        day_high = info.get('dayHigh', 0)
        day_low = info.get('dayLow', 0)
        curr_price = info.get('currentPrice', day_open)
        volume = info.get('volume', 0)
        candle_stat = "🟢 BULLISH" if curr_price > day_open else "🔴 BEARISH"
        return f"""
        - Sektor: {sector} | PER: {per:.2f}x | PBV: {pbv:.2f}x
        - LIVE: Open {day_open} | High {day_high} | Low {day_low} | Last {curr_price}
        - Candle: {candle_stat} | Vol Hari Ini: {volume}
        """
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
# 4. LOGIKA PLAN SAKTI (V5)
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
# 5. ENDPOINT DETAIL (ESTAFET AI)
# ==========================================
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    ticker_lengkap = ticker_polos + ".JK"
    data = get_cached_analysis(ticker_lengkap)
    
    if data['last_price'] == 0:
        return jsonify({"error": "Not Found", "analysis": {"score":0, "verdict":"ERR", "reason":"-", "type":"-"}})
    
    # 1. Ambil Data Teknikal & Live
    info_live = ambil_data_live_lengkap(ticker_lengkap)
    hist_data = data.get('hist_data', {})
    entry, sl, tp = hitung_plan_sakti(data, ticker_fibo=ticker_lengkap)

    score = data['score']
    verdict = data['verdict']
    catatan_histori = hist_data.get('note', 'Valid')
    trend_1y = hist_data.get('trend_1y', 'N/A')
    
    # 2. Ambil Berita Yahoo (Backup)
    list_berita = ambil_berita_saham(ticker_lengkap)
    headlines_yahoo = [b['title'] for b in list_berita[:3]] 
    teks_yahoo = "\n- ".join(headlines_yahoo) if headlines_yahoo else "-"

    # 3. [LANGKAH 1] SURUH GEMINI CARI FAKTA BARU
    laporan_fakta_gemini = agen_pencari_berita_gemini(ticker_polos)

    # 4. [LANGKAH 2] KIRIM SEMUA DATA KE GROQ/DEEPSEEK
    data_context = f"""
    SAHAM: {ticker_polos}
    
    [DATA TEKNIKAL SYSTEM]
    - Skor: {score}/100 | Trend 1Y: {trend_1y}
    - Warning: {catatan_histori}
    {info_live}
    
    [DATA BERITA]
    - Dari Yahoo: {teks_yahoo}
    - LAPORAN WARTAWAN LAPANGAN (GEMINI):
      "{laporan_fakta_gemini}"
    """
    
    analisa_final = agen_analis_utama(data_context)
    
    rincian_teknikal = f"🔍 **SKOR {score} ({verdict})**\n"
    if catatan_histori != "Valid": rincian_teknikal += f"⚠️ {catatan_histori}\n"

    # Gabungkan Laporan Fakta + Analisa Final
    reason_final = f"{rincian_teknikal}\n\n🌍 **LAPORAN FAKTA (GEMINI):**\n{laporan_fakta_gemini}\n\n====================\n🧠 **ANALISA FINAL:**\n{analisa_final}"
    
    pct = data.get('change_pct', 0)
    tanda = "+" if pct >= 0 else ""

    stock_detail = {
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
    }
    return jsonify(stock_detail)

# ==========================================
# 6. SCANNER & WATCHLIST
# ==========================================
def process_single_stock(kode, target_strategy, min_score_needed):
    try:
        ticker = kode + ".JK"
        data = get_cached_analysis(ticker)
        if data['last_price'] == 0: return None
        if data['score'] < min_score_needed: return None 

        tipe_ditemukan = data['type']
        if target_strategy == 'SYARIAH': pass 
        elif target_strategy not in ['ALL', 'WATCHLIST']:
            if target_strategy not in tipe_ditemukan: return None

        entry, sl, tp = hitung_plan_sakti(data, ticker_fibo=None)
        pct = data.get('change_pct', 0)
        tanda = "+" if pct >= 0 else ""
        info_harga = f"Rp {format_angka(data['last_price'])} ({tanda}{pct:.2f}%)"

        return {
            "ticker": kode,
            "company_name": info_harga,
            "badges": { "syariah": kode in DATABASE_SYARIAH, "lq45": True },
            "analysis": {
                "score": int(data['score']),
                "verdict": data['verdict'],
                "reason": data['reason'],
                "type": tipe_ditemukan
            },
            "plan": {"entry": entry, "stop_loss": sl, "take_profit": tp},
            "news": [], 
            "is_watchlist": kode in WATCHLIST
        }
    except: return None

@app.route('/api/scan-results', methods=['GET'])
def get_scan_results():
    target_strategy = request.args.get('strategy', 'ALL') 
    kondisi_market = cek_kondisi_market()
    MIN_SCORE = 60 
    if kondisi_market == "CRASH": MIN_SCORE = 80 
    
    daftar_scan = []
    if target_strategy == 'SYARIAH': daftar_scan = DATABASE_SYARIAH 
    elif target_strategy == 'ALL' or target_strategy == 'WATCHLIST': daftar_scan = WATCHLIST
    else: daftar_scan = MARKET_UNIVERSE

    results = []
    limit_scan = daftar_scan[:60] 
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Alpha Hunter V3 Server berjalan di Port: {port}")
    app.run(host='0.0.0.0', port=port)
