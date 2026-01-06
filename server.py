import os
import time
import concurrent.futures
import yfinance as yf
import math
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# --- IMPORT LIBRARY AI & SEARCH ---
# Dibungkus try-except agar server 'Tahan Banting' kalau library belum update
try:
    from duckduckgo_search import DDGS 
except ImportError:
    DDGS = None
    print("⚠️ Warning: duckduckgo_search belum terinstall. Fitur browsing internet mati.")

from groq import Groq 
from openai import OpenAI 

try:
    from google import genai
except ImportError:
    genai = None

load_dotenv()

# Pastikan file rumus_saham.py ada di folder yang sama (V5 Sniper)
from rumus_saham import analisa_multistrategy, ambil_berita_saham 

app = Flask(__name__)

# ==========================================
# 0. KONFIGURASI AI CLIENT
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None
client_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY and genai else None

# ==========================================
# 1. FITUR BARU: SECTOR & COMMODITY INTELLIGENCE
# ==========================================
def dapatkan_keywords_cerdas(ticker, sektor):
    """
    [FITUR TAMBAHAN]
    Membuat query pencarian yang pintar. 
    Tidak hanya mencari nama saham, tapi juga 'Underlying Asset'-nya.
    """
    sektor = sektor.upper() if sektor else "GENERAL"
    query = f"berita saham {ticker} indonesia terbaru hari ini corporate action"
    
    # Logika Korelasi Pasar Global & Komoditas
    if any(x in sektor for x in ["GOLD", "MINING", "METAL", "BASIC MAT"]):
        query += " + harga emas nikel tembaga dunia hari ini"
    elif any(x in sektor for x in ["OIL", "GAS", "ENERGY"]):
        query += " + harga minyak mentah dunia brent wti"
    elif "COAL" in sektor or ticker in ["ADRO", "PTBA", "ITMG", "HRUM", "BUMI"]:
        query += " + harga batubara Newcastle ICE terbaru"
    elif "BANK" in sektor or "FINANCE" in sektor:
        query += " + suku bunga BI Rate The Fed inflasi"
    elif any(x in sektor for x in ["PALM", "AGRI", "PLANTATION"]):
        query += " + harga CPO crude palm oil malaysia"
    elif "TECH" in sektor or ticker in ["GOTO", "BUKA", "EMTK"]:
        query += " + sentimen saham teknologi global nasdaq"
    elif "CONSUMER" in sektor:
        query += " + daya beli masyarakat inflasi rupiah"
        
    return query

# ==========================================
# 2. AGEN PENCARI BERITA (HYBRID + CERDAS)
# ==========================================
def agen_pencari_berita_robust(ticker, sektor, berita_yahoo_backup):
    """
    Strategi Berlapis:
    1. Pakai Keyword Cerdas untuk DuckDuckGo (Internet).
    2. Kalau Gagal/Kosong, pakai data Yahoo Finance.
    3. Dirangkum oleh Groq.
    """
    laporan_mentah = ""
    sumber_data = "YAHOO (BACKUP)" # Default

    # --- LAYER 1: DUCKDUCKGO (INTERNET + KOMODITAS) ---
    if DDGS:
        try:
            # Gunakan keyword cerdas (Saham + Komoditas terkait)
            query_smart = dapatkan_keywords_cerdas(ticker, sektor)
            print(f"🌍 Searching: {query_smart}...")
            
            results = DDGS().text(query_smart, max_results=5)
            
            if results:
                ddg_text = []
                for r in results:
                    ddg_text.append(f"- {r['title']}: {r['body']}")
                laporan_mentah = "\n".join(ddg_text)
                sumber_data = "INTERNET (REAL-TIME + SEKTORAL)"
            else:
                print("⚠️ DuckDuckGo: Hasil Kosong.")
        except Exception as e:
            print(f"⚠️ DuckDuckGo Error: {e}")

    # --- LAYER 2: FALLBACK KE YAHOO ---
    if not laporan_mentah or len(laporan_mentah) < 50:
        print("🔄 Mengalihkan ke Data Yahoo Finance...")
        yahoo_text = []
        if berita_yahoo_backup:
            for b in berita_yahoo_backup:
                judul = b.get('title', '')
                yahoo_text.append(f"- {judul}")
            laporan_mentah = "\n".join(yahoo_text)
        else:
            laporan_mentah = "Data berita spesifik tidak ditemukan."

    # --- LAYER 3: RANGKUMAN WARTAWAN AI ---
    if not client_groq:
        return f"[{sumber_data}] {laporan_mentah}"

    prompt_wartawan = f"""
    Kamu adalah Reporter Pasar Modal Senior.
    
    DATA MENTAH ({ticker} - Sektor {sektor}):
    {laporan_mentah}
    
    TUGAS:
    1. Cari info langsung tentang {ticker} (Laba, Dividen, Masalah Hukum).
    2. Cari info TIDAK LANGSUNG (Harga Komoditas/Suku Bunga) yang ada di teks.
    3. Rangkum maksimal 4 poin fakta penting. Jika data tidak relevan, tulis "NIL".
    """
    
    try:
        chat = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt_wartawan}],
            model="llama-3.3-70b-versatile",
        )
        return f"**SUMBER: {sumber_data}**\n" + chat.choices[0].message.content.strip()
    except:
        return f"[{sumber_data} - Raw]\n{laporan_mentah}"

# ==========================================
# 3. AGEN KEPALA ANALIS (ANALISA 6 POIN PRO)
# ==========================================
def agen_analis_utama(data_context):
    """
    Otak Utama: Menggabungkan Teknikal + Fundamental + Berita Sektoral.
    """
    prompt_analis = f"""
    Kamu adalah Fund Manager Senior (Institusi) dan Trader Profesional.
    
    DATA LENGKAP SAHAM:
    {data_context}
    
    WAJIB BERIKAN ANALISA TAJAM (6 POIN):
    
    1. 🌍 **Sintesa Berita & Korelasi Sektoral**
       (Jelaskan hubungan berita/komoditas dengan harga saham. Misal: Emas naik -> ANTM naik? Atau justru anomali? Hati-hati jebakan news.)
       
    2. 🕵️‍♂️ **Bandarmologi (Flow Analysis)**
       (Analisis Volume & Smart Money. Apakah Bandar sedang Akumulasi diam-diam atau Distribusi di pucuk?)
       
    3. 📊 **Valuasi & Fundamental**
       (Cek data PER/PBV yang tersedia. Apakah harga sekarang Murah (Undervalued) atau Mahal (Gorengan)?)
       
    4. ⏱️ **Timing & Tren Technical**
       (Lihat Candle & Trend 1 tahun. Apakah momentum 'Buy on Weakness' atau 'Buy on Breakout'?)

    5. 🎯 **Strategi Trading**
       - Pilih: (SCALPING / SWING / INVEST / BPJS / BSJP / HINDARI).
       - Tentukan Entry Price & Target Price (Take Profit) yang logis.
       
    6. ⚖️ **VERDICT FINAL**
       (Kesimpulan: STRONG BUY / BUY / WAIT / SELL).
    
    Jawab dengan bahasa trader Indonesia yang lugas, objektif, dan tanpa basa-basi.
    """

    # Prioritas 1: DeepSeek
    if client_deepseek:
        try:
            res = client_deepseek.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": prompt_analis}]
            )
            return res.choices[0].message.content.strip()
        except: pass
    
    # Prioritas 2: Groq
    if client_groq:
        try:
            chat = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt_analis}],
                model="llama-3.3-70b-versatile",
            )
            return chat.choices[0].message.content.strip()
        except: pass

    # Prioritas 3: Gemini
    if client_gemini:
        try:
            return client_gemini.models.generate_content(
                model='gemini-1.5-flash', contents=prompt_analis
            ).text.strip()
        except: pass
            
    return "Maaf, Analisa AI tidak tersedia saat ini."

# ==========================================
# 4. DATABASE, CACHE & UTILS
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

def validasi_histori_panjang(ticker_lengkap, data_short):
    try:
        hist = yf.Ticker(ticker_lengkap).history(period="1y")
        if hist.empty: return 0, {} 
        current_price = data_short['last_price']
        price_1y_ago = hist['Close'].iloc[0]
        max_1y = hist['High'].max()
        min_1y = hist['Low'].min()
        avg_vol = hist['Volume'].mean()

        penalty = 0
        alasan_penalti = []
        if current_price < price_1y_ago: penalty += 20; alasan_penalti.append("Downtrend 1Y")
        if current_price < 60: penalty += 30; alasan_penalti.append("Saham Gocap")
        if avg_vol < 50000: penalty += 25; alasan_penalti.append("Tidak Likuid")
        
        final_score = max(0, data_short['score'] - penalty)
        hist_data = {"max_1y": max_1y, "min_1y": min_1y, "avg_volume": avg_vol, "note": ", ".join(alasan_penalti) if alasan_penalti else "Valid"}
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
        data['score'] = int(new_score)
        data['hist_data'] = hist_data
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
            "text_summary": f"Sektor: {info.get('sector')} | PER: {info.get('trailingPE', 0):.2f}x | PBV: {info.get('priceToBook', 0):.2f}x"
        }
    except: return {"sektor": "General", "per": 0, "pbv": 0, "market_cap": 0, "text_summary": "Data Fundamental N/A"}

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
# 6. ENDPOINT DETAIL (CORE LOGIC UTAMA)
# ==========================================
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    ticker_lengkap = ticker_polos + ".JK"
    data = get_cached_analysis(ticker_lengkap)
    
    if data['last_price'] == 0:
        return jsonify({"error": "Not Found", "analysis": {"score":0, "verdict":"ERR", "reason":"-", "type":"-"}})
    
    # 1. Ambil Data Teknikal Live
    info_live = ambil_data_live_lengkap(ticker_lengkap)
    hist_data = data.get('hist_data', {})
    entry, sl, tp = hitung_plan_sakti(data, ticker_fibo=ticker_lengkap)

    # 2. [TAMBAHAN] Ambil Data Fundamental Live
    funda = ambil_data_fundamental_live(ticker_lengkap)

    score = data['score']
    verdict = data['verdict']
    catatan_histori = hist_data.get('note', 'Valid')
    trend_1y = hist_data.get('trend_1y', 'N/A')
    
    # 3. Ambil Berita Yahoo (Backup)
    list_berita = ambil_berita_saham(ticker_lengkap)
    
    # 4. Cari Berita Internet (Hybrid + Korelasi Sektoral)
    # Kita kirim 'sektor' agar pencarian lebih cerdas (Misal: Sektor Coal -> Cari harga batubara)
    laporan_fakta_lengkap = agen_pencari_berita_robust(ticker_polos, funda['sektor'], list_berita)

    # 5. Analisa Final oleh Kepala Analis
    data_context = f"""
    SAHAM: {ticker_polos}
    
    [DATA TEKNIKAL SYSTEM]
    - Skor: {score}/100 | Trend 1Y: {trend_1y}
    - Warning: {catatan_histori}
    {info_live}
    
    [DATA FUNDAMENTAL LIVE]
    {funda['text_summary']}
    - Market Cap: {funda['market_cap']:,}
    
    [LAPORAN BERITA & KORELASI]
    {laporan_fakta_lengkap}
    """
    
    analisa_final = agen_analis_utama(data_context)
    
    rincian_teknikal = f"🔍 **SKOR {score} ({verdict})**\n"
    if catatan_histori != "Valid": rincian_teknikal += f"⚠️ {catatan_histori}\n"

    reason_final = f"{rincian_teknikal}\n\n📰 **SENTIMEN & KORELASI:**\n{laporan_fakta_lengkap}\n\n====================\n🧠 **ANALISA FUND MANAGER:**\n{analisa_final}"
    
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
# 7. SCANNER & WATCHLIST (TETAP SAMA)
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

# HALAMAN DEPAN
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "Server Alpha Hunter V7 PRO ONLINE 🚀",
        "features": {
            "search": "Hybrid (DDG + Yahoo + Sectoral)",
            "analysis": "DeepSeek / Groq (6 Points)",
            "scanner": "V5 Sniper"
        },
        "message": "Gunakan endpoint /api/stock-detail?ticker=BBRI"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Alpha Hunter V7 Server berjalan di Port: {port}")
    app.run(host='0.0.0.0', port=port)
