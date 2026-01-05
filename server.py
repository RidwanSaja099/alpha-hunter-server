import os
from flask import Flask, jsonify, request
import concurrent.futures
import time
import yfinance as yf
from google import genai 
from dotenv import load_dotenv 

# Load environment variables dari file .env (jika ada di laptop)
load_dotenv()

# Pastikan file rumus_saham.py ada di folder yang sama
from rumus_saham import analisa_multistrategy, ambil_berita_saham 

app = Flask(__name__)

# ==========================================
# 0. KONFIGURASI AI GEMINI (LOGIKA AMAN & FALLBACK)
# ==========================================

# Prioritas: 
# 1. Environment Variable (Saat di Railway)
# 2. File .env (Saat di Laptop)
# 3. String Hardcoded (Cadangan terakhir, HAPUS ini jika ingin upload ke GitHub public)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAOcwQyQOkVnM0DyFPsBvS0PaQpoUvLGRo") 

# Inisialisasi Client
client = None
try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini Client Berhasil Diinisialisasi (Model: gemini-flash-latest)")
    else:
        print("⚠️ API Key Kosong. Fitur AI tidak akan jalan.")
except Exception as e:
    print(f"⚠️ Gagal Init Gemini: {e}")

CACHE_DATA = {}
CACHE_TIMEOUT = 300  # 5 Menit
MARKET_STATUS = {"condition": "NORMAL", "last_check": 0}

def get_cached_analysis(ticker):
    now = time.time()
    if ticker in CACHE_DATA:
        item = CACHE_DATA[ticker]
        if now - item['timestamp'] < CACHE_TIMEOUT:
            return item['data']
    
    data = analisa_multistrategy(ticker)
    if data['last_price'] > 0: 
        CACHE_DATA[ticker] = {'data': data, 'timestamp': now}
    return data

# [UPDATE PENTING] PROMPT LEBIH CERDAS (MENJELASKAN "KENAPA")
def analisa_dengan_gemini(ticker, data_teknikal, berita_list):
    if not client: return "⚠️ API Key belum diisi / Error Client."

    try:
        # 1. Siapkan Ringkasan Berita
        judul_berita = [b['title'] for b in berita_list[:3]] 
        teks_berita = "\n- ".join(judul_berita) if judul_berita else "Tidak ada berita spesifik hari ini."

        # 2. Buat Prompt YANG LEBIH TAJAM & ANALITIS (CAUSALITY)
        prompt = f"""
        Bertindaklah sebagai Pakar Analis Pasar Modal Indonesia (IDX) yang Kritis.
        Analisa Saham: {ticker}.
        
        DATA SISTEM KAMI:
        - Harga: {data_teknikal['last_price']}
        - Sinyal Teknikal: {data_teknikal['verdict']}
        - Skor Kekuatan: {data_teknikal['score']}/100
        - Faktor Teknikal: {data_teknikal['reason']}
        
        BERITA TERAKHIR:
        {teks_berita}
        
        TUGAS UTAMA (ANALISA CAUSALITY/SEBAB-AKIBAT):
        Jelaskan *MENGAPA* saham ini bergerak {data_teknikal['verdict']}? 
        Jangan hanya mengulang data di atas. Gunakan logikamu untuk menghubungkan berita/sektor dengan teknikal.
        
        Poin Analisa yang wajib ada:
        1. Apa pemicu utamanya? (Kinerja Keuangan? Harga Komoditas? Atau sekadar pantulan teknikal?)
        2. Analisis Risiko: Apa yang harus diwaspadai trader besok?
        
        Jawab dalam 1 paragraf yang padat (maksimal 3-4 kalimat), tajam, dan berwawasan luas. Awali dengan emoji 🧠.
        """

        # 3. Kirim ke AI
        response = client.models.generate_content(
            model='gemini-flash-latest', 
            contents=prompt
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "🤖 AI sedang istirahat (Limit/Error)."

# FUNGSI MARKET SENTINEL
def cek_kondisi_market():
    now = time.time()
    if now - MARKET_STATUS['last_check'] < 900: 
        return MARKET_STATUS['condition']

    try:
        ihsg = yf.Ticker("^JKSE")
        hist = ihsg.history(period="2d")
        if len(hist) >= 2:
            close_now = hist['Close'].iloc[-1]
            close_prev = hist['Close'].iloc[-2]
            change_pct = (close_now - close_prev) / close_prev
            
            if change_pct < -0.008:
                MARKET_STATUS['condition'] = "CRASH"
            else:
                MARKET_STATUS['condition'] = "NORMAL"
        
        MARKET_STATUS['last_check'] = now
    except:
        MARKET_STATUS['condition'] = "NORMAL"
    
    return MARKET_STATUS['condition']

# ==========================================
# 1. DATABASE SAHAM
# ==========================================
DATABASE_SYARIAH = [
    "ACES", "ADRO", "AKRA", "ANTM", "ASII", "ASRI", "AUTO", "BBHI", "BRIS", 
    "BRMS", "BRPT", "BSDE", "BTPS", "BUMI", "CPIN", "CTRA", "CUAN", "DEWA", 
    "DOID", "ELSA", "EMTK", "ENRG", "ERAA", "EXCL", "GOTO", "HRUM", "HATM", 
    "HEAL", "ICBP", "INCO", "INDF", "INKP", "INTP", "ISAT", "ITMG", 
    "JPFA", "JRPT", "JSMR", "KLBF", "MAPI", "MBMA", "MDKA", "MEDC", "MIKA", 
    "MNCN", "MTEL", "MYOR", "NCKL", "PGAS", "PGEO", "PTBA", "PTPP", "PWON", 
    "RALS", "SCMA", "SIDO", "SMGR", "SMRA", "SRTG", "TAPG", "TBIG", "TINS", 
    "TKIM", "TLKM", "TOWR", "TPIA", "UNTR", "UNVR", "WIKA", "WOOD", "AMMN", 
    "BREN", "PANI", "AMRT", "AVIA", "CMRY", "MAPA", "BELI", "HILL", "BBRI" 
]

MARKET_UNIVERSE = [
    "BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "INDF", "GOTO",
    "ARTO", "BUKA", "EMTK", "MDKA", "ANTM", "INCO", "PGAS", "ADRO", "PTBA", "ITMG",
    "UNVR", "HMSP", "GGRM", "CPIN", "JPFA", "KLBF", "KALBE", "SMGR", "INTP", "BRPT",
    "TPIA", "BREN", "AMMN", "CUAN", "DEWA", "BUMI", "BRMS", "PSAB", "MEDC", "AKRA",
    "EXCL", "ISAT", "JSMR", "ACES", "MAPI", "PWON", "BSDE", "CTRA", "SMRA", "ASRI",
    "BRIS", "HRUM", "INKP", "TKIM", "PANI", "AMRT", "ESSA", "MNC", "MNCN"
]

WATCHLIST = ["BBRI", "BBCA", "GOTO", "ANTM", "BREN", "TLKM"] 

@app.route('/')
def home():
    kondisi = cek_kondisi_market()
    return f"Server Alpha Hunter V3 (Gemini Smart Causality): READY | Market: {kondisi}"

# ==========================================
# 2. LOGIKA HITUNGAN PLAN
# ==========================================
def format_angka(nilai):
    return "{:,}".format(int(nilai)).replace(",", ".")

def hitung_plan_sakti(data_analisa):
    harga_sekarang = data_analisa.get('last_price', 0)
    harga_support = data_analisa.get('support', 0)
    tipe_trading = data_analisa.get('type', 'UNKNOWN')
    
    sl_rumus = data_analisa.get('stop_loss', 0)
    tp_rumus = data_analisa.get('target_price', 0)

    if harga_sekarang <= 0: return "-", 0, "-"
    
    base_entry = harga_support if harga_support > 0 else harga_sekarang
    buy_low = base_entry
    buy_high = int(base_entry * 1.015) 
    
    status_entry = ""
    if harga_sekarang > (buy_high * 1.01): status_entry = "\n(Wait Pullback)"
    entry_str = f"{format_angka(buy_low)} - {format_angka(buy_high)}{status_entry}"
    
    sl = sl_rumus if sl_rumus > 0 else int(base_entry * 0.95)
    
    if tipe_trading == "ARA": 
        tp_str = "HOLD SAMPAI ARA 🚀"
        sl = int(base_entry * 0.92)
    elif tipe_trading == "INVEST":
        tp_str = "HOLD JANGKA PANJANG"
        sl = int(base_entry * 0.85)
    else:
        if tp_rumus > 0:
            tp1 = tp_rumus
            tp2 = tp_rumus + (tp_rumus - base_entry) 
            tp_str = f"TP1: {format_angka(tp1)} (RR 1:2)\nTP2: {format_angka(tp2)}"
        else:
            tp1 = int(base_entry * 1.03)
            tp2 = int(base_entry * 1.05)
            tp_str = f"TP1: {format_angka(tp1)}\nTP2: {format_angka(tp2)}"

    return entry_str, sl, tp_str

# ==========================================
# 3. ENDPOINT SCANNER
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

        entry, sl, tp = hitung_plan_sakti(data)
        
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
    if kondisi_market == "CRASH":
        MIN_SCORE = 80 
        print("🛡️ MODE PERTAHANAN: Min Score 80")
    
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

# ==========================================
# 4. ENDPOINT DETAIL
# ==========================================
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    ticker_lengkap = ticker_polos + ".JK"
    data = get_cached_analysis(ticker_lengkap)
    
    if data['last_price'] == 0:
        return jsonify({"error": "Not Found", "analysis": {"score":0, "verdict":"ERR", "reason":"-", "type":"-"}})
    
    entry, sl, tp = hitung_plan_sakti(data)
    pct = data.get('change_pct', 0)
    tanda = "+" if pct >= 0 else ""
    
    # 1. Ambil Berita
    list_berita = ambil_berita_saham(ticker_lengkap)

    # 2. [UPDATE] TANYA GEMINI
    analisa_ai_tambahan = ""
    # Cek apakah key sudah diisi valid
    if GEMINI_API_KEY and len(GEMINI_API_KEY) > 20:
         print(f"🤖 Bertanya ke Gemini tentang {ticker_polos}...")
         analisa_ai_tambahan = analisa_dengan_gemini(ticker_polos, data, list_berita)
    else:
         print("⚠️ API Key Invalid/Kosong, Skip Gemini.")
    
    # 3. Gabungkan Hasil Gemini
    reason_final = data['reason']
    if analisa_ai_tambahan:
        reason_final = f"{data['reason']}\n\n====================\n{analisa_ai_tambahan}"

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
# 5. WATCHLIST & STARTUP
# ==========================================
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
    # [PENTING] Pengaturan Port untuk Railway
    # Railway akan memberikan port via environment variable "PORT"
    # Jika dijalankan di laptop, default ke port 5000
    port = int(os.environ.get("PORT", 5000))
    
    print(f"🚀 Alpha Hunter V3 Server berjalan di Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
