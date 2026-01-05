import os
import time
import concurrent.futures
import yfinance as yf
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# --- IMPORT LIBRARY AI ---
from google import genai
from groq import Groq 
from openai import OpenAI 

# Load environment variables
load_dotenv()

# Pastikan file rumus_saham.py ada di folder yang sama
from rumus_saham import analisa_multistrategy, ambil_berita_saham 

app = Flask(__name__)

# ==========================================
# 0. KONFIGURASI MULTI-AI (SISTEM ANTI-LIMIT)
# ==========================================

# Ambil API Keys dari Environment (.env atau Railway Variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")     # Opsional: Untuk Backup 1
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Opsional: Untuk Backup 2

# Inisialisasi Clients (Safe Init - Tidak error jika key kosong)
client_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

def dapatkan_analisa_ai_cerdas(prompt):
    """
    Sistem Cerdas: Mencoba Gemini -> Gagal? -> Coba Groq -> Gagal? -> Coba DeepSeek
    """
    # 1. COBA GEMINI (Prioritas Utama)
    if client_gemini:
        try:
            print("🤖 Menggunakan Gemini AI...")
            response = client_gemini.models.generate_content(
                model='gemini-flash-latest', 
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini Limit/Error: {e}")
            # Lanjut ke backup...

    # 2. COBA GROQ (Backup Tercepat - Llama 3)
    if client_groq:
        try:
            print("⚡ Gemini sibuk, beralih ke Groq (Llama 3)...")
            chat_completion = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Groq Error: {e}")

    # 3. COBA DEEPSEEK (Backup Terpintar)
    if client_deepseek:
        try:
            print("🧠 Groq sibuk, beralih ke DeepSeek...")
            response = client_deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ DeepSeek Error: {e}")

    return "🤖 Maaf, semua otak AI sedang istirahat. Gunakan data teknikal di atas sebagai acuan utama."

# ==========================================
# 1. DATABASE & CACHE
# ==========================================
CACHE_DATA = {}
CACHE_TIMEOUT = 300  # 5 Menit
MARKET_STATUS = {"condition": "NORMAL", "last_check": 0}

# Database Saham
DATABASE_SYARIAH = ["ACES", "ADRO", "AKRA", "ANTM", "ASII", "BRIS", "BRMS", "BRPT", "BUMI", "CPIN", "GOTO", "ICBP", "INDF", "INKP", "ISAT", "ITMG", "KLBF", "MDKA", "MEDC", "PGAS", "PTBA", "SIDO", "TLKM", "UNTR", "UNVR", "BBRI"]
MARKET_UNIVERSE = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "INDF", "GOTO", "MDKA", "ANTM", "INCO", "PGAS", "ADRO", "PTBA", "BRPT", "BREN", "AMMN"]

# Watchlist Default (10 Saham)
WATCHLIST = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "GOTO", "ANTM", "ADRO", "UNTR"]

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
            MARKET_STATUS['condition'] = "CRASH" if change_pct < -0.008 else "NORMAL"
        MARKET_STATUS['last_check'] = now
    except:
        MARKET_STATUS['condition'] = "NORMAL"
    return MARKET_STATUS['condition']

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
# 3. ENDPOINT DETAIL (CORE LOGIC DENGAN FITUR BARU)
# ==========================================
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    ticker_lengkap = ticker_polos + ".JK"
    data = get_cached_analysis(ticker_lengkap)
    
    if data['last_price'] == 0:
        return jsonify({"error": "Not Found", "analysis": {"score":0, "verdict":"ERR", "reason":"-", "type":"-"}})
    
    # --- FITUR BARU: GENERATE PARAMETER SKOR TEKNIKAL ---
    # Ini membuat alasan kenapa skornya sekian menjadi transparan sebelum AI bicara
    score = data['score']
    verdict = data['verdict']
    
    parameter_list = []
    if score >= 80:
        parameter_list.append("✅ Strong Uptrend (MA20 > MA50)")
        parameter_list.append("✅ Volume Akumulasi Tinggi")
        parameter_list.append("✅ Momentum Positif")
    elif score >= 60:
        parameter_list.append("✅ Indikasi Reversal/Pantulan")
        parameter_list.append("✅ Support Kuat Teruji")
        parameter_list.append("✅ Volume Stabil")
    elif score >= 40:
        parameter_list.append("⚠️ Fase Konsolidasi/Sideways")
        parameter_list.append("⚠️ Volume Belum Signifikan")
    else:
        parameter_list.append("❌ Downtrend Terkonfirmasi")
        parameter_list.append("❌ Tekanan Jual Tinggi")
    
    # Gabungkan jadi string bullet points
    text_parameter = "\n".join(parameter_list)
    rincian_teknikal = f"🔍 **PARAMETER TERPENUHI (Skor {score}):**\n{text_parameter}"

    # --- PERSIAPAN DATA UNTUK AI ---
    list_berita = ambil_berita_saham(ticker_lengkap)
    judul_berita = [b['title'] for b in list_berita[:3]] 
    teks_berita = "\n- ".join(judul_berita) if judul_berita else "Tidak ada berita spesifik hari ini."

    # --- PROMPT KAUSALITAS (SEBAB-AKIBAT) ---
    prompt = f"""
    Bertindaklah sebagai Senior Analis Saham IDX.
    Analisa: {ticker_polos}.
    
    DATA TEKNIS:
    - Status: {verdict} (Skor {score}/100)
    - Harga: {data['last_price']}
    
    BERITA TERAKHIR:
    {teks_berita}
    
    TUGAS (ANALISA MENGAPA/WHY):
    1. Jelaskan secara logis *KENAPA* saham ini mendapat skor {score}? (Hubungkan sentimen berita dengan pergerakan teknikal).
    2. Apa resiko terbesar jika masuk besok?
    
    Jawab singkat (3-4 kalimat), padat, berbobot. Awali dengan emoji 🧠.
    """

    # --- PANGGIL MULTI-AI ---
    analisa_ai_cerdas = dapatkan_analisa_ai_cerdas(prompt)

    # --- GABUNGKAN SEMUA ---
    # Format: Parameter Teknikal + Garis Pemisah + Analisa AI
    reason_final = f"{rincian_teknikal}\n\n====================\n{analisa_ai_cerdas}"

    entry, sl, tp = hitung_plan_sakti(data)
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
# 4. ENDPOINT SCANNER
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
# 5. WATCHLIST MANAGEMENT
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
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Alpha Hunter V3 Server berjalan di Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
