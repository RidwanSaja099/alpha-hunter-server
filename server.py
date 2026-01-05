import os
import time
import concurrent.futures
import yfinance as yf
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import math

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

def dapatkan_analisa_ai_cerdas(prompt):
    """Sistem Cerdas: Mencoba Gemini -> Gagal? -> Coba Groq -> Gagal? -> Coba DeepSeek"""
    # 1. COBA GEMINI
    if client_gemini:
        try:
            print("🤖 Menggunakan Gemini AI...")
            response = client_gemini.models.generate_content(
                model='gemini-flash-latest', 
                contents=prompt
            )
            return response.text.strip()
        except Exception as e: print(f"⚠️ Gemini Limit/Error: {e}")

    # 2. COBA GROQ
    if client_groq:
        try:
            print("⚡ Gemini sibuk, beralih ke Groq (Llama 3)...")
            chat_completion = client_groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e: print(f"⚠️ Groq Error: {e}")

    # 3. COBA DEEPSEEK
    if client_deepseek:
        try:
            print("🧠 Groq sibuk, beralih ke DeepSeek...")
            response = client_deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e: print(f"⚠️ DeepSeek Error: {e}")

    return "🤖 Maaf, semua otak AI sedang istirahat. Gunakan data teknikal di atas sebagai panduan."

# ==========================================
# 1. DATABASE & CACHE
# ==========================================
CACHE_DATA = {}
CACHE_TIMEOUT = 300  # 5 Menit
MARKET_STATUS = {"condition": "NORMAL", "last_check": 0}

DATABASE_SYARIAH = ["ACES", "ADRO", "AKRA", "ANTM", "ASII", "BRIS", "BRMS", "BRPT", "BUMI", "CPIN", "GOTO", "ICBP", "INDF", "INKP", "ISAT", "ITMG", "KLBF", "MDKA", "MEDC", "PGAS", "PTBA", "SIDO", "TLKM", "UNTR", "UNVR", "BBRI"]
MARKET_UNIVERSE = ["BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "INDF", "GOTO", "MDKA", "ANTM", "INCO", "PGAS", "ADRO", "PTBA", "BRPT", "BREN", "AMMN"]
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

# FUNGSI AMBIL DATA LIVE + FUNDAMENTAL (TIDAK HILANG)
def ambil_data_live_lengkap(ticker_lengkap):
    try:
        stock = yf.Ticker(ticker_lengkap)
        info = stock.info
        
        # 1. Data Fundamental
        per = info.get('trailingPE', 0)
        pbv = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0)
        sector = info.get('sector', 'Unknown')
        
        # 2. Data Live Candle
        day_open = info.get('open', 0)
        day_high = info.get('dayHigh', 0)
        day_low = info.get('dayLow', 0)
        curr_price = info.get('currentPrice', day_open) 
        volume = info.get('volume', 0)

        candle_stat = "Normal"
        if curr_price > day_open: candle_stat = "🟢 BULLISH (Hijau)"
        elif curr_price < day_open: candle_stat = "🔴 BEARISH (Merah)"
        else: candle_stat = "🟡 DOJ/CROSS (Sama)"

        if day_high > day_low:
            posisi = (curr_price - day_low) / (day_high - day_low) * 100
            posisi_str = f"{posisi:.0f}% dari Low (Dekat {'High' if posisi > 80 else 'Low'})"
        else:
            posisi_str = "Flat"

        data_teks = f"""
        - Sektor: {sector}
        - Fundamental: PER {per:.2f}x | PBV {pbv:.2f}x | ROE {roe*100:.1f}%
        - DATA LIVE HARI INI:
          > Open: {day_open} | High: {day_high} | Low: {day_low} | Last: {curr_price}
          > Kondisi Candle: {candle_stat}
          > Posisi Intraday: {posisi_str}
          > Volume Hari Ini: {volume} lembar
        """
        return data_teks
    except:
        return "Data Fundamental & Live Tidak Tersedia."

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
# 2. LOGIKA HITUNGAN PLAN (PERBAIKAN ILMIAH & PSIKOLOGIS)
# ==========================================

# 1. Aturan Fraksi Harga (Tick) BEI
def get_tick_size(harga):
    if harga < 200: return 1
    elif harga < 500: return 2
    elif harga < 2000: return 5
    elif harga < 5000: return 10
    else: return 25

# 2. Fungsi Pembulatan ke Tick Terdekat
def bulatkan_ke_tick(harga):
    if harga <= 0: return 0
    tick = get_tick_size(harga)
    return int(round(harga / tick) * tick)

# 3. Fungsi Pembulatan Target Psikologis (Dinamis)
def get_psychological_step(harga):
    """Menentukan kelipatan angka bulat yang wajar berdasarkan harga"""
    if harga < 200: return 10      # Harga 50-200, target bulat tiap 10 perak
    elif harga < 1000: return 50   # Harga 200-1000, target bulat tiap 50 perak (550, 600)
    elif harga < 5000: return 100  # Harga 1000-5000, target bulat tiap 100 perak (2100, 2200)
    else: return 250               # Harga >5000, target bulat tiap 250 perak (5250, 5500)

def format_angka(nilai):
    return "{:,}".format(int(nilai)).replace(",", ".")

def hitung_plan_sakti(data_analisa, ticker_fibo=None):
    harga_sekarang = data_analisa.get('last_price', 0)
    harga_support = data_analisa.get('support', 0)
    tipe_trading = data_analisa.get('type', 'UNKNOWN')

    if harga_sekarang <= 0: return "-", 0, "-"
    
    # --- A. PENENTUAN ENTRY (KAIDAH FRONT RUNNING) ---
    # Jika support tidak terdeteksi (0), gunakan 96% harga sekarang (diskon wajar)
    if harga_support == 0: harga_support = int(harga_sekarang * 0.96)
    
    tick_size = get_tick_size(harga_support)
    
    # Entry Ideal: Support + 2-3 Tick (Supaya dapet barang, jangan pasang pas di support)
    buy_low = bulatkan_ke_tick(harga_support + (2 * tick_size))
    # Area Toleransi: Sampai 5 Tick dari Support
    buy_high = bulatkan_ke_tick(buy_low + (3 * tick_size)) 
    
    status_entry = ""
    # Cek apakah harga sudah lari jauh (>3% dari area beli ideal)
    if harga_sekarang > (buy_high * 1.03): 
        status_entry = "\n⚠️ Harga Lari (Wait Pullback)"
    # Jika harga sekarang malah lebih murah dari buy_low (sedang jebol dikit/diskon), sesuaikan
    elif harga_sekarang < buy_low:
        buy_low = harga_sekarang

    entry_str = f"{format_angka(buy_low)} - {format_angka(buy_high)}{status_entry}"
    
    # --- B. PENENTUAN STOP LOSS (KAIDAH FALSE BREAK) ---
    # SL diletakkan DI BAWAH Support, kasih jarak 5-6 Tick biar gak kena "kocokan bandar"
    sl_raw = harga_support - (6 * get_tick_size(harga_support))
    sl = bulatkan_ke_tick(sl_raw)
    
    # --- C. PENENTUAN TARGET PROFIT (FIBONACCI & PSIKOLOGIS) ---
    
    if tipe_trading == "ARA": 
        tp_str = "HOLD SAMPAI ARA 🚀 (Trailing Stop)"
        sl = bulatkan_ke_tick(harga_sekarang * 0.92)
    elif tipe_trading == "INVEST":
        tp_str = "HOLD JANGKA PANJANG (Cek Fundamental)"
        sl = bulatkan_ke_tick(harga_sekarang * 0.85)
    else:
        # Default TP (Math basic)
        tp1 = bulatkan_ke_tick(buy_low * 1.03) # 3% (Tutup fee + kopi)
        tp2 = bulatkan_ke_tick(buy_low * 1.07) # 7% (Profit standar swing)

        # Coba Gunakan Fibonacci jika ticker_fibo ada (Mode Detail)
        if ticker_fibo:
            try:
                # Tarik data 1 bulan ke belakang
                hist = yf.Ticker(ticker_fibo).history(period="1mo")
                if not hist.empty:
                    swing_high = hist['High'].max()
                    swing_low = hist['Low'].min()
                    swing_range = swing_high - swing_low
                    
                    # TP1: Resistance Terdekat (Swing High sebelumnya)
                    tp1_raw = swing_high
                    # TP2: Fibonacci Extension 1.618 (Golden Ratio)
                    tp2_raw = swing_low + (swing_range * 1.618)
                    
                    # Validasi: TP1 tidak boleh terlalu dekat dengan entry (minimal 2%)
                    if tp1_raw < (buy_low * 1.02): tp1_raw = buy_low * 1.03
                    
                    tp1 = bulatkan_ke_tick(tp1_raw)
                    tp2 = bulatkan_ke_tick(tp2_raw)
            except: pass

        # TP3: TARGET PSIKOLOGIS (PEMBULATAN WAJAR)
        # Mencari angka bulat di atas TP2, tapi langkahnya dinamis
        step = get_psychological_step(tp2)
        
        # Contoh: TP2=2030, step=100 -> Target bulat berikutnya = 2100
        tp3_raw = (int(tp2 / step) + 1) * step
        
        # Pastikan TP3 tidak sama dengan TP2 (harus lebih tinggi)
        if tp3_raw <= tp2: tp3_raw += step
        
        tp3 = bulatkan_ke_tick(tp3_raw)

        tp_str = (f"🎯 TP1: {format_angka(tp1)} (Resist/Aman)\n"
                  f"🚀 TP2: {format_angka(tp2)} (Fibo 1.618)\n"
                  f"💎 TP3: {format_angka(tp3)} (Psikologis)")

    return entry_str, sl, tp_str

# ==========================================
# 3. ENDPOINT DETAIL (CORE AI ANALYSIS)
# ==========================================
@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    ticker_lengkap = ticker_polos + ".JK"
    data = get_cached_analysis(ticker_lengkap)
    
    if data['last_price'] == 0:
        return jsonify({"error": "Not Found", "analysis": {"score":0, "verdict":"ERR", "reason":"-", "type":"-"}})
    
    # Ambil Data LIVE
    info_live = ambil_data_live_lengkap(ticker_lengkap)

    # Generate Skor Teknikal
    score = data['score']
    verdict = data['verdict']
    
    parameter_list = []
    if score >= 80:
        parameter_list.append("✅ Strong Uptrend (MA20 > MA50)")
        parameter_list.append("✅ Akumulasi Volume Tinggi")
        parameter_list.append("✅ Momentum RSI Bullish")
    elif score >= 60:
        parameter_list.append("✅ Potensi Reversal/Pantulan")
        parameter_list.append("✅ Support Kuat Teruji")
    elif score >= 40:
        parameter_list.append("⚠️ Konsolidasi/Sideways")
        parameter_list.append("⚠️ Volume Belum Konfirmasi")
    else:
        parameter_list.append("❌ Downtrend Terkonfirmasi")
        parameter_list.append("❌ Tekanan Jual Dominan")
    
    text_parameter = "\n".join(parameter_list)
    rincian_teknikal = f"🔍 **FAKTOR TEKNIS (Skor {score}):**\n{text_parameter}"

    # Persiapan AI
    list_berita = ambil_berita_saham(ticker_lengkap)
    judul_berita = [b['title'] for b in list_berita[:3]] 
    teks_berita = "\n- ".join(judul_berita) if judul_berita else "Tidak ada berita spesifik 24 jam terakhir."

    # [UPDATE FITUR] Prompt AI dengan Permintaan "Second Opinion"
    prompt = f"""
    Kamu adalah Veteran Pasar Modal Indonesia. Analisa saham: {ticker_polos}.
    
    DATA SAHAM SAAT INI:
    - Sinyal Teknikal: {verdict} (Skor: {score}/100)
    {info_live}
    - BERITA TERAKHIR: {teks_berita}
    
    TUGAS ANALISIS (Jawab 6 Poin Ini dengan Tajam):
    
    1. 🕵️‍♂️ **Analisa Dibalik Layar**
       (Kenapa bergerak begini? Lihat 'Data Live', apakah candle kuat atau lemas? Ada aksi bandar?).
       
    2. 📊 **Cek Valuasi & Fundamental**
       (Murah/Mahal berdasarkan PER/PBV? Perusahaan sehat?).
       
    3. ⏱️ **Timing & Strategi Masuk**
       (Lihat posisi intraday. Apakah ini waktu yang tepat untuk "HAKA" atau "Antri Bawah"?).

    4. 🎯 **GAYA TRADING PALING COCOK (PILIH SATU)**
       - ⚡ **BPJS**: Jika candle hijau tebal pagi-pagi.
       - 🌙 **BSJP**: Jika closing kuat di High sore hari.
       - 🏎️ **SCALPING**: Jika volatilitas tinggi.
       - 🌊 **SWING**: Jika uptrend rapi.
       - 💰 **INVESTASI**: Jika fundamental bagus & murah.
       - ⚠️ **HINDARI**: Jika downtrend.
       
    5. 🔢 **PLAN ANGKA (SECOND OPINION)**
       (Berdasarkan intuisimu sebagai Veteran, berikan angka Entry, Stop Loss, dan TP versimu sendiri. Apakah setuju dengan perhitungan teknikal atau punya pandangan lain? Sebutkan angkanya).
       
    6. ⚖️ **VERDICT AKHIR**
       (Kesimpulan Tegas: LAYAK BELI / TIDAK / WAIT AND SEE).
       
    Gunakan bahasa trader Indonesia yang asik.
    """

    # Panggil Multi-AI
    analisa_ai_cerdas = dapatkan_analisa_ai_cerdas(prompt)

    reason_final = f"{rincian_teknikal}\n\n====================\n{analisa_ai_cerdas}"

    # Panggil Rumus Plan (Aktifkan Fibo History untuk Detail)
    entry, sl, tp = hitung_plan_sakti(data, ticker_fibo=ticker_lengkap)
    
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
# 4. ENDPOINT SCANNER (TETAP SAMA)
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

        # Scanner pakai rumus cepat (tanpa fetch history fibo)
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
# 5. WATCHLIST MANAGEMENT (TETAP SAMA)
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
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Alpha Hunter V3 Server berjalan di Port: {port}")
    app.run(host='0.0.0.0', port=port)
