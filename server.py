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

# Ambil API Keys dari Environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Inisialisasi Clients
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

# FUNGSI AMBIL DATA LIVE + FUNDAMENTAL (TIDAK DIUBAH)
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

# FUNGSI MARKET SENTINEL (TIDAK DIUBAH)
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
# 2. LOGIKA HITUNGAN PLAN (BAGIAN INI DI-UPDATE TOTAL)
# ==========================================

# [UPDATE BAGIAN INI] Menambahkan Aturan Fraksi Harga (Tick Rules) BEI
def get_tick_size(harga):
    """Menentukan kelipatan harga sesuai aturan Bursa Efek Indonesia"""
    if harga < 200: return 1
    elif harga < 500: return 2
    elif harga < 2000: return 5
    elif harga < 5000: return 10
    else: return 25

# [UPDATE BAGIAN INI] Fungsi pembulatan agar harga tidak ganjil
def bulatkan_ke_tick(harga):
    if harga <= 0: return 0
    tick = get_tick_size(harga)
    return int(round(harga / tick) * tick)

def format_angka(nilai):
    return "{:,}".format(int(nilai)).replace(",", ".")

# [UPDATE BAGIAN INI] Logika Ilmiah (Fibo + Psikologis + Fraksi)
def hitung_plan_sakti(data_analisa, ticker_fibo=None):
    harga_sekarang = data_analisa.get('last_price', 0)
    harga_support = data_analisa.get('support', 0)
    tipe_trading = data_analisa.get('type', 'UNKNOWN')

    if harga_sekarang <= 0: return "-", 0, "-"
    
    # 1. ENTRY: Jangan antri pas di support (sering gak dapet), naikkan 2 tick (Front Running)
    if harga_support == 0: harga_support = int(harga_sekarang * 0.95)
    
    tick_size = get_tick_size(harga_support)
    buy_low = bulatkan_ke_tick(harga_support + (2 * tick_size))
    buy_high = bulatkan_ke_tick(buy_low + (3 * tick_size)) # Area toleransi beli
    
    status_entry = ""
    if harga_sekarang > (buy_high * 1.03): status_entry = "\n(Wait Pullback)"
    # Kalau harga sekarang dibawah buy low (diskon), pakai harga sekarang
    if harga_sekarang < buy_low: buy_low = harga_sekarang

    entry_str = f"{format_angka(buy_low)} - {format_angka(buy_high)}{status_entry}"
    
    # 2. STOP LOSS: Beri napas 4 tick di bawah support agar tidak kena "False Break"
    sl_raw = harga_support - (4 * get_tick_size(harga_support))
    sl = bulatkan_ke_tick(sl_raw)
    
    # 3. TARGET PRICE (TP): Menggunakan Fibonacci & Psikologis
    if tipe_trading == "ARA": 
        tp_str = "HOLD SAMPAI ARA 🚀"
        sl = bulatkan_ke_tick(harga_sekarang * 0.92)
    elif tipe_trading == "INVEST":
        tp_str = "HOLD JANGKA PANJANG"
        sl = bulatkan_ke_tick(harga_sekarang * 0.85)
    else:
        # Default TP jika history gagal ditarik
        tp1 = bulatkan_ke_tick(buy_low * 1.04)
        tp2 = bulatkan_ke_tick(buy_low * 1.08)

        # Coba Tarik Fibonacci dari History 1 Bulan Terakhir
        if ticker_fibo:
            try:
                hist = yf.Ticker(ticker_fibo).history(period="1mo")
                if not hist.empty:
                    swing_high = hist['High'].max()
                    swing_low = hist['Low'].min()
                    swing_range = swing_high - swing_low
                    
                    # TP1: Resistensi Swing High Sebelumnya
                    tp1_raw = swing_high
                    # TP2: Fibonacci Extension 1.618 (Golden Ratio)
                    tp2_raw = swing_low + (swing_range * 1.618)
                    
                    # Validasi: TP1 minimal untung 3% (biar nutup fee)
                    if tp1_raw < (buy_low * 1.03): tp1_raw = buy_low * 1.03
                    
                    tp1 = bulatkan_ke_tick(tp1_raw)
                    tp2 = bulatkan_ke_tick(tp2_raw)
            except: pass

        # TP3: Target Psikologis (Angka Bulat)
        # Manusia suka TP di angka cantik: 500, 1000, 2000, 5000
        if tp2 < 500: target_bulat = 500
        elif tp2 < 1000: target_bulat = 1000
        elif tp2 < 2000: target_bulat = 2000
        elif tp2 < 5000: target_bulat = 5000
        else: target_bulat = int(tp2 / 1000 + 1) * 1000
        
        if target_bulat <= tp2: target_bulat = bulatkan_ke_tick(tp2 * 1.05)
        tp3 = target_bulat

        tp_str = (f"🎯 TP1: {format_angka(tp1)} (Resist)\n"
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
    
    # Ambil Data LIVE & Fundamental
    info_live = ambil_data_live_lengkap(ticker_lengkap)

    # GENERATE PARAMETER SKOR TEKNIKAL
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

    # PERSIAPAN DATA UNTUK AI
    list_berita = ambil_berita_saham(ticker_lengkap)
    judul_berita = [b['title'] for b in list_berita[:3]] 
    teks_berita = "\n- ".join(judul_berita) if judul_berita else "Tidak ada berita spesifik 24 jam terakhir."

    # PROMPT AI SUPER LENGKAP
    prompt = f"""
    Kamu adalah Veteran Pasar Modal Indonesia. Analisa saham: {ticker_polos}.
    
    DATA SAHAM SAAT INI:
    - Sinyal Teknikal: {verdict} (Skor: {score}/100)
    {info_live}
    - BERITA TERAKHIR: {teks_berita}
    
    TUGAS ANALISIS (Jawab 5 Poin Ini):
    
    1. 🕵️‍♂️ **Analisa Dibalik Layar**
       (Kenapa bergerak begini? Lihat 'Data Live' di atas, apakah candle hari ini kuat atau lemas? Ada aksi bandar/korporasi?).
       
    2. 📊 **Cek Valuasi & Fundamental**
       (Murah/Mahal berdasarkan data PER/PBV di atas? Apakah perusahaan sehat?).
       
    3. ⏱️ **Timing & Strategi Masuk**
       (Lihat posisi intraday. Apakah ini waktu yang tepat untuk "HAKA" atau "Antri Bawah"?).

    4. 🎯 **GAYA TRADING PALING COCOK (PILIH SATU)**
       Pilih yang paling masuk akal berdasarkan data live hari ini:
       - ⚡ **BPJS (Beli Pagi Jual Sore)**: Jika candle hijau tebal & volume tinggi sejak pagi.
       - 🌙 **BSJP (Beli Sore Jual Pagi)**: Jika harga penutupan kuat di dekat High (Akumulasi sore).
       - 🏎️ **SCALPING/FAST TRADE**: Jika volatilitas tinggi (High-Low range lebar).
       - 🌊 **SWING TRADING**: Jika trend uptrend rapi & santai.
       - 💰 **INVESTASI/NABUNG**: Jika fundamental bagus & harga diskon.
       - ⚠️ **HINDARI DULU**: Jika downtrend atau candle merah pekat.
       *(Jelaskan alasan pemilihanmu dalam 1 kalimat)*.
       
    5. ⚖️ **VERDICT AKHIR**
       (Kesimpulan Tegas: LAYAK BELI / TIDAK / WAIT AND SEE).
       
    Gunakan bahasa trader Indonesia yang asik.
    """

    # PANGGIL MULTI-AI
    analisa_ai_cerdas = dapatkan_analisa_ai_cerdas(prompt)

    # GABUNGKAN SEMUA
    reason_final = f"{rincian_teknikal}\n\n====================\n{analisa_ai_cerdas}"

    # [UPDATE BAGIAN INI] Panggil rumus plan baru dengan Ticker Fibo
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
# 4. ENDPOINT SCANNER (TETAP SAMA TAPI UPDATE PANGGILAN RUMUS)
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

        # [UPDATE BAGIAN INI] Scanner pakai rumus baru tapi tanpa Fibo biar cepat (ticker_fibo=None)
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
