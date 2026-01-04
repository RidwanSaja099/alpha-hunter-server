from flask import Flask, jsonify, request
# PENTING: Pastikan file rumus_saham.py ada di folder yang sama
from rumus_saham import analisa_multistrategy, ambil_berita_saham 

app = Flask(__name__)

# ==========================================
# 1. DATABASE SAHAM (KAMUS DATA LENGKAP)
# ==========================================

# A. DATABASE SYARIAH (Manual Update)
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

# B. MARKET UNIVERSE (Kolam Scanner Otomatis)
MARKET_UNIVERSE = [
    "BBRI", "BBCA", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "INDF", "GOTO",
    "ARTO", "BUKA", "EMTK", "MDKA", "ANTM", "INCO", "PGAS", "ADRO", "PTBA", "ITMG",
    "UNVR", "HMSP", "GGRM", "CPIN", "JPFA", "KLBF", "KALBE", "SMGR", "INTP", "BRPT",
    "TPIA", "BREN", "AMMN", "CUAN", "DEWA", "BUMI", "BRMS", "PSAB", "MEDC", "AKRA",
    "EXCL", "ISAT", "JSMR", "ACES", "MAPI", "PWON", "BSDE", "CTRA", "SMRA", "ASRI",
    "BRIS", "HRUM", "INKP", "TKIM", "PANI", "AMRT"
]

# C. WATCHLIST (Daftar Pantauan User)
WATCHLIST = ["BBRI", "BBCA", "GOTO", "ANTM", "BREN", "TLKM"] 

@app.route('/')
def home():
    return "Server Alpha Hunter: READY (Smart Entry Support Mode)"

# ==========================================
# 2. LOGIKA HITUNGAN (MANAJEMEN RISIKO SMART)
# ==========================================

def format_angka(nilai):
    return "{:,}".format(int(nilai)).replace(",", ".")

# [UPDATE PENTING] Menerima Harga Support untuk Entry
def hitung_plan_sakti(harga_sekarang, harga_support, tipe_trading):
    if harga_sekarang <= 0: return "-", 0, "-"
    
    # Validasi Support (Jaga-jaga kalau 0)
    base_entry = harga_support if harga_support > 0 else harga_sekarang
    
    # 1. RENTANG ENTRY (Berdasarkan Support)
    # Area Beli = Harga Support s/d Support + 1.5%
    buy_low = base_entry
    buy_high = int(base_entry * 1.015) 
    
    # Deteksi: Apakah harga sekarang sudah terbang jauh?
    # Jika harga sekarang > area beli tertinggi, kasih peringatan "Wait Pullback"
    status_entry = ""
    if harga_sekarang > (buy_high * 1.01): # Toleransi 1% lagi
        status_entry = "\n(Wait Pullback)"
    
    entry_str = f"{format_angka(buy_low)} - {format_angka(buy_high)}{status_entry}"
    
    # 2. PLAN BERTINGKAT (Dihitung dari BASE ENTRY, bukan harga pucuk)
    # Ini biar Risk/Reward-nya sehat.
    
    if tipe_trading == "ARA": 
        sl = int(base_entry * 0.90) 
        tp_str = "HOLD SAMPAI ARA 🚀"
    
    elif tipe_trading == "BSJP": 
        sl = int(base_entry * 0.98) 
        tp1 = int(base_entry * 1.02)
        tp2 = int(base_entry * 1.04)
        tp3 = int(base_entry * 1.06)
        tp_str = f"TP1: {format_angka(tp1)}\nTP2: {format_angka(tp2)}\nTP3: {format_angka(tp3)}"
    
    elif tipe_trading == "SCALPING": 
        sl = int(base_entry * 0.97) 
        tp1 = int(base_entry * 1.015) # Cuan tipis buat beli cilok
        tp2 = int(base_entry * 1.03)
        tp3 = int(base_entry * 1.05)
        tp_str = f"TP1: {format_angka(tp1)}\nTP2: {format_angka(tp2)}\nTP3: {format_angka(tp3)}"
    
    elif tipe_trading == "SWING": 
        sl = int(base_entry * 0.95) 
        tp1 = int(base_entry * 1.05)
        tp2 = int(base_entry * 1.10)
        tp3 = int(base_entry * 1.15)
        tp_str = f"TP1: {format_angka(tp1)}\nTP2: {format_angka(tp2)}\nTP3: {format_angka(tp3)}"
    
    elif tipe_trading == "INVEST": 
        sl = int(base_entry * 0.85) 
        tp_str = "HOLD JANGKA PANJANG\n(Incar Dividen & Growth)"
    
    else: 
        sl = int(base_entry * 0.95)
        tp_str = "Wait & See / Open"
        entry_str = "-"

    return entry_str, sl, tp_str

# ==========================================
# 3. ENDPOINT SCANNER (DASHBOARD)
# ==========================================

@app.route('/api/scan-results', methods=['GET'])
def get_scan_results():
    target_strategy = request.args.get('strategy', 'ALL') 
    
    daftar_scan = []
    if target_strategy == 'SYARIAH':
        daftar_scan = DATABASE_SYARIAH 
        print(f"🕌 Scanning {len(daftar_scan)} Saham Syariah...")
    elif target_strategy == 'ALL' or target_strategy == 'WATCHLIST':
        daftar_scan = WATCHLIST
        print(f"🔄 Memuat Watchlist...")
    else:
        daftar_scan = MARKET_UNIVERSE
        print(f"🕵️ Mencari '{target_strategy}' di Pasar...")

    results = []
    limit_scan = daftar_scan[:40] 
    
    for kode in limit_scan:
        try:
            ticker = kode + ".JK"
            # 1. Panggil DOKTER (Rumus Baru: Returns support price)
            data = analisa_multistrategy(ticker)
            
            if data['last_price'] == 0: continue

            tipe_ditemukan = data['type']
            
            # 2. FILTER LOGIC
            if target_strategy == 'SYARIAH':
                if data['score'] < 60: continue 
            elif target_strategy not in ['ALL', 'WATCHLIST']:
                if target_strategy not in tipe_ditemukan: continue

            # 3. HITUNG PLAN PRO (Pake Harga Support)
            harga = data.get('last_price', 0)
            support = data.get('support', harga) # Ambil data support dari rumus
            
            # Pass support ke fungsi hitung
            entry, sl, tp = hitung_plan_sakti(harga, support, tipe_ditemukan)
            
            pct = data.get('change_pct', 0)
            tanda = "+" if pct >= 0 else ""
            info_harga = f"Rp {format_angka(harga)} ({tanda}{pct:.2f}%)"

            stock_item = {
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
            results.append(stock_item)
        except Exception as e:
            # print(f"Error scanning {kode}: {e}")
            continue
            
    return jsonify(results)

# ==========================================
# 4. ENDPOINT DETAIL (SEARCH MANUAL)
# ==========================================

@app.route('/api/stock-detail', methods=['GET'])
def get_stock_detail():
    ticker_polos = request.args.get('ticker')
    if not ticker_polos: return jsonify({"error": "No Ticker"}), 400
    
    print(f"🔎 Mencari {ticker_polos}...")
    ticker_lengkap = ticker_polos + ".JK"
    
    data = analisa_multistrategy(ticker_lengkap)
    
    if data['last_price'] == 0:
        return jsonify({
            "ticker": ticker_polos, "company_name": "NOT FOUND", 
            "badges": {"syariah": False, "lq45": False}, 
            "analysis": {"score": 0, "verdict": "ERROR", "reason": "Data not found", "type": "UNKNOWN"}, 
            "plan": {"entry": "-", "stop_loss": 0, "take_profit": "-"}, 
            "news": [], "is_watchlist": False
        })
    
    # Ambil Data dari Rumus
    harga = data.get('last_price', 0)
    support = data.get('support', harga) # Support Pintar
    tipe = data.get('type', 'WAIT')
    
    # Hitung Plan (Pake Support)
    entry, sl, tp = hitung_plan_sakti(harga, support, tipe)
    
    pct = data.get('change_pct', 0)
    tanda = "+" if pct >= 0 else ""
    tp_display = f"{tp}"

    list_berita = ambil_berita_saham(ticker_lengkap)
    
    stock_detail = {
        "ticker": ticker_polos,
        "company_name": f"Rp {format_angka(harga)} ({tanda}{pct:.2f}%)",
        "badges": { "syariah": ticker_polos in DATABASE_SYARIAH, "lq45": True },
        "analysis": {
            "score": int(data['score']),
            "verdict": data['verdict'],
            "reason": data['reason'],
            "type": tipe
        },
        "plan": { "entry": entry, "stop_loss": sl, "take_profit": tp_display },
        "news": list_berita,
        "is_watchlist": ticker_polos in WATCHLIST
    }
    return jsonify(stock_detail)

# ==========================================
# 5. ENDPOINT WATCHLIST
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
    print("🚀 Server Alpha Hunter: READY (Support Entry + Safety First Mode)")
    app.run(host='0.0.0.0', port=5000, debug=True)