import sqlite3

# Nama Database kita (akan jadi file .db)
DB_NAME = "ihsg_hunter.db"

def create_database():
    print(f"🛠️ Sedang membangun Database {DB_NAME}...")
    
    # 1. Buka Koneksi ke Database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # --- MEMBUAT TABEL 1: MASTER SAHAM (KTP) ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stocks_master (
        ticker TEXT PRIMARY KEY,
        company_name TEXT,
        sector TEXT,
        is_syariah BOOLEAN DEFAULT 0,
        is_lq45 BOOLEAN DEFAULT 0,
        special_status TEXT DEFAULT 'NORMAL'
    )
    ''')
    
    # --- MEMBUAT TABEL 2: HASIL SCAN (ANALISIS) ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scan_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        ticker TEXT,
        scanner_type TEXT,
        accuracy_score INTEGER,
        ai_verdict TEXT,
        ai_reason TEXT,
        entry_area TEXT,
        stop_loss INTEGER,
        take_profit TEXT,
        FOREIGN KEY (ticker) REFERENCES stocks_master(ticker)
    )
    ''')

    # --- MEMBUAT TABEL 3: BERITA & SENTIMEN ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        title TEXT,
        category TEXT,
        ai_sentiment TEXT,
        impact_score TEXT,
        url_link TEXT
    )
    ''')

    # --- MENGISI DATA CONTOH (DUMMY) AGAR TIDAK KOSONG ---
    # Kita isi BBRI dan BREN sebagai contoh awal
    print("📝 Mengisi data contoh awal...")
    
    # Contoh Master Saham
    cursor.execute("INSERT OR IGNORE INTO stocks_master VALUES (?, ?, ?, ?, ?, ?)", 
                   ('BBRI.JK', 'Bank Rakyat Indonesia', 'Finance', 0, 1, 'NORMAL'))
    cursor.execute("INSERT OR IGNORE INTO stocks_master VALUES (?, ?, ?, ?, ?, ?)", 
                   ('BREN.JK', 'Barito Renewables Energy', 'Infrastructure', 1, 1, 'NORMAL'))
    cursor.execute("INSERT OR IGNORE INTO stocks_master VALUES (?, ?, ?, ?, ?, ?)", 
                   ('GOTO.JK', 'GoTo Gojek Tokopedia', 'Technology', 1, 1, 'NORMAL'))

    # Contoh Hasil Scan (Pura-pura hasil scan AI)
    cursor.execute('''
        INSERT INTO scan_results (ticker, scanner_type, accuracy_score, ai_verdict, ai_reason, entry_area, stop_loss, take_profit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('BBRI.JK', 'SWING', 96, 'STRONG BUY', 'Akumulasi Asing masif & Teknikal Rebound di Support Kuat.', '3600-3620', 3550, 'TP1: 3800, TP2: 4000'))

    # Simpan Perubahan
    conn.commit()
    conn.close()
    
    print("✅ SUKSES! Database 'ihsg_hunter.db' berhasil dibuat.")

if __name__ == "__main__":
    create_database()