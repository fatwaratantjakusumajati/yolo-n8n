import cv2
import time
import datetime
import json
import os
os.environ["OPENVINO_DEVICE"] = "CPU"
import threading
import requests
import psycopg2
from ultralytics import YOLO
import base64
import gc
import queue
import logging
import numpy as np
import traceback
from dotenv import load_dotenv

# Muat variabel dari file .env (harus ada di folder yang sama dengan script
# ini, atau di working directory saat script dijalankan). Kalau .env tidak
# ada, os.getenv() di bawah akan pakai nilai default yang disediakan supaya
# script tetap bisa jalan (tapi sebaiknya .env selalu ada di production).
load_dotenv()

# ======================================================================
# CREDENTIALS — dari .env (JANGAN hardcode username/password di sini lagi)
# ======================================================================
RTSP_USER = os.getenv("RTSP_USER")
RTSP_PASS = os.getenv("RTSP_PASS")
RTSP_HOST = os.getenv("RTSP_HOST")
RTSP_PORT = os.getenv("RTSP_PORT")

DB_HOST     = os.getenv("DB_HOST")
_DB_PORT_RAW = os.getenv("DB_PORT")
DB_PORT     = int(_DB_PORT_RAW) if _DB_PORT_RAW else None
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

MAIN_MODEL_PATH   = os.getenv("MAIN_MODEL_PATH")
PERSON_MODEL_PATH = os.getenv("PERSON_MODEL_PATH")

# ---- Validasi: pastikan semua kredensial/path/host/port wajib sudah diisi di .env ----
# Sengaja HANYA menyebut nama variabel yang kosong, bukan nilainya, supaya
# tidak ada info sensitif (termasuk host & port) yang ikut tercetak di log/console.
_REQUIRED_ENV_VARS = {
    "RTSP_USER": RTSP_USER,
    "RTSP_PASS": RTSP_PASS,
    "RTSP_HOST": RTSP_HOST,
    "RTSP_PORT": RTSP_PORT,
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "MAIN_MODEL_PATH": MAIN_MODEL_PATH,
    "PERSON_MODEL_PATH": PERSON_MODEL_PATH,
}
_missing_env_vars = [k for k, v in _REQUIRED_ENV_VARS.items() if not v]
if _missing_env_vars:
    raise RuntimeError(
        "Variabel .env berikut belum diisi (cek file .env di folder yang sama "
        f"dengan script ini): {', '.join(_missing_env_vars)}"
    )

# ======================================================================
# CONFIGURATION — CCTV ALERT (1 kamera dengan YOLO detection)
# ======================================================================
CCTV_SOURCE         = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}:{RTSP_PORT}/Streaming/Channels/1402"
N8N_WEBHOOK_URL     = os.getenv("N8N_WEBHOOK_URL")
N8N_HEARTBEAT_URL   = os.getenv("N8N_HEARTBEAT_URL")
CAMERA_NAME         = os.getenv("CAMERA_NAME", "Kamera 7")  # nama kamera, bukan info sensitif

_missing_url_vars = [k for k, v in {
    "N8N_WEBHOOK_URL": N8N_WEBHOOK_URL,
    "N8N_HEARTBEAT_URL": N8N_HEARTBEAT_URL,
}.items() if not v]
if _missing_url_vars:
    raise RuntimeError(
        "Variabel .env berikut belum diisi (cek file .env di folder yang sama "
        f"dengan script ini): {', '.join(_missing_url_vars)}"
    )

ROI_COORDINATES     = [(577, 182), (464, 186), (396, 31), (442, 27)]

_roi_pts_arr = np.array(ROI_COORDINATES, dtype=np.int32)
ROI_BBOX_X1  = int(_roi_pts_arr[:, 0].min())
ROI_BBOX_Y1  = int(_roi_pts_arr[:, 1].min())
ROI_BBOX_X2  = int(_roi_pts_arr[:, 0].max())
ROI_BBOX_Y2  = int(_roi_pts_arr[:, 1].max())
ROI_CROP_PADDING = 40

WAKTU_MAKSIMAL_DIAM = 259200
ALERT_COOLDOWN      = 30
ALERT_SEND_DELAY    = 2

ESCALATION_THRESHOLDS = [259200, 432000, 604800, 1209600, 2592000] # 3, 5, 7, 14, 30 hari

ALERT_ZONE_SNAP     = 10
PIXEL_TOLERANCE     = 15
CONF_THRESHOLD      = 0.2   # dinaikkan dari 0.1 - nilai lama terlalu longgar, menangkap lantai/bayangan/orang sebagai "box"
IOU_THRESHOLD       = 0.45  # diturunkan dari 0.7 - NMS jadi lebih agresif membuang box duplikat/tumpang tindih

# ---- FILTER PERSON (anti false-positive "box" dari kepala/badan orang) ----
# Model "box" cuma dilatih 1 kelas, jadi kadang salah nangkep kepala/badan
# orang yang lewat/berhenti di ROI sebagai "box". Untuk nyaring ini, jalankan
# model umum COCO (punya kelas "person") di frame yang sama, lalu buang
# deteksi "box" yang overlap signifikan dengan deteksi "person".
PERSON_FILTER_ENABLED    = True   # DIAKTIFKAN KEMBALI. Sekarang, kalau folder OpenVINO custom di
                                    # PERSON_MODEL_PATH (lihat .env) tidak ada/tidak valid, kode di
                                    # main() otomatis FALLBACK ke model pretrained "yolov8n.pt"
                                    # (auto-download oleh ultralytics, punya kelas 'person' id=0)
                                    # alih-alih membiarkan person_model None diam-diam. Kalaupun
                                    # model gagal dimuat total, try/except di sekitar predict()
                                    # tetap memastikan ini TIDAK bisa menjatuhkan main()/reset tracker.
# PERSON_MODEL_PATH diambil dari .env di bagian atas file (lihat CREDENTIALS)
PERSON_CONF_THRESHOLD    = 0.35
PERSON_OVERLAP_THRESHOLD = 0.2   # persentase area box "box" yang ketutup person -> dianggap false-positive

# Jumlah kemunculan berturut-turut (dalam siklus deteksi setelah FRAME_SKIP) yang
# dibutuhkan sebelum sebuah deteksi dianggap "box valid" dan mulai dihitung timernya.
# Ini menyaring deteksi sekali-lewat (noise satu frame) supaya tidak langsung tercatat.
CONFIRMATION_HITS_REQUIRED = 3
IMG_SIZE            = 640
MAX_RECONNECT       = 10
ROI_OVERLAP_RATIO   = 0.0

KELAS_DIMONITOR    = None

DEBUG_MODE          = True
OFFLINE_THRESHOLD   = 60
FRAME_SKIP          = 10

CAMERA_FPS               = 25
MISS_TOLERANCE_SECONDS   = 600
MISS_TOLERANCE           = max(1, int(MISS_TOLERANCE_SECONDS * CAMERA_FPS / FRAME_SKIP))
POSITION_TIMER_GRACE_PERIOD = 259200 # 3 hari

GLOBAL_ALERT_COOLDOWN = 60
ALERT_ZONE_DEDUP_RADIUS = 50

CLEANUP_INTERVAL    = 3600
MAX_HISTORY_AGE     = 3456000

HEARTBEAT_INTERVAL  = 600

ALERT_TIMEOUT       = 30
ALERT_MAX_RETRY     = 3
ALERT_RETRY_DELAY   = 5

# ======================================================================
# PREVIEW KONFIGURASI
# ======================================================================
SHOW_PREVIEW = False
PREVIEW_WIDTH = 640

# ======================================================================
# CONFIGURATION — MONITOR ALERT (22 kamera, cek offline/online)
# ======================================================================
MONITOR_OFFLINE_THRESHOLD = 60
CHECK_INTERVAL            = 60
NOTIF_COOLDOWN            = 300

# Dibuat otomatis dari kredensial .env (pola channel NVR: kamera ke-N -> N*100+2),
# supaya password RTSP tidak lagi tertulis berulang 22x di source code.
CAMERAS = [
    {
        "name": f"Kamera {i}",
        "url": f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}:{RTSP_PORT}/Streaming/Channels/{i * 100 + 2}",
    }
    for i in range(1, 23)
]

# ======================================================================
# CONFIGURATION — DATABASE
# ======================================================================
# DB_HOST/PORT/NAME/USER/PASSWORD diambil dari .env di bagian atas file
# (lihat CREDENTIALS), bukan hardcode di sini lagi.

# ======================================================================
# LOGGING SETUP
# ======================================================================
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    file_handler = RotatingFileHandler("cctv_alert.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

setup_logging()
log = logging.getLogger(__name__)

# ======================================================================
# DATABASE
# ======================================================================
# ----------------------------------------------------------------------
# PERBAIKAN: connection pool + retry
# ----------------------------------------------------------------------
# Dulu setiap fungsi DB (insert_staging_row, update_alert_escalation, dst)
# membuka koneksi BARU sendiri lewat psycopg2.connect(). Kalau banyak box
# terkonfirmasi BERSAMAAN dalam satu siklus frame (mis. satu truk bongkar
# banyak barang sekaligus), script bisa mencoba buka belasan koneksi baru
# dalam waktu sangat singkat -> gampang kena limit max_connections Postgres.
# Akibatnya sebagian INSERT berhasil tapi UPDATE penyimpan foto yang
# menyusul gagal, dan foto jadi kosong PERMANEN untuk baris itu (karena
# tidak ada retry, dan tidak ada UPDATE susulan kalau barang keburu keluar
# dari ROI sebelum ambang alert berikutnya).
#
# Solusinya: (1) pakai POOL koneksi yang dibagi bareng, jadi tidak ada lagi
# lonjakan pembukaan koneksi baru; (2) tambah RETRY otomatis khusus untuk
# operasi tulis (INSERT/UPDATE) supaya gangguan sesaat tidak langsung bikin
# baris kehilangan foto selamanya.
from psycopg2 import pool as _pg_pool

DB_POOL_MIN_CONN = 2
DB_POOL_MAX_CONN = 15   # beri ruang untuk lonjakan banyak box bersamaan

DB_WRITE_RETRY       = 3
DB_WRITE_RETRY_DELAY = 1.5  # detik, dikali percobaan ke-berapa (backoff)

_db_pool = None
_db_pool_lock = threading.Lock()

def _get_pool():
    global _db_pool
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                try:
                    _db_pool = _pg_pool.ThreadedConnectionPool(
                        DB_POOL_MIN_CONN, DB_POOL_MAX_CONN,
                        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                        user=DB_USER, password=DB_PASSWORD
                    )
                    log.info(f"[DB] Connection pool dibuat (min={DB_POOL_MIN_CONN}, max={DB_POOL_MAX_CONN})")
                except Exception as e:
                    log.error(f"[DB ERROR] Gagal membuat connection pool: {e}")
                    _db_pool = None
    return _db_pool

def get_db_connection():
    """Ambil koneksi dari pool (BUKAN bikin koneksi baru tiap panggil)."""
    p = _get_pool()
    if p is None:
        return None
    try:
        conn = p.getconn()
        # kalau koneksi lama ternyata sudah putus/rusak, buang & minta yang baru
        if conn.closed:
            p.putconn(conn, close=True)
            conn = p.getconn()
        return conn
    except Exception as e:
        log.error(f"[DB ERROR] Gagal ambil koneksi dari pool: {e}")
        return None

def release_db_connection(conn):
    """Kembalikan koneksi ke pool. GANTI semua conn.close() dengan ini."""
    if conn is None:
        return
    p = _get_pool()
    try:
        if p is not None:
            p.putconn(conn)
        else:
            conn.close()
    except Exception as e:
        log.warning(f"[DB] Gagal mengembalikan koneksi ke pool: {e}")

def db_write_with_retry(func, *args, **kwargs):
    """
    Jalankan fungsi penulis DB (insert_staging_row / update_alert_escalation /
    update_exit_status) dengan retry otomatis kalau percobaan pertama gagal
    total (exception, atau fungsi mengembalikan None padahal seharusnya ada
    hasil). Dipakai supaya gangguan koneksi sesaat tidak langsung bikin
    foto/baris hilang permanen.
    """
    last_result = None
    for attempt in range(1, DB_WRITE_RETRY + 1):
        try:
            last_result = func(*args, **kwargs)
            if last_result:
                return last_result
        except Exception as e:
            log.error(f"[DB WRITE RETRY] {func.__name__} percobaan {attempt}/{DB_WRITE_RETRY} error: {e}")
        if attempt < DB_WRITE_RETRY:
            wait = DB_WRITE_RETRY_DELAY * attempt
            log.warning(f"[DB WRITE RETRY] {func.__name__} gagal, coba lagi dalam {wait:.1f}s "
                        f"(percobaan {attempt}/{DB_WRITE_RETRY})")
            time.sleep(wait)
    log.error(f"[DB WRITE RETRY] {func.__name__} tetap gagal setelah {DB_WRITE_RETRY} percobaan")
    return last_result

def simpan_log_offline(camera_name, duration):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO alert_log (camera, class_name, duration, timestamp, alert_level)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            camera_name,
            "KAMERA OFFLINE",
            int(duration),
            datetime.datetime.now(),
            "CRITICAL"
        ))
        conn.commit()
        log.info(f"[DB] Log offline tersimpan: {camera_name}")
    except Exception as e:
        log.error(f"[DB ERROR] {e}")
    finally:
        release_db_connection(conn)
def simpan_log_online(camera_name):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO alert_log (camera, class_name, duration, timestamp, alert_level)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            camera_name,
            "KAMERA ONLINE",
            0,
            datetime.datetime.now(),
            "INFO"
        ))
        conn.commit()
        log.info(f"[DB] Log online tersimpan: {camera_name}")
    except Exception as e:
        log.error(f"[DB ERROR] {e}")
    finally:
        release_db_connection(conn)
def insert_staging_row(camera_name, class_name, first_detected_dt, foto_bytes=None, file_name=None):
    """
    Buat SATU baris di alert_log untuk satu objek yang baru terkonfirmasi diam di ROI.
    Kolom 'first_detected' & 'timestamp' diisi waktu deteksi pertama, duration=0.
    Mengembalikan id baris (dipakai untuk UPDATE saat eskalasi/keluar) supaya
    tidak ada baris duplikat untuk objek yang sama.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO alert_log
                (camera, class_name, duration, timestamp, foto_base64, file_name, alert_level, first_detected)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            camera_name,
            class_name,
            0,
            first_detected_dt,
            psycopg2.Binary(foto_bytes) if foto_bytes else None,
            file_name,
            "STAGING",
            first_detected_dt,
        ))
        new_id = cur.fetchone()[0]
        conn.commit()
        log.info(f"[DB] Baris staging dibuat id={new_id} camera={camera_name} class={class_name}")
        return new_id
    except Exception as e:
        log.error(f"[DB ERROR] insert_staging_row: {e}")
        return None
    finally:
        release_db_connection(conn)
def update_alert_escalation(db_id, alert_dt, duration, level_index, foto_bytes=None, file_name=None):
    """
    Update baris staging yang sudah ada saat terjadi alert eskalasi (durasi diam melewati
    salah satu ESCALATION_THRESHOLDS). level_index 1/2/3 -> kolom alert_sent_1/2/3.
    Eskalasi ke-4 & ke-5 (tabel cuma punya 3 kolom alert_sent) akan menimpa alert_sent_3.
    """
    if db_id is None:
        return False
    col = f"alert_sent_{min(level_index, 3)}"
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        # PENTING: duration TIDAK lagi dikirim dari tracker["start"] di memori
        # (yang bisa desync dari first_detected kalau _position_timer sempat
        # ke-reset akibat goyangnya ukuran box, sedangkan _position_db_id yang
        # nunjuk ke baris ini tetap dipertahankan). duration sekarang SELALU
        # dihitung ulang dari first_detected yang tersimpan di baris ini
        # sendiri, supaya first_detected, timestamp, dan duration di DB
        # dijamin selalu konsisten satu sama lain, apapun yang terjadi di
        # cache memori.
        if foto_bytes:
            cur.execute(f"""
                UPDATE alert_log
                SET timestamp = %s,
                    duration = EXTRACT(EPOCH FROM (%s - first_detected))::int,
                    {col} = %s,
                    alert_level = %s,
                    foto_base64 = %s,
                    file_name = %s
                WHERE id = %s
            """, (alert_dt, alert_dt, alert_dt, "WARNING",
                psycopg2.Binary(foto_bytes), file_name, db_id))
        else:
            cur.execute(f"""
                UPDATE alert_log
                SET timestamp = %s,
                    duration = EXTRACT(EPOCH FROM (%s - first_detected))::int,
                    {col} = %s,
                    alert_level = %s
                WHERE id = %s
            """, (alert_dt, alert_dt, alert_dt, "WARNING", db_id))
        conn.commit()
        log.info(f"[DB] Eskalasi #{level_index} tersimpan id={db_id} durasi_tracker={int(duration)}s "
                 f"(kolom duration di-derive ulang dari first_detected di DB)")
        return True
    except Exception as e:
        log.error(f"[DB ERROR] update_alert_escalation: {e}")
        return False
    finally:
        release_db_connection(conn)

def update_exit_status(db_id, exit_dt, duration):
    """Update baris saat objek dianggap keluar dari ROI (tracker dihapus krn hilang lama)."""
    if db_id is None:
        return False
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        # Sama seperti update_alert_escalation: duration di-derive ulang dari
        # first_detected baris ini sendiri, bukan dari timer tracker di memori.
        cur.execute("""
            UPDATE alert_log
            SET duration = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - first_detected))::int,
                timestamp = CURRENT_TIMESTAMP, alert_level = %s
            WHERE id = %s
        """, ("SELESAI", db_id))
        conn.commit()
        log.info(f"[DB] Status keluar tersimpan id={db_id} durasi_tracker={int(duration)}s "
                 f"(kolom duration di-derive ulang dari first_detected di DB)")
        return True
    except Exception as e:
        log.error(f"[DB ERROR] update_exit_status: {e}")
        return False
    finally:
        release_db_connection(conn)
# ======================================================================
# ALERT HISTORY
# ======================================================================
_alert_history      = {}
_alert_history_lock = threading.Lock()

def get_alert_history_key(cx, cy, class_name):
    snap_cx = (cx // ALERT_ZONE_SNAP) * ALERT_ZONE_SNAP
    snap_cy = (cy // ALERT_ZONE_SNAP) * ALERT_ZONE_SNAP
    return (snap_cx, snap_cy, class_name)

def get_last_alert_time(cx, cy, class_name):
    key = get_alert_history_key(cx, cy, class_name)
    with _alert_history_lock:
        return _alert_history.get(key, 0)

def set_last_alert_time(cx, cy, class_name, t=None):
    key = get_alert_history_key(cx, cy, class_name)
    with _alert_history_lock:
        _alert_history[key] = t if t is not None else time.time()

_last_global_alert = 0
_global_alert_lock = threading.Lock()

def can_send_global_alert():
    global _last_global_alert
    with _global_alert_lock:
        now = time.time()
        if now - _last_global_alert >= GLOBAL_ALERT_COOLDOWN:
            _last_global_alert = now
            return True
        return False

_active_alert_zone = {}
_active_alert_zone_lock = threading.Lock()

def is_duplicate_alert(cx, cy, class_name):
    with _active_alert_zone_lock:
        now = time.time()

        expired = [k for k, v in _active_alert_zone.items()
            if now - v['time'] > ALERT_COOLDOWN]
        for k in expired:
            del _active_alert_zone[k]

        for key, val in _active_alert_zone.items():
            if val['class'] != class_name:
                continue
            dist = ((cx - val['cx'])**2 + (cy - val['cy'])**2)**0.5
            if dist < ALERT_ZONE_DEDUP_RADIUS:
                return True

        _active_alert_zone[f"{cx}_{cy}_{class_name}"] = {
            'cx': cx, 'cy': cy,
            'class': class_name,
            'time': now
        }
        return False

# ======================================================================
# BARANG MASUK / KELUAR — NOTIFIKASI n8n (opsional, DB sudah dijaga anti-duplikat
# lewat db_id per tracker: insert sekali di insert_staging_row, update selanjutnya)
# ======================================================================
def notif_barang_masuk(camera_name, class_name, tkey, entry_dt):
    try:
        requests.post(
            N8N_WEBHOOK_URL,
            data={
                "class_name":   "BARANG MASUK",
                "duration":     0,
                "timestamp":    entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "camera":       camera_name,
                "image_base64": "",
                "object_id":    tkey,
                "object_class": class_name,
            },
            timeout=10
        )
    except Exception as e:
        log.warning(f"[BARANG MASUK] Gagal kirim webhook: {e}")

def notif_barang_keluar(camera_name, class_name, tkey, entry_dt, exit_dt, durasi):
    try:
        requests.post(
            N8N_WEBHOOK_URL,
            data={
                "class_name":   "BARANG KELUAR",
                "duration":     int(durasi),
                "timestamp":    exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "camera":       camera_name,
                "image_base64": "",
                "object_id":    tkey,
                "object_class": class_name,
                "masuk_pada":   entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
            },
            timeout=10
        )
    except Exception as e:
        log.warning(f"[BARANG KELUAR] Gagal kirim webhook: {e}")

# ======================================================================
# POSITION TIMER
# ======================================================================
# ----------------------------------------------------------------------
# PERBAIKAN DUPLIKAT #1 — pencocokan posisi berbasis JARAK, bukan grid snap
# ----------------------------------------------------------------------
# Sebelumnya posisi disamakan dengan "snap" ke grid ALERT_ZONE_SNAP=10px
# lewat floor-division (cx // 10 * 10). Itu punya EFEK BATAS GRID yang
# keras: dua titik yang jaraknya cuma 1-2px tapi kebetulan jatuh di SISI
# LAIN garis grid (misal cx=519 -> bucket 510, cx=521 -> bucket 520)
# dianggap POSISI BERBEDA TOTAL. Barang yang diam sama sekali tidak
# bergeser pun bisa "terpental" jadi identitas baru gara-gara ini, lalu
# jadi baris/alert duplikat di dashboard.
#
# Sekarang: cari entry TERDEKAT dalam radius POSITION_MATCH_RADIUS (jarak
# Euclidean asli ke posisi terakhir yang tersimpan), bukan berdasarkan
# kotak grid. Kalau tidak ada yang cukup dekat, baru dianggap posisi baru.
_position_timer      = {}
_position_timer_lock = threading.Lock()
POSITION_SIZE_TOLERANCE = 0.5
POSITION_MATCH_RADIUS   = 25  # px — toleransi jarak untuk dianggap "posisi yang sama"

def _find_nearby_key(store, cx, cy, class_name, radius):
    """Cari key (cx, cy, class_name) di 'store' yang class-nya sama dan
    jaraknya ke (cx,cy) < radius. Dipanggil sambil memegang lock store-nya."""
    best_key, best_dist = None, float("inf")
    for key in store.keys():
        kx, ky, kcls = key
        if kcls != class_name:
            continue
        dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
        if dist < radius and dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key

def get_position_start(cx, cy, class_name, bw=None, bh=None):
    with _position_timer_lock:
        key = _find_nearby_key(_position_timer, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is None:
            return None
        data = _position_timer[key]
        gap = time.time() - data["last_seen"]
        if gap > POSITION_TIMER_GRACE_PERIOD:
            del _position_timer[key]
            return None

        if bw is not None and bh is not None and "size" in data:
            pw, ph = data["size"]
            size_diff_w = abs(bw - pw) / max(pw, 1)
            size_diff_h = abs(bh - ph) / max(ph, 1)
            if size_diff_w > POSITION_SIZE_TOLERANCE or size_diff_h > POSITION_SIZE_TOLERANCE:
                del _position_timer[key]
                return None

        return data["start"]

def set_position_start(cx, cy, class_name, t=None, bw=None, bh=None):
    now = t if t is not None else time.time()
    with _position_timer_lock:
        key = _find_nearby_key(_position_timer, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is None:
            _position_timer[(cx, cy, class_name)] = {"start": now, "last_seen": now, "size": (bw, bh)}

def touch_position_start(cx, cy, class_name, bw=None, bh=None):
    with _position_timer_lock:
        key = _find_nearby_key(_position_timer, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is not None:
            _position_timer[key]["last_seen"] = time.time()
            if bw is not None and bh is not None:
                _position_timer[key]["size"] = (bw, bh)

# ----------------------------------------------------------------------
# POSITION -> DB_ID MAPPING (SENGAJA TERPISAH dari _position_timer!)
# ----------------------------------------------------------------------
# JANGAN digabung ke dalam record _position_timer. get_position_start() di
# atas menghapus SELURUH record-nya begitu ukuran box berbeda >50% dari
# sebelumnya (POSITION_SIZE_TOLERANCE) -- itu masuk akal untuk keputusan
# "apakah timer durasi boleh dilanjutkan", tapi KALAU db_id ikut nebeng di
# situ, db_id ikut kehapus tiap kali ukuran box goyang (wajar terjadi di
# tumpukan box yang padat/berdesakan -- box lain lewat di depan, sebagian
# ketutup, dst), dan itu memicu insert_staging_row baru lagi. db_id
# SEHARUSNYA hanya boleh direset saat barang benar-benar dianggap keluar
# (lewat jalur miss_count timeout di cleanup tracker), bukan setiap kali
# estimasi ukuran box goyang sedikit.
#
# ----------------------------------------------------------------------
# PERBAIKAN DUPLIKAT #2 — TAHAN RESTART PROSES (root cause utama duplikat)
# ----------------------------------------------------------------------
# _position_db_id ini SEBELUMNYA cuma hidup di memori proses Python. Dari
# log ketahuan proses ini beberapa kali RESTART TOTAL (bukan cuma main()
# yang di-supervisor, tapi seluruh interpreter Python jalan ulang dari nol
# -- misal karena crash native/OOM, restart manual, atau reboot mesin).
# Begitu proses restart, dict ini KOSONG LAGI, padahal barang yang sudah
# lama diam (kadang sudah 4-5 hari, sudah punya baris DB terbuka) masih
# ada di ROI. Ketika barang itu nanti mencapai ambang alert lagi, kode
# tidak ketemu db_id lamanya (karena mapping-nya hilang) -> bikin baris DB
# BARU untuk barang yang SAMA -> DUPLIKAT DI DASHBOARD.
#
# Perbaikan: mapping posisi -> db_id (dan waktu mulai diam) di-checkpoint
# ke file JSON di disk setiap kali berubah, dan di-load kembali begitu
# script (proses baru) mulai. Jadi identitas barang tidak hilang lagi
# walau prosesnya sempat mati/di-restart, ROI/kamera tidak berubah.
POSITION_STATE_FILE = os.getenv("POSITION_STATE_FILE", "position_state.json")

def _save_position_state():
    """Tulis snapshot _position_timer + _position_db_id ke disk (atomic write).
    Dipanggil tiap kali db_id berubah (assign/clear) dan berkala dari
    cleanup_old_entries(), supaya kalau proses mati mendadak, checkpoint
    yang hilang paling banyak cuma perubahan sejak penyimpanan terakhir."""
    try:
        with _position_timer_lock, _position_db_id_lock:
            timer_records = [
                {"cx": k[0], "cy": k[1], "class": k[2],
                 "start": v["start"], "last_seen": v["last_seen"], "size": v.get("size")}
                for k, v in _position_timer.items()
            ]
            db_id_records = [
                {"cx": k[0], "cy": k[1], "class": k[2],
                 "db_id": v["db_id"], "last_seen": v["last_seen"]}
                for k, v in _position_db_id.items()
            ]
        payload = {"position_timer": timer_records, "position_db_id": db_id_records}
        tmp_path = POSITION_STATE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, POSITION_STATE_FILE)  # replace atomik, tidak akan corrupt kalau mati di tengah
    except Exception as e:
        log.warning(f"[POSITION STATE] Gagal menyimpan checkpoint: {e}")

def _load_position_state():
    """Muat kembali _position_timer + _position_db_id dari checkpoint disk
    (kalau ada) SEBELUM main() mulai. Ini yang bikin identitas barang tahan
    restart proses. Entry yang sudah lewat POSITION_TIMER_GRACE_PERIOD (3
    hari) langsung dibuang saat load supaya tidak menghidupkan lagi barang
    yang memang sudah lama tidak keliatan."""
    if not os.path.exists(POSITION_STATE_FILE):
        return
    try:
        with open(POSITION_STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        now = time.time()
        loaded_timer = 0
        loaded_db_id = 0
        with _position_timer_lock:
            for rec in payload.get("position_timer", []):
                if now - rec["last_seen"] > POSITION_TIMER_GRACE_PERIOD:
                    continue
                key = (rec["cx"], rec["cy"], rec["class"])
                size = tuple(rec["size"]) if rec.get("size") else None
                _position_timer[key] = {"start": rec["start"], "last_seen": rec["last_seen"], "size": size}
                loaded_timer += 1
        with _position_db_id_lock:
            for rec in payload.get("position_db_id", []):
                if now - rec["last_seen"] > POSITION_TIMER_GRACE_PERIOD:
                    continue
                key = (rec["cx"], rec["cy"], rec["class"])
                _position_db_id[key] = {"db_id": rec["db_id"], "last_seen": rec["last_seen"]}
                loaded_db_id += 1
        log.info(f"[POSITION STATE] Checkpoint dimuat dari {POSITION_STATE_FILE}: "
                 f"{loaded_timer} timer posisi, {loaded_db_id} mapping db_id "
                 f"(mencegah barang lama dianggap baru setelah restart)")
    except Exception as e:
        log.warning(f"[POSITION STATE] Gagal memuat checkpoint ({e}), mulai dari kosong")

_position_db_id      = {}
_position_db_id_lock = threading.Lock()

# Muat checkpoint SEKARANG (saat modul di-import / proses baru mulai),
# sebelum main() pertama kali jalan.
_load_position_state()

def get_position_db_id(cx, cy, class_name):
    with _position_db_id_lock:
        key = _find_nearby_key(_position_db_id, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is None:
            return None
        data = _position_db_id[key]
        if time.time() - data["last_seen"] > POSITION_TIMER_GRACE_PERIOD:
            del _position_db_id[key]
            return None
        return data["db_id"]

def set_position_db_id(cx, cy, class_name, db_id):
    with _position_db_id_lock:
        key = _find_nearby_key(_position_db_id, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is None:
            key = (cx, cy, class_name)
        _position_db_id[key] = {"db_id": db_id, "last_seen": time.time()}
    _save_position_state()

def touch_position_db_id(cx, cy, class_name):
    with _position_db_id_lock:
        key = _find_nearby_key(_position_db_id, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is not None:
            _position_db_id[key]["last_seen"] = time.time()

def clear_position_db_id(cx, cy, class_name):
    """Dipanggil HANYA saat barang benar-benar dianggap keluar (cleanup tracker,
    miss_count > MISS_TOLERANCE). Ini satu-satunya jalur yang boleh mereset
    identitas posisi ini, supaya barang baru yang muncul di titik yang sama
    nanti dianggap objek baru (baris DB baru)."""
    with _position_db_id_lock:
        key = _find_nearby_key(_position_db_id, cx, cy, class_name, POSITION_MATCH_RADIUS)
        if key is not None:
            _position_db_id.pop(key, None)
    _save_position_state()

# ======================================================================
# CLEANUP
# ======================================================================
def cleanup_old_entries():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        now = time.time()
        with _alert_history_lock:
            old_keys = [k for k, v in _alert_history.items() if now - v > MAX_HISTORY_AGE]
            for k in old_keys:
                del _alert_history[k]
        with _position_timer_lock:
            old_keys = [k for k, v in _position_timer.items() if now - v["last_seen"] > MAX_HISTORY_AGE]
            for k in old_keys:
                del _position_timer[k]
        with _position_db_id_lock:
            old_keys = [k for k, v in _position_db_id.items() if now - v["last_seen"] > MAX_HISTORY_AGE]
            for k in old_keys:
                del _position_db_id[k]
        # Checkpoint berkala ke disk (jaring pengaman tambahan selain checkpoint
        # langsung di set_position_db_id/clear_position_db_id) -- kalau ada
        # perubahan lain yg belum ke-flush, paling lama basi CLEANUP_INTERVAL detik.
        _save_position_state()
        gc.collect()

# ======================================================================
# HEARTBEAT
# ======================================================================
_start_time = time.time()

def heartbeat_worker():
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        try:
            resp = requests.post(
                N8N_HEARTBEAT_URL,
                data={
                    "status":    "alive",
                    "camera":    CAMERA_NAME,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "uptime":    int(time.time() - _start_time),
                },
                timeout=10
            )
            if resp.status_code == 200:
                log.info(f"[HEARTBEAT] OK — uptime {int(time.time() - _start_time)}s")
            else:
                log.warning(f"[HEARTBEAT] Respons tidak normal: {resp.status_code}")
        except Exception as e:
            log.warning(f"[HEARTBEAT] Gagal kirim: {e}")

# ======================================================================
# KAMERA OFFLINE MONITOR
# ======================================================================
_last_frame_time  = [time.time()]
_offline_notified = [False]

def check_kamera_offline():
    while True:
        time.sleep(10)
        offline_duration = time.time() - _last_frame_time[0]
        if offline_duration > OFFLINE_THRESHOLD and not _offline_notified[0]:
            _offline_notified[0] = True
            log.warning(f"[OFFLINE] {CAMERA_NAME} offline {int(offline_duration)}s — mengirim notifikasi...")
            try:
                requests.post(
                    N8N_WEBHOOK_URL,
                    data={
                        "class_name":   "KAMERA OFFLINE",
                        "duration":     int(offline_duration),
                        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "camera":       CAMERA_NAME,
                        "image_base64": ""
                    },
                    timeout=10
                )
            except Exception as e:
                log.error(f"[OFFLINE] Gagal kirim notif: {e}")
            simpan_log_offline(CAMERA_NAME, offline_duration)
        elif offline_duration <= OFFLINE_THRESHOLD:
            if _offline_notified[0]:
                log.info(f"[OFFLINE] {CAMERA_NAME} kembali online")
                try:
                    requests.post(
                        N8N_WEBHOOK_URL,
                        data={
                            "class_name":   "KAMERA ONLINE",
                            "duration":     0,
                            "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "camera":       CAMERA_NAME,
                            "image_base64": ""
                        },
                        timeout=10
                    )
                except Exception as e:
                    log.error(f"[OFFLINE] Gagal kirim notif online: {e}")
                simpan_log_online(CAMERA_NAME)
            _offline_notified[0] = False

# ======================================================================
# ALERT QUEUE & WORKER
# ======================================================================
_alert_queue = queue.Queue(maxsize=50)

def _alert_worker():
    while True:
        try:
            task = _alert_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        (webhook_url, obj_id, class_name, duration,
         timestamp, cam_name, img_path, callback) = task

        success = False

        for attempt in range(ALERT_MAX_RETRY):
            try:
                log.info(f"[ALERT SEND] Percobaan {attempt + 1}/{ALERT_MAX_RETRY} "
                         f"| class={class_name} | id={obj_id} | duration={int(duration)}s")

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                resp = requests.post(
                    webhook_url,
                    data={
                        "id":           int(obj_id),
                        "class_name":   class_name,
                        "duration":     int(duration),
                        "timestamp":    timestamp,
                        "camera":       cam_name,
                        "image_base64": img_b64,
                    },
                    timeout=ALERT_TIMEOUT,
                )

                log.info(f"[ALERT SENT] ID:{obj_id} CLASS:{class_name} "
                         f"HTTP {resp.status_code} "
                         f"(percobaan {attempt + 1})")
                success = True
                break

            except requests.exceptions.Timeout:
                log.warning(f"[ALERT TIMEOUT] Percobaan {attempt + 1}/{ALERT_MAX_RETRY} "
                            f"timeout setelah {ALERT_TIMEOUT}s — "
                            f"{'retry...' if attempt + 1 < ALERT_MAX_RETRY else 'menyerah.'}")
                if attempt + 1 < ALERT_MAX_RETRY:
                    time.sleep(ALERT_RETRY_DELAY)

            except requests.exceptions.ConnectionError as e:
                log.error(f"[ALERT ERROR] Koneksi gagal (n8n mati?): {e}")
                break

            except Exception as e:
                log.error(f"[ALERT ERROR] Error tidak terduga: {e}")
                break

        callback(success=success)

        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            log.warning(f"[ALERT] Gagal hapus file temp: {e}")

        time.sleep(ALERT_SEND_DELAY)


def prepare_alert_image(frame, max_dim=200, quality=50):
    """
    Mengecilkan gambar alert menjadi sangat kecil agar base64 pendek.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer


def queue_alert(webhook_url, tkey, obj_id, class_name, duration, frame, callback):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img_path  = f"alert_{obj_id}_{int(time.time_ns())}.jpg"

    # Simpan gambar yang SUDAH DIKECILKAN
    try:
        small_buffer = prepare_alert_image(frame, max_dim=240, quality=60)
        with open(img_path, 'wb') as f:
            f.write(small_buffer)
    except Exception as e:
        log.error(f"[ALERT] Gagal menyimpan gambar kecil: {e}")
        # Fallback: resize sangat kecil
        tiny = cv2.resize(frame, (40, 30))
        cv2.imwrite(img_path, tiny, [cv2.IMWRITE_JPEG_QUALITY, 15])

    try:
        _alert_queue.put_nowait((
            webhook_url, obj_id, class_name, duration,
            timestamp, CAMERA_NAME, img_path, callback
        ))
    except queue.Full:
        log.warning("[ALERT] Queue penuh, alert dilewati")
        if os.path.exists(img_path):
            os.remove(img_path)


# ======================================================================
# UTILITIES
# ======================================================================
class FrameGrabber:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = False
        self.reconnect_count = 0
        self.last_frame_time = time.time()

    def start(self):
        self.cap = open_stream(self.source)
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.reconnect_count += 1
                if self.reconnect_count > MAX_RECONNECT:
                    log.error("[GRABBER] Gagal reconnect setelah max percobaan")
                    time.sleep(5)
                    self.reconnect_count = 0
                    continue
                log.warning(f"[GRABBER] Reconnect percobaan {self.reconnect_count}/{MAX_RECONNECT}...)")
                if self.cap is not None:
                    self.cap.release()
                time.sleep(min(2 * self.reconnect_count, 10))
                self.cap = open_stream(self.source, self.reconnect_count)
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.reconnect_count += 1
                log.warning(f"[GRABBER] Frame gagal dibaca (percobaan{self.reconnect_count})")
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue
            
            self.reconnect_count = 0
            self.last_frame_time = time.time()

            with self.lock:
                self.frame = frame
                self.ret = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


def open_stream(source, attempt=0):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        log.warning(f"[STREAM] Gagal dibuka (percobaan {attempt + 1})")
    return cap


def is_inside_roi_polygon(bx1, by1, bx2, by2, polygon):
    cx = int((bx1 + bx2) / 2)
    cy = by2
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0


# ======================================================================
# FILTER UKURAN/RASIO BOX (ADAPTIF) — mengurangi false positive tanpa retrain
# ======================================================================
# Ide: box/pallet asli di ROI ini punya rentang ukuran yang relatif konsisten
# (kamera diam, jarak objek ke lensa tidak jauh berubah). Deteksi yang jauh
# di luar rentang ukuran box yang PERNAH benar-benar terkonfirmasi (misal
# bayangan tipis-memanjang, pantulan lampu, atau potongan objek lain) bisa
# dibuang sebelum sempat masuk tracker. Threshold-nya BELAJAR OTOMATIS dari
# histori box yang sudah lolos konfirmasi, jadi tidak perlu diukur manual
# dan otomatis menyesuaikan kalau ROI/kamera berubah.
_confirmed_box_sizes      = []
_confirmed_box_sizes_lock = threading.Lock()
SIZE_HISTORY_MAX_SAMPLES  = 500   # jumlah sampel maksimal yang disimpan
SIZE_FILTER_MIN_SAMPLES   = 30    # filter ukuran baru aktif setelah histori cukup
SIZE_FILTER_PERCENTILE_LOW  = 5
SIZE_FILTER_PERCENTILE_HIGH = 95
SIZE_FILTER_MARGIN          = 0.35  # toleransi tambahan 35% di luar rentang percentile

# Rasio lebar:tinggi box/pallet wajar — berlaku dari awal (tidak perlu histori),
# menyaring bentuk yang jelas aneh (terlalu panjang/tipis khas bayangan).
ASPECT_RATIO_MIN = 0.15
ASPECT_RATIO_MAX = 6.5

def record_confirmed_box_size(bw, bh):
    """Dipanggil setiap sebuah box LOLOS konfirmasi -> jadi 'contoh' ukuran wajar."""
    with _confirmed_box_sizes_lock:
        _confirmed_box_sizes.append((bw, bh))
        if len(_confirmed_box_sizes) > SIZE_HISTORY_MAX_SAMPLES:
            _confirmed_box_sizes.pop(0)

def is_size_plausible(bw, bh):
    """True kalau ukuran box masuk akal dibanding histori box yang sudah terkonfirmasi."""
    if bw <= 0 or bh <= 0:
        return False

    ratio = bw / bh
    if ratio < ASPECT_RATIO_MIN or ratio > ASPECT_RATIO_MAX:
        return False

    with _confirmed_box_sizes_lock:
        n = len(_confirmed_box_sizes)
        if n < SIZE_FILTER_MIN_SAMPLES:
            return True  # histori belum cukup -> jangan filter dulu, biar tidak salah buang
        areas = [w * h for w, h in _confirmed_box_sizes]

    area   = bw * bh
    lo     = float(np.percentile(areas, SIZE_FILTER_PERCENTILE_LOW))
    hi     = float(np.percentile(areas, SIZE_FILTER_PERCENTILE_HIGH))
    lo_adj = lo * (1 - SIZE_FILTER_MARGIN)
    hi_adj = hi * (1 + SIZE_FILTER_MARGIN)
    return lo_adj <= area <= hi_adj


def box_overlap_ratio(box_a, box_b):
    """Persentase area box_a yang ketutup box_b (bukan IoU biasa -> dipakai
    karena kepala orang biasanya lebih kecil dari bbox 'box', jadi yang
    relevan adalah seberapa besar area BOX terpakan oleh person, bukan IoU
    dua-duanya."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter_area / area_a


def filter_person_false_positives(all_detections, person_boxes, overlap_threshold):
    """Buang deteksi 'box' yang overlap signifikan dengan deteksi 'person'."""
    if not person_boxes:
        return all_detections

    filtered = []
    for det in all_detections:
        bx1, by1, bx2, by2, class_name, conf, tag, oid = det
        box_coords = (bx1, by1, bx2, by2)
        max_overlap = max(
            (box_overlap_ratio(box_coords, pbox) for pbox in person_boxes),
            default=0.0
        )
        if max_overlap >= overlap_threshold:
            log.debug(f"[PERSON FILTER] {class_name} conf={conf:.2f} box={box_coords} "
                      f"dibuang, overlap {max_overlap:.2f} dengan deteksi person")
            continue
        filtered.append(det)
    return filtered


def find_pos_tracker(pos_trackers, cx, cy, bw, bh, threshold, size_threshold=0.4):
    best_key, best_dist = None, float("inf")
    for key, data in pos_trackers.items():
        px, py = data["pos"]
        pw, ph = data.get("base_size", data.get("size", (bw, bh)))
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        adaptive_threshold = max(threshold, 0.25 * max(pw, ph))
        size_diff_w = abs(bw - pw) / max(pw, 1)
        size_diff_h = abs(bh - ph) / max(ph, 1)
        if size_diff_w > size_threshold or size_diff_h > size_threshold:
            continue
        if dist < adaptive_threshold and dist < best_dist:
            best_dist = dist
            best_key  = key
    return best_key


# ======================================================================
# MONITOR 22 KAMERA
# ======================================================================
camera_status = {
    cam["name"]: {
        "online":     True,
        "last_seen":  time.time(),
        "last_notif": 0
    }
    for cam in CAMERAS
}

import socket
from urllib.parse import urlparse

def cek_kamera(camera):
    try:
        parsed = urlparse(camera["url"])
        host = parsed.hostname
        port = parsed.port or 554
        with socket.create_connection((host, port), timeout=3):
            return True
    except Exception:
        return False

def kirim_notif_offline(camera_name, duration):
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            data={
                "class_name":   "KAMERA OFFLINE",
                "duration":     int(duration),
                "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera":       camera_name,
                "image_base64": ""
            },
            timeout=10
        )
        log.info(f"[MONITOR] {camera_name} offline notif HTTP {response.status_code}")
    except Exception as e:
        log.error(f"[MONITOR ERROR] {camera_name}: {e}")

def monitor_kamera(camera):
    name = camera["name"]
    log.info(f"[MONITOR] Mulai monitor {name}")
    while True:
        online       = cek_kamera(camera)
        current_time = time.time()
        status       = camera_status[name]
        if online:
            if not status["online"]:
                log.info(f"[MONITOR] {name} kembali online")
                try:
                    requests.post(
                        N8N_WEBHOOK_URL,
                        data={
                            "class_name":   "KAMERA ONLINE",
                            "duration":     0,
                            "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "camera":       name,
                            "image_base64": ""
                        },
                        timeout=10
                    )
                except Exception as e:
                    log.error(f"[MONITOR ERROR] Gagal kirim notif online {name}: {e}")
                simpan_log_online(name)
            status["online"]    = True
            status["last_seen"] = current_time
        else:
            was_online          = status["online"]
            status["online"]    = False
            offline_duration    = current_time - status["last_seen"]
            log.warning(f"[MONITOR] {name} offline {int(offline_duration)}s")
            if offline_duration >= MONITOR_OFFLINE_THRESHOLD:
                if current_time - status["last_notif"] >= NOTIF_COOLDOWN:
                    kirim_notif_offline(name, offline_duration)
                    status["last_notif"] = current_time
                if was_online:
                    simpan_log_offline(name, offline_duration)
        time.sleep(CHECK_INTERVAL)

def start_monitor_alert():
    log.info(f"[MONITOR] Memulai monitor {len(CAMERAS)} kamera...")
    for camera in CAMERAS:
        t = threading.Thread(target=monitor_kamera, args=(camera,), daemon=True)
        t.start()
        time.sleep(2)

    def status_loop():
        while True:
            time.sleep(60)
            online_count = sum(1 for s in camera_status.values() if s["online"])
            log.info(f"[MONITOR STATUS] {online_count}/{len(CAMERAS)} kamera online")
    threading.Thread(target=status_loop, daemon=True).start()


# ======================================================================
# MAIN
# ======================================================================
def main():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|buffer_size;1"
    log.info("=" * 60)
    log.info(f"[START] {CAMERA_NAME} — {datetime.datetime.now()}")
    log.info(f"[CONFIG] IMG_SIZE={IMG_SIZE} | CONF_THRESHOLD={CONF_THRESHOLD}")
    log.info("=" * 60)

    threading.Thread(target=_alert_worker,        daemon=True).start()
    threading.Thread(target=check_kamera_offline, daemon=True).start()
    threading.Thread(target=cleanup_old_entries,  daemon=True).start()
    threading.Thread(target=heartbeat_worker,     daemon=True).start()
    start_monitor_alert()

    log.info("[INFO] Memuat model YOLO...")
    if os.path.exists(MAIN_MODEL_PATH):
        model = YOLO(MAIN_MODEL_PATH, task="detect")
        log.info(f"[INFO] Menggunakan model OpenVINO: {MAIN_MODEL_PATH}")
    else:
        model = YOLO("yolov8n.pt")
        log.info(f"[INFO] Path model OpenVINO '{MAIN_MODEL_PATH}' tidak ditemukan — "
                 f"menggunakan model PyTorch fallback (yolov8n.pt)")

    log.info(f"[INFO] Kelas tersedia: {list(model.names.values())}")

    # Person Filter
    person_model = None
    person_cls_id = 0 # COCO: person = class 0

    if PERSON_FILTER_ENABLED:
        try:
            if os.path.exists(PERSON_MODEL_PATH):
                person_model = YOLO(PERSON_MODEL_PATH, task="detect")
                log.info(f"[INFO] Person model (OpenVINO custom) dimuat: {PERSON_MODEL_PATH}")
            else:
                # Folder OpenVINO custom belum ada/valid -> fallback ke model
                # pretrained standar ultralytics (auto-download kalau belum ada
                # di cache lokal). Model ini punya kelas COCO lengkap dengan
                # 'person' di index 0, jadi cukup untuk keperluan filter ini
                # walau tidak seringan versi OpenVINO custom.
                log.warning(f"[PERSON FILTER] Path '{PERSON_MODEL_PATH}' tidak ditemukan — "
                            f"fallback ke model pretrained yolov8n.pt untuk filter person")
                person_model = YOLO("yolov8n.pt")

            log.info(f"[INFO] Person model classes: {list(person_model.names.values())}")

            # Pastikan class 0 memang person
            if person_model.names.get(0) != "person":
                log.warning("[PERSON FILTER] Class 0 bukan 'person'. Filter dinonaktifkan.")
                person_model = None

        except Exception as e:
            # PENTING: kalau person model gagal total dimuat, JANGAN sampai ini
            # menjatuhkan main() (lihat catatan lama di sekitar predict() di bawah
            # soal kenapa itu berbahaya untuk state tracker). Cukup nonaktifkan
            # filter untuk sesi ini.
            log.error(f"[PERSON FILTER] Gagal memuat model: {e}")
            person_model = None

    grabber = FrameGrabber(CCTV_SOURCE).start()
    pos_trackers     = {}
    _pos_key_counter = [0]
    frame_counter    = 0

    def next_key():
        _pos_key_counter[0] += 1
        return _pos_key_counter[0]

    log.info("[INFO] Sistem Pengawasan Aktif.")
    track_call_count = 0
    RESET_TRACKER_EVERY = 10000

    while True:
        ret, frame = grabber.read()
        if not ret or frame is None:
            time.sleep(0.5)
            continue
        
        _last_frame_time[0] = time.time()
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        current_time = time.time()
        h, w = frame.shape[:2]

        crop_x1 = max(0, ROI_BBOX_X1 - ROI_CROP_PADDING)
        crop_y1 = max(0, ROI_BBOX_Y1 - ROI_CROP_PADDING)
        crop_x2 = min(w, ROI_BBOX_X2 + ROI_CROP_PADDING)
        crop_y2 = min(h, ROI_BBOX_Y2 + ROI_CROP_PADDING)
        roi_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        roi_preview = roi_frame.copy()

        t0 = time.time()
        results = model.track(
            roi_frame,
            persist=True,
            tracker="bytetrack.yaml",
            device="CPU",
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            agnostic_nms=True,
            max_det=300,
            augment=False,
            verbose=False,
        )
        log.info(f"[TIMING] model.track selesai dalam {time.time()-t0:.2f}s | roi_frame shape={roi_frame.shape}")

        all_detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_m = results[0].boxes.xyxy.cpu().numpy()
            cls_m   = results[0].boxes.cls.int().cpu().numpy()
            conf_m  = results[0].boxes.conf.cpu().numpy()
            ids_m   = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else [None] * len(boxes_m)

            log.info(f"[DETEKSI MENTAH] Jumlah: {len(boxes_m)}")
            for box, cls, conf, oid in zip(boxes_m, cls_m, conf_m, ids_m):
                bx1, by1, bx2, by2 = map(int, box)
                bx1 += crop_x1; bx2 += crop_x1
                by1 += crop_y1; by2 += crop_y1
                class_name = model.names[cls]
                log.info(f"  - {class_name} conf={conf:.2f} box=({bx1},{by1},{bx2},{by2}) oid={oid}")
                all_detections.append((bx1, by1, bx2, by2, class_name, float(conf), "main", oid))
        else:
            log.info("[DETEKSI MENTAH] Tidak ada deteksi")
        del results

        # ---- FILTER PERSON: buang deteksi "box" yang sebenarnya kepala/badan orang ----
        if person_model is not None and all_detections:
            try:
                person_results = person_model.predict(
                    roi_frame,
                    classes=[0],
                    conf=PERSON_CONF_THRESHOLD,
                    device="CPU",
                    imgsz=IMG_SIZE,
                    verbose=False,
                )
                person_boxes = []
                if person_results[0].boxes is not None and len(person_results[0].boxes) > 0:
                    pboxes_m = person_results[0].boxes.xyxy.cpu().numpy()
                    for pbox in pboxes_m:
                        px1, py1, px2, py2 = map(int, pbox)
                        px1 += crop_x1; px2 += crop_x1
                        py1 += crop_y1; py2 += crop_y1
                        person_boxes.append((px1, py1, px2, py2))

                if person_boxes:
                    jumlah_sebelum = len(all_detections)
                    all_detections = filter_person_false_positives(
                        all_detections, person_boxes, PERSON_OVERLAP_THRESHOLD
                    )
                    if len(all_detections) < jumlah_sebelum:
                        log.info(f"[PERSON FILTER] {jumlah_sebelum - len(all_detections)} deteksi "
                                 f"dibuang karena overlap dengan {len(person_boxes)} deteksi person")
                del person_results
            except Exception as e:
                # PENTING: model person gagal/rusak TIDAK BOLEH menjatuhkan seluruh main().
                # Kalau ini crash sampai ke supervisor, SEMUA state tracker (pos_trackers,
                # hit_count, confirmed) ikut hilang tiap restart -> box valid tidak pernah
                # sempat "confirmed". Cukup skip filter person untuk frame ini saja.
                log.error(f"[PERSON FILTER] predict() gagal, filter dilewati untuk frame ini: {e}")

        track_call_count += 1
        if track_call_count >= RESET_TRACKER_EVERY:
            track_call_count = 0
            log.info("[MAINTENANCE] Tracker state di-reset preventif")

        frame_clean = frame.copy()
        active_keys = []

        for det in all_detections:
            bx1, by1, bx2, by2, class_name, conf, _, oid = det
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2

            in_roi = is_inside_roi_polygon(bx1, by1, bx2, by2, ROI_COORDINATES)

            if KELAS_DIMONITOR is not None and class_name not in KELAS_DIMONITOR:
                continue
            if not in_roi:
                continue

            bw = bx2 - bx1
            bh = by2 - by1

            if not is_size_plausible(bw, bh):
                log.debug(f"[FILTER UKURAN] {class_name} dibuang, ukuran {bw}x{bh} di luar rentang wajar")
                continue

            if oid is None:
                oid = -1

            tkey = find_pos_tracker(pos_trackers, cx, cy, bw, bh, PIXEL_TOLERANCE)
            if tkey is None:
                tkey = next_key()
                # PENTING: belum panggil get_position_start/set_position_start di sini.
                # Timer persisten baru dimulai setelah deteksi ini terbukti konsisten
                # (lihat blok konfirmasi di bawah), supaya deteksi sekali-lewat (noise)
                # tidak langsung tercatat sebagai "box baru" di staging.
                pos_trackers[tkey] = {
                    "start": None,
                    "pos": (cx, cy),
                    "size": (bw, bh),
                    "base_size": (bw, bh),
                    "alerted": False,
                    "miss_count": 0,
                    "hit_count": 1,
                    "confirmed": False,
                    "first_seen": current_time,
                    "class": class_name,
                    "box_size": (bx2 - bx1) * (by2 - by1),
                    "last_alert_durasi": 0,
                    "last_alert_time": 0,
                    "last_oid": oid,
                    "db_id": None,
                    "escalation_count": 0,
                    "confirm_class": class_name,
                }
                log.info(f"[Tracker Kandidat] #{tkey} di ({cx},{cy}) hit 1/{CONFIRMATION_HITS_REQUIRED}, belum dikonfirmasi")

            tracker = pos_trackers[tkey]
            tracker["miss_count"] = 0
            tracker["class"] = class_name
            tracker["last_oid"] = oid
            tracker["pos"] = (cx, cy)
            tracker["size"] = (bw, bh)
            tracker["box_size"] = (bx2 - bx1) * (by2 - by1)
            active_keys.append(tkey)

            if not tracker["confirmed"]:
                if class_name != tracker.get("confirm_class"):
                    # Kelas berubah di tengah proses konfirmasi (mis. box vs pallet
                    # ketuker) -> model masih ragu, jangan lolos, hitung ulang dari 1.
                    log.info(f"[Tracker Kandidat] #{tkey} kelas berubah "
                             f"({tracker.get('confirm_class')} -> {class_name}), hitung ulang konfirmasi")
                    tracker["confirm_class"] = class_name
                    tracker["hit_count"] = 1
                    continue

                tracker["hit_count"] += 1
                if tracker["hit_count"] < CONFIRMATION_HITS_REQUIRED:
                    log.info(f"[Tracker Kandidat] #{tkey} hit {tracker['hit_count']}/{CONFIRMATION_HITS_REQUIRED}, menunggu konfirmasi")
                    continue

                # Ambang konfirmasi tercapai -> baru sekarang dianggap box valid.
                existing_start = get_position_start(cx, cy, class_name, bw, bh)
                if existing_start is None:
                    start_time = tracker["first_seen"]
                    set_position_start(cx, cy, class_name, start_time, bw, bh)
                    log.info(f"[Tracker Terkonfirmasi] #{tkey} di ({cx},{cy}) timer dimulai dari deteksi pertama")
                else:
                    start_time = existing_start
                    log.info(f"[Tracker Revived] #{tkey} melanjutkan timer (sudah diam {int(time.time() - start_time)}s)")

                tracker["start"] = start_time
                tracker["confirmed"] = True
                record_confirmed_box_size(bw, bh)
                log.info(f"[Tracker Terkonfirmasi] #{tkey} total tracker aktif: {len(pos_trackers)}")
                # CATATAN: TIDAK insert ke DB di sini. Baris DB baru dibuat nanti
                # kalau objek ini benar-benar mencapai WAKTU_MAKSIMAL_DIAM (lihat
                # blok eskalasi di bawah) -> dashboard cuma nampilin barang yang
                # sudah diam >= 3 hari, bukan semua box yang lewat sebentar di ROI.

            touch_position_start(cx, cy, class_name, bw, bh)
            if tracker["db_id"] is not None:
                touch_position_db_id(cx, cy, class_name)
            durasi = current_time - tracker["start"]

            if durasi >= WAKTU_MAKSIMAL_DIAM:
                last_durasi = tracker["last_alert_durasi"]
                next_threshold = None
                for threshold in ESCALATION_THRESHOLDS:
                    if threshold > last_durasi and durasi >= threshold:
                        next_threshold = threshold
                        break

                if next_threshold is not None:
                    if is_duplicate_alert(cx, cy, class_name):
                        continue

                    log.warning(f"[ALERT] {class_name} tkey:{tkey} diam {int(durasi)}s — melewati ambang {int(next_threshold/3600)} jam!")
                    set_last_alert_time(cx, cy, class_name)
                    tracker["last_alert_durasi"] = next_threshold
                    tracker["last_alert_time"] = current_time

                    padding = 60
                    x1c = max(0, cx - padding)
                    y1c = max(0, cy - padding)
                    x2c = min(w, cx + padding)
                    y2c = min(h, cy + padding)

                    if frame_clean is None:
                        continue
                    if x2c <= x1c or y2c <= y1c:
                        alert_frame = frame_clean.copy()
                    else:
                        alert_frame = frame_clean[y1c:y2c, x1c:x2c].copy()

                    cv2.rectangle(alert_frame,
                                  (bx1 - x1c, by1 - y1c),
                                  (bx2 - x1c, by2 - y1c),
                                  (0, 255, 255), 2)

                    _cx, _cy, _cn = cx, cy, class_name

                    def make_callback(acx, acy, acn):
                        def cb(success):
                            if not success:
                                set_last_alert_time(acx, acy, acn, 0)
                                tracker["last_alert_durasi"] = 0
                                tracker["last_alert_time"] = 0
                        return cb

                    # ---- baris DB baru dibuat DI SINI (baru capai ambang alert),
                    # bukan pas konfirmasi awal -> dashboard cuma nampilin barang
                    # yang beneran sudah diam >= WAKTU_MAKSIMAL_DIAM ----
                    if tracker["db_id"] is None:
                        existing_db_id = get_position_db_id(cx, cy, class_name)
                        if existing_db_id is not None:
                            # Posisi fisik ini sudah punya baris DB yg masih terbuka
                            # (tracker sempat pecah krn kedip/goyang ukuran) -> PAKAI
                            # LAGI baris yang sama, jangan insert baru.
                            tracker["db_id"] = existing_db_id
                            log.info(f"[DB] Tracker #{tkey} pakai kembali baris id={existing_db_id} "
                                     f"(posisi sama, mencegah duplikat di dashboard)")
                        else:
                            entry_dt = datetime.datetime.fromtimestamp(tracker["start"])
                            log.info(f"[BARANG MASUK] #{tkey} {class_name} pertama terdeteksi "
                                     f"{entry_dt.strftime('%Y-%m-%d %H:%M:%S')}, mencapai ambang alert sekarang")
                            tracker["db_id"] = db_write_with_retry(
                                insert_staging_row, CAMERA_NAME, class_name, entry_dt
                            )
                            if tracker["db_id"] is not None:
                                set_position_db_id(cx, cy, class_name, tracker["db_id"])
                            else:
                                log.error(f"[DB] Gagal insert baris staging untuk #{tkey} "
                                          f"setelah {DB_WRITE_RETRY}x percobaan, foto/alert kali ini "
                                          f"kemungkinan tidak akan tersimpan ke DB")

                    # ---- update baris DB yang sama (bukan bikin baris baru): isi alert_sent_N ----
                    tracker["escalation_count"] += 1
                    alert_dt = datetime.datetime.fromtimestamp(current_time)
                    _, alert_foto_buf = cv2.imencode('.jpg', alert_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    escalation_ok = db_write_with_retry(
                        update_alert_escalation,
                        tracker["db_id"], alert_dt, durasi, tracker["escalation_count"],
                        foto_bytes=alert_foto_buf.tobytes(),
                        file_name=f"{class_name}_{tkey}_alert{tracker['escalation_count']}.jpg"
                    )
                    if not escalation_ok:
                        log.error(f"[DB] Gagal simpan eskalasi/foto id={tracker['db_id']} "
                                  f"setelah {DB_WRITE_RETRY}x percobaan")

                    log.info(f"[ALERT] Mengirim duration={int(durasi)}s ke n8n (db_id={tracker['db_id']})")
                    queue_alert(N8N_WEBHOOK_URL, tkey, tracker["db_id"], class_name,
                                durasi, alert_frame,
                                make_callback(_cx, _cy, _cn))

        # Cleanup tracker
        for old_key in list(pos_trackers.keys()):
            if old_key not in active_keys:
                pos_trackers[old_key]["miss_count"] += 1
                if pos_trackers[old_key]["miss_count"] > MISS_TOLERANCE:
                    old_data = pos_trackers[old_key]
                    log.warning(f"[TRACKER HAPUS] #{old_key} class={old_data['class']} hilang > {MISS_TOLERANCE_SECONDS}s")

                    # ---- catat waktu barang keluar (tanggal & jam nyata) ----
                    if old_data.get("confirmed") and old_data.get("db_id") is not None:
                        exit_dt     = datetime.datetime.now()
                        durasi_diam = current_time - old_data["start"]
                        entry_dt    = datetime.datetime.fromtimestamp(old_data["start"])
                        log.info(f"[BARANG KELUAR] #{old_key} {old_data['class']} keluar pada "
                                 f"{exit_dt.strftime('%Y-%m-%d %H:%M:%S')} "
                                 f"(masuk: {entry_dt.strftime('%Y-%m-%d %H:%M:%S')}, lama diam: {int(durasi_diam)}s)")
                        db_write_with_retry(update_exit_status, old_data["db_id"], exit_dt, durasi_diam)
                        # CATATAN: notif_barang_keluar() SENGAJA tidak dipanggil, alasan sama
                        # dengan di blok BARANG MASUK di atas (lihat catatan di sana).
                        ex_cx, ex_cy = old_data["pos"]
                        clear_position_db_id(ex_cx, ex_cy, old_data["class"])

                    del pos_trackers[old_key]

        # ============ LIVE PREVIEW ============
        if SHOW_PREVIEW:
            display = roi_preview.copy()
            off_x, off_y = crop_x1, crop_y1

            for tkey, tkr in pos_trackers.items():
                tx, ty = tkr["pos"]
                bw, bh = tkr["size"]
                cls = tkr["class"]
                if tkr["start"] is None:
                    label = f"#{tkey} {cls} (menunggu konfirmasi {tkr['hit_count']}/{CONFIRMATION_HITS_REQUIRED})"
                    box_color = (0, 165, 255)  # oranye = belum dikonfirmasi
                else:
                    dur = current_time - tkr["start"]
                    label = f"#{tkey} {cls} {int(dur)}s"
                    box_color = (0, 255, 255)  # kuning = terkonfirmasi
                bx1 = tx - bw//2 - off_x
                by1 = ty - bh//2 - off_y
                bx2 = tx + bw//2 - off_x
                by2 = ty + bh//2 - off_y
                cv2.rectangle(display, (bx1, by1), (bx2, by2), box_color, 2)
                cv2.putText(display, label, (bx1, by1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            h_disp, w_disp = display.shape[:2]
            if w_disp > PREVIEW_WIDTH:
                scale = PREVIEW_WIDTH / w_disp
                display = cv2.resize(display, (int(w_disp*scale), int(h_disp*scale)))

            cv2.imshow("CCTV ROI - " + CAMERA_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                log.info("[PREVIEW] Keluar...")
                break

        del all_detections
        del frame_clean

        if frame_counter % (300 * FRAME_SKIP) == 0:
            uptime_jam = (time.time() - _start_time) / 3600
            log.info(f"[STATUS] Uptime:{uptime_jam:.1f}j | Tracker:{len(pos_trackers)} | AlertQueue:{_alert_queue.qsize()}")

    grabber.stop()
    cv2.destroyAllWindows()
    log.info("[STOP] Program selesai.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        from calibrate import calibrate_roi
        calibrate_roi(CCTV_SOURCE)
    else:
        while True:
            try:
                main()
                log.warning("[SUPERVISOR] main() keluar tanpa exception. Restart dalam 10 detik...")
            except Exception as e:
                log.error(f"[SUPERVISOR] main CRASH: {e}")
                log.error("[SUPERVISOR] Traceback lengkap \n" + traceback.format_exc())
            time.sleep(10)