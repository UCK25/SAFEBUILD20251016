
import sqlite3
import os
import json
import hashlib
import binascii
from datetime import datetime
from config import DB_PATH

# Password hashing settings
_HASH_NAME = 'sha256'
_HASH_ITERATIONS = 100_000

def _hash_password(password: str) -> str:
    """Return a string salt$hash_hex using PBKDF2-HMAC."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(_HASH_NAME, password.encode('utf-8'), salt, _HASH_ITERATIONS)
    return binascii.hexlify(salt).decode() + '$' + binascii.hexlify(dk).decode()

def _verify_password(stored: str, provided: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split('$')
        salt = binascii.unhexlify(salt_hex)
        dk = hashlib.pbkdf2_hmac(_HASH_NAME, provided.encode('utf-8'), salt, _HASH_ITERATIONS)
        return binascii.hexlify(dk).decode() == hash_hex
    except Exception:
        return False

def init_db():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabla Usuarios (CU-001 a CU-006)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'supervisor',
            qr_code TEXT UNIQUE
        )
    ''')
    
    # Tabla Cámaras (CU-007 a CU-011)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            camera_index INTEGER,
            config TEXT
        )
    ''')
    
    # Tabla Incidentes (CU-014, CU-017, CU-019 a CU-021)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT,
            type TEXT,
            timestamp TEXT,
            description TEXT,
            status TEXT DEFAULT 'open',
            evidence_path TEXT,
            user_identified TEXT
        )
    ''')

    # Ensure schema has columns for occurrences and last_seen (added later for dedupe/aggregation)
    cursor.execute("PRAGMA table_info(incidents)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'occurrences' not in cols:
        try:
            cursor.execute("ALTER TABLE incidents ADD COLUMN occurrences INTEGER DEFAULT 1")
        except Exception:
            pass
    if 'last_seen' not in cols:
        try:
            cursor.execute("ALTER TABLE incidents ADD COLUMN last_seen TEXT")
        except Exception:
            pass

    # Tabla de auditoría para acciones administrativas (C, p.e. reset de contraseñas)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            performed_by TEXT,
            action TEXT,
            target_user_id INTEGER,
            details TEXT
        )
    ''')
    
    # Tabla para registro de descargas de QR
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qr_downloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            downloaded_by INTEGER NOT NULL,
            target_user_id INTEGER NOT NULL,
            download_path TEXT,
            FOREIGN KEY(downloaded_by) REFERENCES users(id),
            FOREIGN KEY(target_user_id) REFERENCES users(id)
        )
    ''')
    
    # Insertar usuario por defecto si no existe (con password hasheada)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        hashed = _hash_password('admin123')
        cursor.execute("INSERT INTO users (username, password, role, qr_code) VALUES (?, ?, ?, ?)", 
                      ('admin', hashed, 'admin', 'ADMIN001'))
    
    conn.commit()
    conn.close()


def ensure_admin_exists(default_username='admin', default_password='admin123', default_qr='ADMIN001'):
    """
    Ensure there is at least one user with role 'admin' in the database.
    If none exists, insert a default admin with the provided password (hashed).
    This is idempotent and safe to call on server startup.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
        row = cursor.fetchone()
        if not row or row[0] == 0:
            hashed = _hash_password(default_password)
            cursor.execute("INSERT INTO users (username, password, role, qr_code) VALUES (?, ?, ?, ?)",
                           (default_username, hashed, 'admin', default_qr))
            conn.commit()
            return True
        return False
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

# CRUD Usuarios
def register_user(username, password, qr_code=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        hashed = _hash_password(password)
        cursor.execute("INSERT INTO users (username, password, qr_code) VALUES (?, ?, ?)", 
                      (username, hashed, qr_code))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def authenticate_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, password FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        stored_hash = row[3]
        if _verify_password(stored_hash, password):
            return (row[0], row[1], row[2])
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_user_by_qr(qr_code):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role FROM users WHERE qr_code=?", (qr_code,))
        user = cursor.fetchone()
        return user
    finally:
        try:
            conn.close()
        except Exception:
            pass

def list_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, qr_code FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

def get_user_by_id(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, qr_code FROM users WHERE id=?", (user_id,))
        user = cursor.fetchone()
        return user
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_password(user_id, new_password):
    # Backward-compatible update_password with audit logging support
    return reset_user_password(performed_by=None, target_user_id=user_id, new_password=new_password)


def reset_user_password(performed_by, target_user_id, new_password):
    """Reset a user's password and log the action in audit_logs.
    performed_by: identifier (username or id) of the actor, may be None for system.
    target_user_id: integer id of the user whose password is being reset.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        hashed = _hash_password(new_password)
        cursor.execute("UPDATE users SET password=? WHERE id=?", (hashed, target_user_id))
        # Insert audit log
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        actor = str(performed_by) if performed_by is not None else 'system'
        details = f'Password reset for user_id={target_user_id}'
        cursor.execute('''INSERT INTO audit_logs (timestamp, performed_by, action, target_user_id, details)
                          VALUES (?, ?, ?, ?, ?)''', (ts, actor, 'reset_password', target_user_id, details))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def delete_user(user_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_user_role(user_id, role):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def update_user(user_id, username=None, qr_code=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # build dynamic set
        fields = []
        params = []
        if username is not None:
            fields.append('username=?')
            params.append(username)
        if qr_code is not None:
            fields.append('qr_code=?')
            params.append(qr_code)
        if not fields:
            return False
        params.append(user_id)
        sql = f"UPDATE users SET {', '.join(fields)} WHERE id=?"
        cursor.execute(sql, params)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # e.g., duplicate username or qr_code
        return False
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

# CRUD Cámaras
def register_camera(name, camera_index, config=None):
    if config is None:
        config = json.dumps({"conf": 0.4})
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO cameras (name, camera_index, config) VALUES (?, ?, ?)", 
                  (name, camera_index, config))
    conn.commit()
    conn.close()

def list_cameras():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, camera_index FROM cameras")
    cameras = cursor.fetchall()
    conn.close()
    return cameras

# CRUD Incidentes
def register_incident(camera_name, incident_type, description, user_identified=None, evidence_path=None, dedupe_window_minutes: int = 60):
    """
    Registra un incidente en la BD.
    - Si existe un incidente similar (misma cámara, mismo tipo, mismo usuario) dentro de `dedupe_window_minutes`,
      actualiza el registro existente incrementando `occurrences` y actualizando `last_seen` / `evidence_path`.
    - Si no existe, inserta un nuevo registro con `occurrences=1`.
    Esto evita contar múltiples registros por la misma situación continuada.
    """
    ts_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Buscar último incidente similar
        user_cmp = user_identified or 'unknown'
        cursor.execute('''
            SELECT id, timestamp, occurrences, evidence_path
            FROM incidents
            WHERE camera_name=? AND type=? AND COALESCE(user_identified, 'unknown')=?
            ORDER BY timestamp DESC LIMIT 1
        ''', (camera_name, incident_type, user_cmp))
        last = cursor.fetchone()

        if last and dedupe_window_minutes is not None and dedupe_window_minutes >= 0:
            last_id, last_ts, last_occ, last_evidence = last
            try:
                last_dt = datetime.strptime(last_ts, '%Y-%m-%d %H:%M:%S')
                now_dt = datetime.strptime(ts_now, '%Y-%m-%d %H:%M:%S')
                delta_sec = abs((now_dt - last_dt).total_seconds())
            except Exception:
                delta_sec = None

            if delta_sec is not None and delta_sec <= dedupe_window_minutes * 60:
                # Actualizar registro existente
                new_occ = (last_occ or 1) + 1
                new_last_seen = ts_now
                # Preferir guardar evidence_path si se proporciona (actualizar), o mantener existente
                new_evidence = evidence_path or last_evidence
                cursor.execute('''
                    UPDATE incidents SET occurrences=?, last_seen=?, evidence_path=?, description=? WHERE id=?
                ''', (new_occ, new_last_seen, new_evidence, description, last_id))
                conn.commit()
                return

        # Insertar nuevo registro
        cursor.execute('''INSERT INTO incidents (camera_name, type, timestamp, description, evidence_path, user_identified, occurrences, last_seen)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                       (camera_name, incident_type, ts_now, description, evidence_path, user_identified, 1, ts_now))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

def list_incidents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incidents ORDER BY timestamp DESC")
    incidents = cursor.fetchall()
    conn.close()
    return incidents

def update_incident(incident_id, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE incidents SET status=? WHERE id=?", (status, incident_id))
    conn.commit()
    conn.close()

def update_incident_user_recent(camera_name, incident_type, user_identified, within_seconds: int = 10):
    """
    If there's a recent incident for camera_name and incident_type with unknown user
    within `within_seconds`, update its user_identified to the provided value.
    Returns True if updated, False otherwise.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp FROM incidents
            WHERE camera_name=? AND type=? AND COALESCE(user_identified, 'unknown')='unknown'
            ORDER BY timestamp DESC LIMIT 1
        ''', (camera_name, incident_type))
        row = cursor.fetchone()
        if not row:
            return False
        inc_id, ts = row
        try:
            from datetime import datetime
            last_dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            now_dt = datetime.now()
            delta_sec = abs((now_dt - last_dt).total_seconds())
        except Exception:
            delta_sec = None

        if delta_sec is not None and delta_sec <= within_seconds:
            cursor.execute('UPDATE incidents SET user_identified=? WHERE id=?', (user_identified, inc_id))
            conn.commit()
            return True
        return False
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def log_qr_download(downloaded_by_id, target_user_id, download_path):
    """QRのダウンロードを記録"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''INSERT INTO qr_downloads 
                         (timestamp, downloaded_by, target_user_id, download_path)
                         VALUES (?, ?, ?, ?)''',
                      (timestamp, downloaded_by_id, target_user_id, download_path))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_qr_downloads(user_id=None):
    """QRのダウンロード履歴を取得"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if user_id:
            cursor.execute('''
                SELECT qd.*, u1.username as downloaded_by_name, u2.username as target_name
                FROM qr_downloads qd
                JOIN users u1 ON qd.downloaded_by = u1.id
                JOIN users u2 ON qd.target_user_id = u2.id
                WHERE qd.downloaded_by = ? OR qd.target_user_id = ?
                ORDER BY qd.timestamp DESC
            ''', (user_id, user_id))
        else:
            cursor.execute('''
                SELECT qd.*, u1.username as downloaded_by_name, u2.username as target_name
                FROM qr_downloads qd
                JOIN users u1 ON qd.downloaded_by = u1.id
                JOIN users u2 ON qd.target_user_id = u2.id
                ORDER BY qd.timestamp DESC
            ''')
        return cursor.fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Reportes (CU-022 a CU-023)
import csv
from datetime import datetime, timedelta
from typing import List, Optional
import os

# Optional: report generation to PDF requires reportlab. We'll import lazily and handle missing dependency.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

def _parse_timestamp(ts_str):
    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')


def _are_incidents_similar(inc1, inc2, time_window_minutes=60):
    """同様のインシデントかどうかを判断（同一カメラ・同一タイプ・同一ユーザーかつ時間差が閾値以内）"""
    # DB schema: 0=id,1=camera_name,2=type,3=timestamp,4=description,5=status,6=evidence_path,7=user_identified
    cam1, cam2 = inc1[1], inc2[1]
    type1, type2 = inc1[2], inc2[2]
    user1, user2 = inc1[7], inc2[7]

    if cam1 != cam2 or type1 != type2 or (user1 or 'unknown') != (user2 or 'unknown'):
        return False

    time1 = _parse_timestamp(inc1[3])
    time2 = _parse_timestamp(inc2[3])
    return abs((time1 - time2).total_seconds()) <= (time_window_minutes * 60)


def generate_report(output_path='reporte_analizado.csv', time_window_minutes=60):
    """
    Genera un "reporte analizado" que agrupa detecciones repetidas del mismo
    incumplimiento (misma cámara, mismo tipo, mismo usuario) dentro de una ventana
    de tiempo (por defecto 60 minutos) en una sola entrada.
    Salida: CSV con columnas formales y legibles.
    """
    incidents = list_incidents()
    # ordenar cronológicamente ascendente para agrupar por ocurrencia temprana
    incidents_sorted = sorted(incidents, key=lambda r: _parse_timestamp(r[3]))

    grouped: List[tuple] = []
    for inc in incidents_sorted:
        if not grouped:
            grouped.append(inc)
            continue

        last = grouped[-1]
        if _are_incidents_similar(last, inc, time_window_minutes=time_window_minutes):
            # Si es similar a la última agrupada, ignoramos la nueva (no acumulamos contador)
            # Conservamos la primera aparición (last)
            continue
        else:
            grouped.append(inc)

    # Escribir CSV formal: seleccionar columnas (ID, Cámara, Tipo, Timestamp, Descripción, Status, Usuario Identificado, EvidencePath)
    rows = []
    for inc in grouped:
        # inc indices: 0=id,1=camera_name,2=type,3=timestamp,4=description,5=status,6=evidence_path,7=user_identified
        row = [inc[0], inc[1], inc[2], inc[3], inc[4], inc[5], inc[7], inc[6]]
        rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Cámara', 'Tipo', 'Timestamp', 'Descripción', 'Status', 'Usuario Identificado', 'EvidencePath'])
        writer.writerows(rows)

    print(f"Reporte analizado completado: {len(incidents)} registros originales -> {len(rows)} registros reportados en '{output_path}'")
    return output_path


def generate_report_by_period(year: Optional[int] = None, month: Optional[int] = None, output_path: str = 'reporte_analizado_periodo.csv', time_window_minutes: int = 60):
    """
    Genera un reporte analizado filtrado por año y mes (si se pasan).
    year: 4 dígitos (ej. 2025)
    month: 1-12
    """
    incidents = list_incidents()

    # Filtrar por año/mes si se especifica
    filtered = []
    for inc in incidents:
        try:
            ts = _parse_timestamp(inc[3])
        except Exception:
            continue
        if year and ts.year != int(year):
            continue
        if month and ts.month != int(month):
            continue
        filtered.append(inc)

    # Reusar generación agrupada sobre los filtrados
    # ordenar cronológicamente ascendente para agrupar por ocurrencia temprana
    incidents_sorted = sorted(filtered, key=lambda r: _parse_timestamp(r[3]))

    grouped: List[tuple] = []
    for inc in incidents_sorted:
        if not grouped:
            grouped.append(inc)
            continue

        last = grouped[-1]
        if _are_incidents_similar(last, inc, time_window_minutes=time_window_minutes):
            continue
        else:
            grouped.append(inc)

    rows = []
    for inc in grouped:
        row = [inc[0], inc[1], inc[2], inc[3], inc[4], inc[5], inc[7], inc[6]]
        rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Cámara', 'Tipo', 'Timestamp', 'Descripción', 'Status', 'Usuario Identificado', 'EvidencePath'])
        writer.writerows(rows)

    print(f"Reporte analizado por periodo completado: {len(filtered)} registros originales -> {len(rows)} registros reportados en '{output_path}'")
    return output_path


def generate_report_xlsx(year: Optional[int] = None, month: Optional[int] = None, output_path: str = 'reporte_analizado.xlsx', time_window_minutes: int = 60):
    """
    Genera un reporte en formato Excel (.xlsx) que incluye:
      - Incumplimientos agrupados (window configurable)
      - Columna de 'occurrences' y 'last_seen'
      - Imágenes de evidencia embebidas (si `evidence_path` apunta a archivo existente)
      - Gráfica simple de incidentes por cámara
    Requiere `openpyxl` y `Pillow` instalados.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage
        # Usamos gráfico circular en lugar de barras
        from openpyxl.chart import Reference, PieChart
        from openpyxl.chart.label import DataLabelList
    except Exception:
        raise RuntimeError('Para generar Excel instale openpyxl: pip install openpyxl')

    try:
        from PIL import Image as PILImage
    except Exception:
        raise RuntimeError('Para insertar imágenes instale pillow: pip install pillow')

    # Reusar la función de filtrado por periodo
    incidents = list_incidents()
    filtered = []
    for inc in incidents:
        try:
            ts = _parse_timestamp(inc[3])
        except Exception:
            continue
        if year and ts.year != int(year):
            continue
        if month and ts.month != int(month):
            continue
        filtered.append(inc)

    incidents_sorted = sorted(filtered, key=lambda r: _parse_timestamp(r[3]))
    grouped = []
    for inc in incidents_sorted:
        if not grouped:
            grouped.append(inc)
            continue
        last = grouped[-1]
        if _are_incidents_similar(last, inc, time_window_minutes=time_window_minutes):
            continue
        grouped.append(inc)

    wb = Workbook()
    ws = wb.active
    ws.title = 'Incumplimientos'

    headers = ['ID', 'Cámara', 'Tipo', 'Timestamp', 'LastSeen', 'Occurrences', 'Descripción', 'Usuario Identificado', 'Evidencia (imagen)']
    ws.append(headers)

    # Collect counts per camera for chart
    camera_counts = {}

    for inc in grouped:
        inc_id = inc[0]
        cam = inc[1]
        tipo = inc[2]
        ts = inc[3]
        desc = inc[4]
        status = inc[5]
        evidence = inc[6]
        user = inc[7] or 'unknown'
        # occurrences and last_seen might be in columns 8 and 9 if present
        occ = None
        last_seen = None
        try:
            # attempt to read occurrences/last_seen from the tuple if present
            occ = inc[8] if len(inc) > 8 else 1
            last_seen = inc[9] if len(inc) > 9 else ts
        except Exception:
            occ = 1
            last_seen = ts

        ws.append([inc_id, cam, tipo, ts, last_seen, occ, desc, user, ''])

        camera_counts[cam] = camera_counts.get(cam, 0) + 1

    # Ensure thumb directory exists next to the DB and insert images where evidence path exists (one image per row) into column 10 (J)
    thumb_dir = os.path.join(os.path.dirname(DB_PATH), 'thumb') if DB_PATH else 'thumb'
    try:
        os.makedirs(thumb_dir, exist_ok=True)
    except Exception:
        pass

    for idx, inc in enumerate(grouped, start=1):
        evidence = inc[6]
        excel_row = idx + 1
        img_placed = False
        # prefer explicit evidence_path stored in DB
        candidates = []
        if evidence:
            candidates.append(evidence)
        # fallback: search captures directory for matching file names using timestamp
        try:
            captures_dir = os.path.join(os.path.dirname(DB_PATH), 'captures') if DB_PATH else 'captures'
            if os.path.isdir(captures_dir):
                # try to find a file that includes the date portion YYYYMMDD from timestamp
                datepart = ''
                try:
                    datepart = _parse_timestamp(inc[3]).strftime('%Y%m%d')
                except Exception:
                    datepart = ''
                for fn in os.listdir(captures_dir):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if datepart and datepart in fn:
                            candidates.append(os.path.join(captures_dir, fn))
                # if nothing matched by date, include first image as fallback
                if not candidates:
                    for fn in os.listdir(captures_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            candidates.append(os.path.join(captures_dir, fn))
                            break
        except Exception:
            pass

        for cand in candidates:
            try:
                if not cand or not os.path.isfile(cand):
                    continue
                pil = PILImage.open(cand)
                # resize if too large for Excel
                max_w, max_h = 400, 300
                w, h = pil.size
                scale = min(1.0, max_w / float(w), max_h / float(h))
                if scale < 1.0:
                    # Choose resample filter compatible with Pillow versions
                    resample_filter = getattr(PILImage, 'LANCZOS', None)
                    if resample_filter is None:
                        try:
                            resample_filter = PILImage.Resampling.LANCZOS
                        except Exception:
                            resample_filter = PILImage.BICUBIC
                    pil = pil.resize((int(w * scale), int(h * scale)), resample_filter)
                    # save thumbnail into thumb/ directory
                    base, ext = os.path.splitext(os.path.basename(cand))
                    thumb_name = f"{base}_thumb{ext}"
                    tmp_path = os.path.join(thumb_dir, thumb_name)
                    pil.save(tmp_path)
                    img_path = tmp_path
                else:
                    img_path = cand
                img = XLImage(img_path)
                # place image in column J (10)
                cell = f'J{excel_row}'
                ws.add_image(img, cell)
                img_placed = True
                break
            except Exception:
                continue
        # if no image placed, leave cell blank (no path shown)

    # Add a pie chart (gráfico circular) showing distribution por cámara (muestra porcentajes)
    chart_sheet = wb.create_sheet(title='Resumen')
    chart_sheet.append(['Cámara', 'Incidentes'])
    for cam, cnt in camera_counts.items():
        chart_sheet.append([cam, cnt])

    try:
        pie = PieChart()
        # data: las cantidades (omitimos la fila de encabezado al indicar titles_from_data=False)
        data = Reference(chart_sheet, min_col=2, min_row=2, max_row=len(camera_counts) + 1)
        labels = Reference(chart_sheet, min_col=1, min_row=2, max_row=len(camera_counts) + 1)
        pie.add_data(data, titles_from_data=False)
        pie.set_categories(labels)
        pie.title = 'Distribución de Incidentes por Cámara'
        # Mostrar porcentajes en etiquetas
        data_labels = DataLabelList()
        data_labels.showPercent = True
        data_labels.showVal = False
        pie.dataLabels = data_labels
        pie.height = 10
        pie.width = 10
        chart_sheet.add_chart(pie, 'D2')
    except Exception:
        pass

    wb.save(output_path)
    print(f'Reporte Excel generado: {output_path} (incidencias originales: {len(filtered)} -> reportadas: {len(grouped)})')
    return output_path


def generate_report_pdf(output_pdf='reporte_analizado.pdf', time_window_minutes=60):
    """
    Genera un reporte en PDF que contiene el resumen de incumplimientos (agrupados) y
    embebe imágenes de evidencia (si existen) directamente en el PDF.
    Requiere `reportlab` instalado; si no está presente, lanza excepción con instrucción.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError('reportlab no está instalado. Instale con: pip install reportlab')

    csv_path = generate_report(output_path='reporte_analizado_temp.csv', time_window_minutes=time_window_minutes)

    # Leer registros optimizados desde CSV temporal
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph('Reporte Analizado - Incumplimientos', styles['Title']))
    story.append(Spacer(1, 12))

    for r in rows:
        ts = r.get('Timestamp')
        cam = r.get('Cámara')
        tipo = r.get('Tipo')
        desc = r.get('Descripción')
        user = r.get('Usuario Identificado') or 'unknown'

        story.append(Paragraph(f'<b>{tipo}</b> - Cámara: {cam} - Usuario: {user} - Fecha/Hora: {ts}', styles['Heading3']))
        story.append(Paragraph(desc, styles['Normal']))
        story.append(Spacer(1, 6))

        # Intentar adjuntar la imagen si description contiene path o si hay imagen en carpeta captures
        # Se busca evidencia en la columna Descripción y en la base data (no tenemos path explícito en CSV), intentamos heurística
        # Buscamos en captures/ por filename que contenga la fecha/hora o tipo
        img_added = False
        captures_dir = os.path.join(os.path.dirname(DB_PATH), 'captures') if DB_PATH else 'captures'
        # busqueda heuristica: archivos en captures cuyo nombre contenga la fecha corta (YYYYMMDD) o tipo
        try:
            if os.path.isdir(captures_dir):
                for fn in os.listdir(captures_dir):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full = os.path.join(captures_dir, fn)
                        # insertar la primera imagen encontrada (no ideal, pero útil si evidencia_path no se guardó)
                        story.append(Image(full, width=400, height=300))
                        story.append(Spacer(1, 12))
                        img_added = True
                        break
        except Exception:
            pass

        if not img_added:
            story.append(Paragraph('No hay imagen de evidencia disponible.', styles['Italic']))
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF generado: {output_pdf}")
    return output_pdf