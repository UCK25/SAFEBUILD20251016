# database.py - ACTUALIZADO
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
    
    # Insertar usuario por defecto si no existe (con password hasheada)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        hashed = _hash_password('admin123')
        cursor.execute("INSERT INTO users (username, password, role, qr_code) VALUES (?, ?, ?, ?)", 
                      ('admin', hashed, 'admin', 'ADMIN001'))
    
    conn.commit()
    conn.close()

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
def register_incident(camera_name, incident_type, description, user_identified=None, evidence_path=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO incidents (camera_name, type, timestamp, description, evidence_path, user_identified) 
                   VALUES (?, ?, ?, ?, ?, ?)''',
                   (camera_name, incident_type, timestamp, description, evidence_path, user_identified))
    conn.commit()
    conn.close()

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

# Reportes (CU-022 a CU-023)
import csv
def generate_report(output_path='report.csv'):
    incidents = list_incidents()
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Cámara', 'Tipo', 'Timestamp', 'Descripción', 'Status', 'Usuario Identificado'])
        writer.writerows(incidents)
    return output_path