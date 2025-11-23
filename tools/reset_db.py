"""
Reset the SQLite database to a clean state and create default users:
  - admin / admin123 (role: admin)
  - supervisor / supervisor123 (role: supervisor)
  - guest / guest123 (role: guest)

This script deletes the configured DB file (if present), initializes schema, and inserts the users.
Run from repository root: python tools/reset_db.py
"""
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from config import DB_PATH, CAPTURES_DIR
import database


def reset_db():
    # delete DB file if exists
    try:
        if os.path.exists(DB_PATH):
            print(f"Removing existing DB: {DB_PATH}")
            os.remove(DB_PATH)
    except Exception as e:
        print("Could not remove DB file:", e)

    # clear captures directory (logs and evidence) to return to fresh state
    try:
        if os.path.exists(CAPTURES_DIR):
            print(f"Clearing captures directory: {CAPTURES_DIR}")
            for fn in os.listdir(CAPTURES_DIR):
                path = os.path.join(CAPTURES_DIR, fn)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"Could not remove {path}: {e}")
        else:
            try:
                os.makedirs(CAPTURES_DIR, exist_ok=True)
            except Exception:
                pass
    except Exception as e:
        print('Error clearing captures dir:', e)

    # initialize DB (creates schema and default admin if none)
    try:
        print("Initializing DB schema...")
        database.init_db()
    except Exception as e:
        print("init_db failed:", e)
        return False

    # ensure admin user has known password (init_db inserts admin with admin123 by default)
    try:
        # ensure admin exists; if not, create
        cur = database.get_user_by_qr  # just to ensure import
    except Exception:
        pass

    # Insert supervisor and guest accounts with default QR codes
    SUPERVISOR_QR = 'SUPERVISOR001'
    GUEST_QR = 'GUEST001'
    try:
        ok_sup = database.register_user('supervisor', 'supervisor123', SUPERVISOR_QR, role='supervisor')
        if ok_sup:
            print('Created user: supervisor')
        else:
            print('User supervisor may already exist or insertion failed')
    except Exception as e:
        print('register_user(supervisor) failed:', e)

    try:
        ok_guest = database.register_user('guest', 'guest123', GUEST_QR, role='guest')
        if ok_guest:
            print('Created user: guest')
        else:
            print('User guest may already exist or insertion failed')
    except Exception as e:
        print('register_user(guest) failed:', e)

    # Ensure guest role is 'guest' (register_user default role is 'supervisor' in schema); update if needed
    try:
        # find guest id
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, username, role FROM users WHERE username IN ('admin','supervisor','guest')")
        rows = cur.fetchall()
        print('Current users after creation:')
        for r in rows:
            print(' -', r)
        # set guest role explicitly
        cur.execute("UPDATE users SET role='guest' WHERE username='guest'")
        # ensure supervisor role
        cur.execute("UPDATE users SET role='supervisor' WHERE username='supervisor'")
        # ensure admin exists with role admin (init_db should have created it)
        cur.execute("UPDATE users SET role='admin' WHERE username='admin'")
        # ensure default QR codes assigned and unique for supervisor/guest
        try:
            # clear any other user that mistakenly has these QR codes
            cur.execute("UPDATE users SET qr_code=NULL WHERE qr_code IN (?, ?) AND username NOT IN ('supervisor','guest')", (SUPERVISOR_QR, GUEST_QR))
        except Exception:
            pass
        try:
            cur.execute("UPDATE users SET qr_code=? WHERE username='supervisor'", (SUPERVISOR_QR,))
        except Exception:
            pass
        try:
            cur.execute("UPDATE users SET qr_code=? WHERE username='guest'", (GUEST_QR,))
        except Exception:
            pass
        conn.commit()
        conn.close()
    except Exception as e:
        print('Post-insert role adjustment failed:', e)

    # recreate empty log files
    try:
        qr_file = os.path.join(CAPTURES_DIR, 'qr_scans.json')
        incidents_file = os.path.join(CAPTURES_DIR, 'incidents.json')
        try:
            with open(qr_file, 'w', encoding='utf-8') as f:
                import json
                json.dump([], f)
        except Exception as e:
            print('Could not write qr_scans.json:', e)
        try:
            with open(incidents_file, 'w', encoding='utf-8') as f:
                import json
                json.dump([], f)
        except Exception as e:
            print('Could not write incidents.json:', e)
    except Exception:
        pass

    print('Database reset complete.')
    return True


if __name__ == '__main__':
    ok = reset_db()
    sys.exit(0 if ok else 1)
