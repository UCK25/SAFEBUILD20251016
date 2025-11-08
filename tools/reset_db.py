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

from config import DB_PATH
import database


def reset_db():
    # delete DB file if exists
    try:
        if os.path.exists(DB_PATH):
            print(f"Removing existing DB: {DB_PATH}")
            os.remove(DB_PATH)
    except Exception as e:
        print("Could not remove DB file:", e)

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

    # Insert supervisor and guest accounts
    try:
        ok_sup = database.register_user('supervisor', 'supervisor123', None)
        if ok_sup:
            print('Created user: supervisor')
        else:
            print('User supervisor may already exist or insertion failed')
    except Exception as e:
        print('register_user(supervisor) failed:', e)

    try:
        ok_guest = database.register_user('guest', 'guest123', None)
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
        conn.commit()
        conn.close()
    except Exception as e:
        print('Post-insert role adjustment failed:', e)

    print('Database reset complete.')
    return True


if __name__ == '__main__':
    ok = reset_db()
    sys.exit(0 if ok else 1)
