import os
import sys
import sqlite3

# Ensure repo root is on sys.path so `from config import DB_PATH` works
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('SELECT id, username, role FROM users')
rows = cur.fetchall()
print('Users in DB:')
for r in rows:
    print(r)
conn.close()
