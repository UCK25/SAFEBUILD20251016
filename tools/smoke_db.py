import os, sqlite3
os.chdir(r"C:\Users\kenka\Downloads\SAFEBUILD20251001")
from database import init_db, list_users, register_user, get_user_by_id, reset_user_password, list_cameras, register_camera

print('Running DB smoke test in', os.getcwd())
init_db()
print('DB initialized')
users = list_users()
print('Users:', users)
# create temporary test user
ok = register_user('test_user_for_smoke', 'TestPass123!', 'TESTQR_SMOKE')
print('register_user ok?', ok)
users = list_users()
print('Users after add (tail 5):', users[-5:])
# find user id
uid = None
for u in users:
    if u[1] == 'test_user_for_smoke':
        uid = u[0]
        break
print('new uid', uid)
if uid:
    # reset password with actor 'admin'
    ok = reset_user_password('admin', uid, 'NewSm0kePass!')
    print('reset_user_password ok?', ok)
    # show latest audit logs
    con = sqlite3.connect('safety_monitor.db')
    cur = con.cursor()
    cur.execute('SELECT id, timestamp, performed_by, action, target_user_id, details FROM audit_logs ORDER BY id DESC LIMIT 5')
    rows = cur.fetchall()
    print('Audit logs (recent):')
    for r in rows:
        print(r)
    con.close()
# Camera create and delete
cams_before = list_cameras()
print('Cameras before:', cams_before)
register_camera('smoke_cam_x', None, None)
cams_after = list_cameras()
print('Cameras after add:', cams_after[-5:])
# remove the smoke camera
con = sqlite3.connect('safety_monitor.db')
cur = con.cursor()
cur.execute("DELETE FROM cameras WHERE name=?", ('smoke_cam_x',))
con.commit()
con.close()
print('Deleted smoke camera')
print('Done')
