"""
Lightweight simulation test for QR dedupe logic.
This script reproduces the essential parts of _add_qr_log_entry but does not import
the full server (avoids heavy deps like ultralytics/torch). It writes to the same
`captures/qr_scans.json` so you can inspect the real file.
"""
import time
import json
import os
import tempfile


# use same path as server
QR_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'captures', 'qr_scans.json')


def _read_qr_log():
    try:
        if os.path.exists(QR_LOG_FILE):
            with open(QR_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _write_qr_log_atomic(entries):
    try:
        tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8')
        json.dump(entries, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        name = tmp.name
        tmp.close()
        os.replace(name, QR_LOG_FILE)
        return True
    except Exception:
        try:
            with open(QR_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


def add_qr_log_entry_sim(entry, dedupe_seconds=5):
    # normalize
    if 'ts_epoch' not in entry:
        entry['ts_epoch'] = time.time()
    if 'ts' not in entry:
        entry['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')

    logs = _read_qr_log() or []
    if logs:
        last = logs[0]
        try:
            same_qr = False
            same_user = False
            try:
                same_qr = str(last.get('qr')) == str(entry.get('qr'))
            except Exception:
                same_qr = False
            try:
                last_user = last.get('user')
                cur_user = entry.get('user')
                same_user = (last_user is not None and cur_user is not None and str(last_user) == str(cur_user))
            except Exception:
                same_user = False
            if (same_qr or same_user):
                last_epoch = last.get('ts_epoch')
                if last_epoch is not None and float(entry['ts_epoch']) - float(last_epoch) <= float(dedupe_seconds):
                    return False
        except Exception:
            pass

    # re-read and write atomically to avoid races
    logs_now = _read_qr_log() or []
    if logs_now:
        try:
            last_now = logs_now[0]
            same_qr_now = False
            same_user_now = False
            try:
                same_qr_now = str(last_now.get('qr')) == str(entry.get('qr'))
            except Exception:
                same_qr_now = False
            try:
                last_user_now = last_now.get('user')
                cur_user = entry.get('user')
                same_user_now = (last_user_now is not None and cur_user is not None and str(last_user_now) == str(cur_user))
            except Exception:
                same_user_now = False
            if (same_qr_now or same_user_now):
                last_epoch = last_now.get('ts_epoch')
                if last_epoch is not None and float(entry['ts_epoch']) - float(last_epoch) <= float(dedupe_seconds):
                    return False
        except Exception:
            pass

    new_logs = [entry] + logs_now[:99]
    ok = _write_qr_log_atomic(new_logs)
    return ok


def read_and_print():
    data = _read_qr_log() or []
    print('entries:', len(data))
    if data:
        print(data[0])


if __name__ == '__main__':
    # ensure directory
    os.makedirs(os.path.dirname(QR_LOG_FILE), exist_ok=True)
    try:
        if os.path.exists(QR_LOG_FILE):
            os.remove(QR_LOG_FILE)
    except Exception:
        pass

    entry = {'qr': 'TESTQR123', 'camera': 'UnitTestCam'}
    print('Calling helper 3 times quickly (0.3s intervals)')
    for i in range(3):
        ok = add_qr_log_entry_sim(dict(entry), dedupe_seconds=5)
        print(f'call {i+1}, ok={ok}')
        time.sleep(0.3)

    read_and_print()
    print('Sleeping 6 seconds to exceed dedupe window...')
    time.sleep(6)
    ok2 = add_qr_log_entry_sim(dict(entry), dedupe_seconds=5)
    print('call after wait, ok=', ok2)
    read_and_print()
