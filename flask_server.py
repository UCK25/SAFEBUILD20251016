from flask import Flask, request, jsonify, send_from_directory, redirect, make_response, send_file
import os, io, time, json
from PIL import Image
import numpy as np
import cv2
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Importaciones del proyecto: base de datos y configuración
from database import (
    init_db, authenticate_user, list_incidents, generate_report,
    list_users, register_incident, get_user_by_qr, list_cameras,
    get_user_by_id, generate_report_xlsx, generate_report_pdf, generate_report_by_period, update_user, update_user_role, update_incident_user_recent
)
from config import CAPTURES_DIR, DB_PATH, DEFAULT_CONF, PROJECT_ROOT

# itsdangerous para cookies de sesión firmadas
try:
    from itsdangerous import URLSafeSerializer
except Exception:
    URLSafeSerializer = None

SECRET_KEY = os.environ.get('SAFEBUILD_SECRET', 'change-me-to-a-secure-random-key')
SERIALIZER = URLSafeSerializer(SECRET_KEY) if URLSafeSerializer is not None else None

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Asegurar que exista el directorio de capturas
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Inicializar la base de datos (idempotente)
try:
    init_db()
except Exception:
    pass

# Cargar el modelo YOLO (primero variable de entorno, luego rutas del repo, finalmente fallback)
MODEL_FILE_ENV = os.environ.get('MODEL_FILE')
MODEL_PATHS = []
if MODEL_FILE_ENV:
    MODEL_PATHS.append(MODEL_FILE_ENV)
MODEL_PATHS.extend([
    os.path.join(PROJECT_ROOT, 'runs/detect/train9/weights/best.pt'),
    os.path.join(PROJECT_ROOT, 'best.pt')
])

model = None
for p in MODEL_PATHS:
    try:
        if os.path.exists(p):
            model = YOLO(p)
            # Usar GPU si está disponible, en caso contrario CPU
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            print(f"Loaded model from {p} using {device}")
            break
    except Exception as e:
        print('Error loading model', e)

if model is None:
    try:
        model = YOLO('yolov8n.pt')
        model.to('cpu')
        print('Loaded fallback yolov8n.pt')
    except Exception as e:
        print('Failed to load fallback model:', e)
        model = None


def normalize_class_name(n):
    n = n.lower()
    mapping = {
        "helmet": "casco",
        "not_helmet": "sin casco",
        "reflective": "chaleco",
        "not_reflective": "sin chaleco",
    }
    n2 = n.replace('-', '_').replace(' ', '_')
    return mapping.get(n, mapping.get(n2, n))


def detect_qr_code(image_np):
    """Detectar código QR en la imagen y devolver (valor, puntos) si hay alguno."""
    # Intentar usar pyzbar si está instalado (más robusto en muchos casos)
    try:
        import pyzbar.pyzbar as pyzbar
    except Exception:
        pyzbar = None

    try:
        if pyzbar is not None:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            objs = pyzbar.decode(gray)
            if objs:
                first = objs[0]
                val = first.data.decode('utf-8') if hasattr(first, 'data') else None
                pts = getattr(first, 'polygon', None)
                return val, pts

        # Fallback a OpenCV QRCodeDetector
        detector = cv2.QRCodeDetector()
        value, points, _ = detector.detectAndDecode(image_np)
        if value:
            return value, points
    except Exception as e:
        print(f"QR detection error: {e}")
    return None, None

def detect_and_annotate(image_np, camera_name: str = 'Camera_1'):
    """Ejecutar detección, anotar la imagen y devolver BytesIO JPEG (o None).
    También registra incidentes y logs QR si se detectan.
    """
    # preparar imagen anotada para dibujar QR y cajas
    annotated = image_np.copy()

    # Detectar QR antes de la inferencia para poder asociar usuario a un incidente
    qr_value, qr_points = detect_qr_code(image_np)
    user = None
    if qr_value:
        try:
            user = get_user_by_qr(qr_value)
            if user:
                tsnow = time.strftime('%Y-%m-%d %H:%M:%S')
                pts_serial = None
                try:
                    if qr_points is not None:
                        # normalize to list of [x,y]
                        arr = None
                        try:
                            import numpy as _np
                            arr = _np.array(qr_points, dtype=int)
                        except Exception:
                            try:
                                arr = np.array(qr_points, dtype=int)
                            except Exception:
                                arr = None
                        if arr is not None and arr.size:
                            pts_t = arr.reshape((-1, 2))
                            pts_serial = [[int(x), int(y)] for x, y in pts_t]
                except Exception:
                    pts_serial = None

                # prefer direct mapping from detected QR; if not present, try recent QR log for same camera
                user_ident = (user[1] if user else None) or _find_recent_qr_user(camera_name, within_seconds=5)
                log_entry = {
                    'ts': tsnow,
                    'qr': qr_value,
                    'user': user_ident,
                    'points': pts_serial,
                    'image_w': int(image_np.shape[1]) if hasattr(image_np, 'shape') else None,
                    'image_h': int(image_np.shape[0]) if hasattr(image_np, 'shape') else None,
                    'camera': camera_name
                }
                # Avoid spamming identical QR entries: if last entry is same qr+camera within 5s, skip
                try:
                    logs = _read_qr_log()
                    last = logs[0] if logs else None
                    skip = False
                    # consider same qr within window as duplicate (ignore camera)
                    if last and last.get('qr') == log_entry['qr']:
                        try:
                            from datetime import datetime as _dt
                            last_ts = _dt.strptime(last.get('ts'), '%Y-%m-%d %H:%M:%S')
                            now_ts = _dt.strptime(log_entry.get('ts'), '%Y-%m-%d %H:%M:%S')
                            if abs((now_ts - last_ts).total_seconds()) <= 5:
                                skip = True
                        except Exception:
                            pass
                    if not skip:
                        # use dedupe helper to add entry (will handle ts/ts_epoch and short dedupe)
                        try:
                            ok = _add_qr_log_entry(log_entry, dedupe_seconds=5)
                            # ok == False means helper considered this a duplicate
                        except Exception:
                            # helper failed; best-effort atomic write
                            try:
                                import tempfile
                                existing = _read_qr_log() or []
                                new_logs = [log_entry] + existing[:99]
                                tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8')
                                json.dump(new_logs, tmp, ensure_ascii=False, indent=2)
                                tmp.flush()
                                tmp_name = tmp.name
                                tmp.close()
                                os.replace(tmp_name, QR_LOG_FILE)
                            except Exception:
                                try:
                                    _write_qr_log([log_entry] + (existing if 'existing' in locals() else [])[:99])
                                except Exception:
                                    pass
                        # si el QR tiene usuario conocido, intentar asignarlo a un incidente reciente desconocido
                        try:
                            if user_ident:
                                # probar actualizar incidentes recientes para ambos tipos (casco/chaleco)
                                try:
                                    update_incident_user_recent(camera_name, 'Falta Casco', user_ident, within_seconds=5)
                                except Exception:
                                    pass
                                try:
                                    update_incident_user_recent(camera_name, 'Falta Chaleco', user_ident, within_seconds=5)
                                except Exception:
                                    pass
                                # refrescar log de incidentes local
                                try:
                                    incidents = list_incidents()
                                    simple = []
                                    for inc in incidents:
                                        simple.append({'id': inc[0], 'camera_name': inc[1], 'type': inc[2], 'timestamp': inc[3], 'description': inc[4], 'status': inc[5], 'evidence_path': inc[6], 'user_identified': inc[7]})
                                    _write_incident_log(simple[:200])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
                # dibujar caja del QR y nombre de usuario en verde
                try:
                    pts = qr_points
                    if pts is not None:
                        # pts puede ser lista de puntos o numpy array
                        arr = None
                        try:
                            import numpy as _np
                            arr = _np.array(pts, dtype=int)
                        except Exception:
                            try:
                                arr = np.array(pts, dtype=int)
                            except Exception:
                                arr = None
                        if arr is not None and arr.size:
                            pts_t = arr.reshape((-1, 2))
                            pts_list = [(int(x), int(y)) for x, y in pts_t]
                            cv2.polylines(annotated, [np.array(pts_list, dtype=int)], True, (0,255,0), 2)
                            # label with username
                            try:
                                name = user[1]
                                cv2.putText(annotated, name, (pts_list[0][0], max(pts_list[0][1]-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception as e:
            print(f"QR logging error: {e}")

    if model is None:
        return None
    # annotated ya preparado arriba
    try:
        # Inferencia a baja resolución para reducir latencia
        results = model.predict(source=image_np, imgsz=320, conf=DEFAULT_CONF, iou=0.5, max_det=50, verbose=False)
    except Exception:
        try:
            results = model(image_np, imgsz=320, conf=0.35, verbose=False)
        except Exception as e:
            print('Inference error', e)
            results = None
    detected = []
    if results:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and hasattr(boxes, 'xyxy'):
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = [int(v) for v in box]
                cid = int(cls_ids[i]) if len(cls_ids) > i else 0
                raw = r.names.get(cid, str(cid))
                cname = normalize_class_name(raw)
                if cname in ("casco", "sin casco", "chaleco", "sin chaleco"):
                    # color: si 'sin' -> rojo (no lo lleva), si presente -> celeste
                    if 'sin' in cname:
                        color = (0, 0, 255)  # rojo BGR
                    else:
                        color = (255, 191, 0)  # celeste BGR
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    # label legible
                    if 'sin' in cname:
                        # nombre de incidente legible
                        if 'casco' in cname:
                            label = 'Falta Casco'
                        else:
                            label = 'Falta Chaleco'
                    else:
                        if 'casco' in cname:
                            label = 'Casco'
                        else:
                            label = 'Chaleco'

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    tx1, ty1 = x1, max(y1 - th - 8, 0)
                    tx2, ty2 = x1 + tw + 8, ty1 + th + 6
                    cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
                    cv2.putText(annotated, label, (x1 + 4, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    detected.append(label)

                    # Si es una violación (sin casco / sin chaleco), registrar incidente con formato legible
                    if 'sin' in cname:
                        try:
                            tsf = time.strftime('%Y%m%d_%H%M%S')
                            evidence_path = os.path.join(CAPTURES_DIR, f'violation_{tsf}.jpg')
                            cv2.imwrite(evidence_path, annotated)

                            # Tipo legible
                            if 'casco' in cname:
                                incident_type = 'Falta Casco'
                            else:
                                incident_type = 'Falta Chaleco'

                            ts_text = time.strftime('%Y-%m-%d %H:%M:%S')
                            user_ident = None
                            try:
                                if user:
                                    user_ident = user[1]
                            except Exception:
                                user_ident = None

                            description = f"{incident_type} — {camera_name} {ts_text}\nUsuario: {user_ident or 'unknown'}"

                            # dedupe window: 5 seconds expressed in minutes to avoid duplicates
                            dedupe_min = 5.0 / 60.0
                            register_incident(
                                camera_name=camera_name,
                                incident_type=incident_type,
                                description=description,
                                user_identified=user_ident,
                                evidence_path=evidence_path,
                                dedupe_window_minutes=dedupe_min
                            )

                            # Actualizar log JSON local con lo guardado en la BD (últimas entradas)
                            try:
                                incidents = list_incidents()
                                simple = []
                                for inc in incidents:
                                    simple.append({
                                        'id': inc[0],
                                        'camera_name': inc[1],
                                        'type': inc[2],
                                        'timestamp': inc[3],
                                        'ts': inc[3],
                                        'summary': inc[2],
                                        'detail': inc[4],
                                        'description': inc[4],
                                        'status': inc[5],
                                        'evidence_path': inc[6],
                                        'user_identified': inc[7]
                                    })
                                _write_incident_log(simple[:200])
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"Error registering incident: {e}")
    # Superponer marca de tiempo
    try:
        ts_text = time.strftime('%Y-%m-%d %H:%M:%S')
        status = f"{ts_text} | Detecciones: {len(detected)}"
        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (6,6), (6 + tw + 8, 6 + th + 8), (0,0,0), -1)
        cv2.putText(annotated, status, (10, 6 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    except Exception:
        pass
    ret, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        return None
    return io.BytesIO(buf.tobytes())


# Helpers simples para QR y logs (subconjunto)
QR_LOG_FILE = os.path.join(CAPTURES_DIR, 'qr_scans.json')
INCIDENT_LOG_FILE = os.path.join(CAPTURES_DIR, 'incidents.json')


def _read_qr_log():
    try:
        if os.path.exists(QR_LOG_FILE):
            with open(QR_LOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # normalize entries to frontend expectations: ts, qr, user, points, image_w/image_h
                out = []
                for e in (data or []):
                    try:
                        ne = dict(e)
                        if 'ts' not in ne and 'timestamp' in ne:
                            ne['ts'] = ne.get('timestamp')
                        if 'qr' not in ne and 'qr_value' in ne:
                            ne['qr'] = ne.get('qr_value')
                        if 'user' not in ne and 'username' in ne:
                            ne['user'] = ne.get('username')
                        # keep points/image sizes if present
                        out.append(ne)
                    except Exception:
                        continue
                return out
    except Exception:
        pass
    return []


def _write_qr_log(data):
    try:
        with open(QR_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _add_qr_log_entry(entry, dedupe_seconds=5):
    """Add a QR log entry with short-time deduplication for flask server.
    Ensures 'ts' and 'ts_epoch' exist and skips writing if the latest entry has same qr+camera within dedupe_seconds.
    """
    try:
        if not isinstance(entry, dict):
            return False
        import time
        if 'ts_epoch' not in entry:
            entry['ts_epoch'] = time.time()
        if 'ts' not in entry:
            entry['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Read current log and perform dedupe; to avoid races between processes
        # re-read immediately before writing and perform the dedupe check again.
        import tempfile
        logs = _read_qr_log() or []
        try:
            last = logs[0] if logs else None
            if last:
                # consider duplicate if same QR value OR same resolved user (when present)
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
                if same_qr or same_user:
                    last_epoch = last.get('ts_epoch')
                    cur_epoch = entry.get('ts_epoch')
                    if last_epoch is not None and cur_epoch is not None:
                        if float(cur_epoch) - float(last_epoch) <= float(dedupe_seconds):
                            return False
        except Exception:
            pass

        # Re-load just before write to handle concurrent writers
        try:
            logs_now = _read_qr_log() or []
            last_now = logs_now[0] if logs_now else None
            if last_now:
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
                if same_qr_now or same_user_now:
                    try:
                        last_epoch = last_now.get('ts_epoch')
                        cur_epoch = entry.get('ts_epoch')
                        if last_epoch is not None and cur_epoch is not None and float(cur_epoch) - float(last_epoch) <= float(dedupe_seconds):
                            return False
                    except Exception:
                        pass
            # insert into the freshly read list and atomically replace file
            logs_now.insert(0, entry)
            # write to temp file then replace
            try:
                tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8')
                json.dump(logs_now[:100], tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                tmp_name = tmp.name
                tmp.close()
                os.replace(tmp_name, QR_LOG_FILE)
                return True
            except Exception:
                # fallback to best-effort write
                try:
                    _write_qr_log(logs_now[:100])
                    return True
                except Exception:
                    return False
        except Exception:
            return False
    except Exception:
        return False


def _find_recent_qr_user(camera_name: str, within_seconds: int = 5):
    """Buscar en el log local de QR una entrada del mismo camera_name con user presente
    dentro de `within_seconds`. Retorna el nombre de usuario o None."""
    try:
        logs = _read_qr_log()
        if not logs:
            return None
        from datetime import datetime as _dt
        now = _dt.now()
        for entry in logs:
            try:
                if not entry:
                    continue
                if entry.get('camera') != camera_name:
                    continue
                user = entry.get('user')
                if not user:
                    continue
                ts = entry.get('ts') or entry.get('timestamp')
                if not ts:
                    continue
                entry_dt = _dt.strptime(ts, '%Y-%m-%d %H:%M:%S')
                if abs((now - entry_dt).total_seconds()) <= within_seconds:
                    return user
            except Exception:
                continue
    except Exception:
        return None
    return None


def _read_incident_log():
    try:
        if os.path.exists(INCIDENT_LOG_FILE):
            with open(INCIDENT_LOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                out = []
                for e in (data or []):
                    try:
                        ne = dict(e)
                        # normalize timestamp -> ts for frontend
                        if 'ts' not in ne and 'timestamp' in ne:
                            ne['ts'] = ne.get('timestamp')
                        # summary/detail convenience fields
                        if 'summary' not in ne:
                            ne['summary'] = ne.get('type') or ''
                        if 'detail' not in ne:
                            ne['detail'] = ne.get('description') or ''
                        # user normalization
                        if 'user' not in ne:
                            ne['user'] = ne.get('user_identified')
                        out.append(ne)
                    except Exception:
                        continue
                return out
    except Exception:
        pass
    return []


@app.route('/api/users', methods=['GET', 'PUT'])
def api_users():
    # GET: list users (respect session)
    if request.method == 'GET':
        current = _get_current_user_from_cookie()
        if not current:
            return jsonify([])
        rows = list_users()
        out = []
        for r in rows:
            out.append({'id': r[0], 'username': r[1], 'role': r[2], 'qr_code': r[3]})
        return jsonify(out)

    # PUT: update user (body contains id, username, role optional, qr_code optional)
    data = request.get_json() or {}
    uid_raw = data.get('id')
    try:
        uid = int(uid_raw)
    except Exception:
        return jsonify({'ok': False, 'error': 'missing or invalid id'}), 400
    current = _get_current_user_from_cookie()
    if not current:
        return jsonify({'ok': False, 'error': 'authentication required'}), 401
    cur_id, cur_name, cur_role = current
    cur_role = str(cur_role).lower()
    # permission: admin can edit most, supervisor limited, guests only self
    target = None
    try:
        from database import get_user_by_id
        target = get_user_by_id(uid)
    except Exception:
        target = None
    if not target:
        return jsonify({'ok': False, 'error': 'user not found'}), 404

    target_role = str(target[2]).lower()
    if cur_role == 'guest' and int(cur_id) != int(uid):
        return jsonify({'ok': False, 'error': 'not allowed'}), 403
    if cur_role == 'supervisor' and target_role == 'admin' and int(cur_id) != int(uid):
        return jsonify({'ok': False, 'error': 'not allowed'}), 403

    username = data.get('username')
    qr = data.get('qr_code')
    new_role = data.get('role')
    from database import update_user, update_user_role
    ok = update_user(uid, username=username, qr_code=qr)
    if not ok:
        return jsonify({'ok': False, 'error': 'update failed (dup?)'}), 400
    if new_role:
        try:
            update_user_role(uid, new_role)
        except Exception:
            pass
    return jsonify({'ok': True})


@app.route('/api/download_qr/<int:user_id>')
def api_download_qr(user_id: int):
    current = _get_current_user_from_cookie()
    if not current:
        return jsonify({'ok': False}), 401
    cur_id, cur_name, cur_role = current
    try:
        target = get_user_by_id(user_id)
    except Exception:
        target = None
    if not target:
        return jsonify({'ok': False, 'error': 'user not found'}), 404
    qr = None
    try:
        qr = target[3]
    except Exception:
        qr = None
    if not qr:
        return jsonify({'ok': False, 'error': 'no qr assigned'}), 404
    # permission checks
    allowed = False
    try:
        if int(cur_id) == int(user_id):
            allowed = True
        elif str(cur_role).lower() == 'admin':
            if str(target[2]).lower() != 'admin':
                allowed = True
        elif str(cur_role).lower() == 'supervisor':
            if str(target[2]).lower() == 'guest':
                allowed = True
    except Exception:
        allowed = False
    if not allowed:
        return jsonify({'ok': False, 'error': 'not allowed'}), 403
    # generate qr image
    try:
        import qrcode
    except Exception:
        qrcode = None
    if qrcode is None:
        return jsonify({'ok': False, 'error': 'qrcode lib missing'}), 500
    try:
        img = qrcode.make(qr)
        bio = io.BytesIO()
        img.save(bio, 'PNG')
        bio.seek(0)
        # log download
        try:
            from database import log_qr_download
            log_qr_download(int(cur_id), int(user_id), f'qr_download_{user_id}.png')
        except Exception:
            pass
        return send_file(bio, mimetype='image/png', as_attachment=False, download_name=f'QR_{target[1]}.png')
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/last_qr')
def api_last_qr():
    logs = _read_qr_log()
    if not logs:
        return jsonify({'ok': False, 'entry': None})
    return jsonify({'ok': True, 'entry': logs[0]})


@app.route('/api/last_incidents')
def api_last_incidents():
    logs = _read_incident_log()
    if not logs:
        return jsonify({'ok': False, 'entry': None})
    return jsonify({'ok': True, 'entry': logs[0]})


@app.route('/report/csv')
def report_csv():
    user = _get_current_user_from_cookie()
    if not user:
        return jsonify({'ok': False}), 401
    if str(user[2]).lower() == 'guest':
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    out = generate_report(output_path='reporte_analizado.csv')
    if not os.path.exists(out):
        return jsonify({'ok': False, 'error': 'report failed'}), 500
    return send_file(out, mimetype='text/csv', as_attachment=True, download_name=os.path.basename(out))


@app.route('/report/xlsx')
def report_xlsx():
    """Return a ZIP that contains two files:
      - incidents_raw.csv (raw incidents for the period)
      - reporte_analizado.xlsx (analyzed excel) if possible; otherwise include analyzed CSV and a README explaining missing dependency.
    """
    user = _get_current_user_from_cookie()
    if not user:
        return jsonify({'ok': False}), 401
    if str(user[2]).lower() == 'guest':
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    # allow optional query params year/month (GET args)
    year = request.args.get('year')
    month = request.args.get('month')

    # generate raw incidents CSV for period (or all)
    try:
        if year or month:
            raw_csv = generate_report_by_period(year=int(year) if year else None, month=int(month) if month else None, output_path='incidents_raw.csv')
        else:
            raw_csv = generate_report(output_path='incidents_raw.csv')
    except Exception:
        # fallback to default full report
        raw_csv = generate_report(output_path='incidents_raw.csv')

    zip_name = f'reporte_bundle_{time.strftime("%Y%m%d_%H%M%S")}.zip'
    try:
        # create XLSX; if this fails we fail explicitly because requirement is to return raw CSV + XLSX together
        xlsx_out = generate_report_xlsx(year=int(year) if year else None, month=int(month) if month else None, output_path='reporte_analizado.xlsx')
        # build zip with raw CSV + xlsx
        import zipfile
        with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(raw_csv, arcname=os.path.basename(raw_csv))
            zf.write(xlsx_out, arcname=os.path.basename(xlsx_out))
        return send_file(zip_name, mimetype='application/zip', as_attachment=True, download_name=os.path.basename(zip_name))
    except Exception as e:
        # Explicit failure: XLSX not created — inform operator to install dependencies
        return jsonify({'ok': False, 'error': 'Could not generate XLSX. Install required dependencies: pip install openpyxl pillow'}), 500


@app.route('/report/pdf')
def report_pdf():
    user = _get_current_user_from_cookie()
    if not user:
        return jsonify({'ok': False}), 401
    if str(user[2]).lower() == 'guest':
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    try:
        out = generate_report_pdf(output_pdf='reporte_analizado.pdf')
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
    return send_file(out, mimetype='application/pdf', as_attachment=True, download_name=os.path.basename(out))


def _write_incident_log(data):
    try:
        with open(INCIDENT_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# Session helpers

def _get_current_user_from_cookie():
    cookie = request.cookies.get('safebuild_session')
    if not cookie or SERIALIZER is None:
        return None
    try:
        data = SERIALIZER.loads(cookie)
        return (data.get('id'), data.get('username'), data.get('role'))
    except Exception:
        return None


@app.route('/')
def index():
    return redirect('/dashboard')


@app.route('/dashboard')
def dashboard():
    path = os.path.join(app.static_folder, 'dashboard.html')
    if os.path.exists(path):
        return send_from_directory(app.static_folder, 'dashboard.html')
    return '<html><body><h1>Dashboard no disponible</h1></body></html>'


@app.route('/manage')
def manage():
    path = os.path.join(app.static_folder, 'manage.html')
    if os.path.exists(path):
        return send_from_directory(app.static_folder, 'manage.html')
    return '<html><body><h1>Manage no disponible</h1></body></html>'


@app.route('/api/incidents')
def api_list_incidents():
    rows = list_incidents()
    out = []
    for r in rows:
        try:
            ts = r[3]
        except Exception:
            ts = None
        out.append({'id': r[0], 'camera_name': r[1], 'type': r[2], 'timestamp': ts, 'description': r[4], 'status': r[5], 'user_identified': r[7]})
    return jsonify(out)


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'ok': False, 'message': 'Missing credentials'}), 400
    user = authenticate_user(username, password)
    if not user:
        return jsonify({'ok': False, 'message': 'Invalid credentials'}), 401
    resp = make_response(jsonify({'ok': True, 'user': {'id': user[0], 'username': user[1], 'role': user[2]}}))
    if SERIALIZER is not None:
        token = SERIALIZER.dumps({'id': int(user[0]), 'username': user[1], 'role': user[2]})
        proto = request.headers.get('x-forwarded-proto') or request.scheme
        secure = True if str(proto).lower() == 'https' else False
        resp.set_cookie('safebuild_session', token, httponly=True, samesite='Lax', secure=secure, max_age=24*3600)
    return resp


@app.route('/api/me')
def api_me():
    user = _get_current_user_from_cookie()
    if not user:
        return jsonify({'ok': False}), 401
    uid, uname, urole = user
    return jsonify({'ok': True, 'user': {'id': uid, 'username': uname, 'role': urole}})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    resp = make_response(jsonify({'ok': True}))
    resp.delete_cookie('safebuild_session')
    return resp


@app.route('/detect', methods=['POST'])
def detect():
    if 'frame' not in request.files:
        return jsonify({'ok': False, 'error': 'missing file'}), 400
    f = request.files['frame']
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]
    except Exception as e:
        return jsonify({'ok': False, 'error': 'invalid image'}), 400
    out = detect_and_annotate(img_np)
    if out is None:
        return jsonify({'ok': False, 'error': 'no model'}), 500
    out.seek(0)
    # save a copy for evidence
    try:
        tsf = time.strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(CAPTURES_DIR, f'capture_{tsf}.jpg')
        with open(fname, 'wb') as fh:
            fh.write(out.read())
        out.seek(0)
    except Exception:
        pass
    return (out.getvalue(), 200, {'Content-Type': 'image/jpeg'})


@app.route('/detect_json', methods=['POST'])
def detect_json():
    # Similar al endpoint original, pero optimizado para baja latencia: devuelve cajas y QR (si aplica)
    if 'frame' not in request.files:
        return jsonify({'ok': False, 'error': 'missing file'}), 400
    f = request.files['frame']
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]
    except Exception:
        return jsonify({'ok': False, 'error': 'invalid image'}), 400

    if model is None:
        return jsonify({'ok': False, 'error': 'no model'}), 500

    try:
        # Inferencia a resolución reducida para mejorar el tiempo de respuesta
        results = model.predict(source=img_np, imgsz=320, conf=DEFAULT_CONF, iou=0.5, max_det=50, verbose=False)
    except Exception:
        try:
            results = model(img_np, imgsz=320, conf=0.35, verbose=False)
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    boxes_out = []
    qr_found = None
    qr_user = None
    qr_obj = None
    # detectar QR de forma rápida (si hay) y mapear a usuario si existe
    try:
        qr_val, qr_points = detect_qr_code(img_np)
        if qr_val:
            qr_found = qr_val
            try:
                u = get_user_by_qr(qr_found)
                if u:
                    qr_user = u[1]
            except Exception:
                qr_user = None
            # construir objeto QR para respuesta: ts, qr, user, points, image sizes
            try:
                tsnow = time.strftime('%Y-%m-%d %H:%M:%S')
                pts_serial = None
                if qr_points is not None:
                    try:
                        import numpy as _np
                        arr = _np.array(qr_points, dtype=int)
                        pts_t = arr.reshape((-1,2))
                        pts_serial = [[int(x), int(y)] for x,y in pts_t]
                    except Exception:
                        try:
                            pts_serial = [[int(p[0]), int(p[1])] for p in qr_points]
                        except Exception:
                            pts_serial = None
                qr_obj = {
                    'ts': tsnow,
                    'qr': qr_found,
                    'user': qr_user,
                    'points': pts_serial,
                    'image_w': int(img_np.shape[1]) if hasattr(img_np, 'shape') else None,
                    'image_h': int(img_np.shape[0]) if hasattr(img_np, 'shape') else None
                }
                # registrar en log local (mantener últimas 100 entradas)
                try:
                    cam_for_qr = request.form.get('camera') or request.args.get('camera') or 'Camera_1'
                except Exception:
                    cam_for_qr = 'Camera_1'
                qr_obj['camera'] = cam_for_qr
                # avoid duplicate QR log entries (same qr+camera within 5s)
                try:
                    logs = _read_qr_log()
                    last = logs[0] if logs else None
                    skip = False
                    if last and last.get('qr') == qr_obj.get('qr') and last.get('camera') == qr_obj.get('camera'):
                        try:
                            from datetime import datetime as _dt
                            last_ts = _dt.strptime(last.get('ts'), '%Y-%m-%d %H:%M:%S')
                            now_ts = _dt.strptime(qr_obj.get('ts'), '%Y-%m-%d %H:%M:%S')
                            if abs((now_ts - last_ts).total_seconds()) <= 5:
                                skip = True
                        except Exception:
                            pass
                    if not skip:
                        try:
                            ok = _add_qr_log_entry(qr_obj, dedupe_seconds=5)
                        except Exception:
                            try:
                                import tempfile
                                existing = _read_qr_log() or []
                                new_logs = [qr_obj] + existing[:99]
                                tmp = tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8')
                                json.dump(new_logs, tmp, ensure_ascii=False, indent=2)
                                tmp.flush()
                                tmp_name = tmp.name
                                tmp.close()
                                os.replace(tmp_name, QR_LOG_FILE)
                            except Exception:
                                try:
                                    _write_qr_log([qr_obj] + (existing if 'existing' in locals() else [])[:99])
                                except Exception:
                                    pass
                        # si detectamos usuario, intentar vincular a incidentes recientes desconocidos
                        try:
                            if qr_obj.get('user'):
                                try:
                                    update_incident_user_recent(cam_for_qr, 'Falta Casco', qr_obj.get('user'), within_seconds=5)
                                except Exception:
                                    pass
                                try:
                                    update_incident_user_recent(cam_for_qr, 'Falta Chaleco', qr_obj.get('user'), within_seconds=5)
                                except Exception:
                                    pass
                                try:
                                    incidents = list_incidents()
                                    simple = []
                                    for inc in incidents:
                                        simple.append({'id': inc[0], 'camera_name': inc[1], 'type': inc[2], 'timestamp': inc[3], 'description': inc[4], 'status': inc[5], 'evidence_path': inc[6], 'user_identified': inc[7]})
                                    _write_incident_log(simple[:200])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                qr_obj = None
    except Exception:
        qr_found = None

    if results:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and hasattr(boxes, 'xyxy'):
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            for i, box in enumerate(xyxy):
                x1,y1,x2,y2 = [float(v) for v in box]
                cid = int(cls_ids[i]) if len(cls_ids)>i else 0
                raw = r.names.get(cid, str(cid))
                cname = normalize_class_name(raw)
                if cname in ("casco", "sin casco", "chaleco", "sin chaleco"):
                    boxes_out.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': cname})
                    # si es violación, registrar incidente de forma mínima
                    if 'sin' in cname:
                        try:
                            cam = request.form.get('camera') or request.args.get('camera') or 'Camera_1'
                            # Prefer direct QR mapping; if missing, look for a recent QR nearby (within 5s)
                            user_ident = qr_user or _find_recent_qr_user(cam, within_seconds=5)
                            ts = time.strftime('%Y%m%d_%H%M%S')
                            ev_path = os.path.join(CAPTURES_DIR, f'inc_{ts}.jpg')
                            try:
                                cv2.imwrite(ev_path, img_np)
                            except Exception:
                                ev_path = None

                            # Map normalized classname to human-friendly incident type to match other paths
                            if 'casco' in cname:
                                incident_type = 'Falta Casco'
                            else:
                                incident_type = 'Falta Chaleco'

                            # description consistent with other code paths
                            description = f"{incident_type} — {cam} {time.strftime('%Y-%m-%d %H:%M:%S')}\nUsuario: {user_ident or 'unknown'}"

                            # dedupe window: 5 seconds expressed in minutes
                            dedupe_min = 5.0 / 60.0
                            register_incident(
                                camera_name=cam,
                                incident_type=incident_type,
                                description=description,
                                user_identified=user_ident,
                                evidence_path=ev_path,
                                dedupe_window_minutes=dedupe_min
                            )
                            try:
                                incidents = list_incidents()
                                simple = []
                                for inc in incidents:
                                    simple.append({'id': inc[0], 'camera_name': inc[1], 'type': inc[2], 'timestamp': inc[3], 'description': inc[4], 'status': inc[5], 'evidence_path': inc[6], 'user_identified': inc[7]})
                                _write_incident_log(simple[:200])
                            except Exception:
                                pass
                        except Exception:
                            pass

    # devolver qr como objeto compatible con frontend (o null)
    return jsonify({'ok': True, 'boxes': boxes_out, 'qr': qr_obj, 'qr_user': qr_user})


if __name__ == '__main__':
    # for local debugging
    app.run(host='0.0.0.0', port=8000)
