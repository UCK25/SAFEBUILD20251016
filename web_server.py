from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io, os, cv2, numpy as np, time, json
from ultralytics import YOLO
from PIL import Image
from typing import Optional

# optional faster QR fallback
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except Exception:
    pyzbar_decode = None

# DB
from database import (
    list_incidents, update_incident, list_users, authenticate_user,
    generate_report, generate_report_xlsx, generate_report_pdf,
    list_cameras, register_camera, get_user_by_id, register_incident,
    register_user, delete_user, update_user
)
from config import CAPTURES_DIR
from config import DB_PATH

# Simple session cookie using itsdangerous
try:
    from itsdangerous import URLSafeSerializer
except Exception:
    URLSafeSerializer = None

try:
    import qrcode
except Exception:
    qrcode = None

# SECRET_KEY - prefer environment variable
SECRET_KEY = os.environ.get('SAFEBUILD_SECRET', 'change-me-to-a-secure-random-key')
SERIALIZER = URLSafeSerializer(SECRET_KEY) if URLSafeSerializer is not None else None

app = FastAPI(title="SafeBuild Web Detector")

# Allow all origins for demo; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html + js)
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
# Serve static files and note DB initialization is intentionally disabled.
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Ensure captures dir exists and serve as static for evidence images
try:
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")
except Exception:
    # If CAPTURES_DIR is not writable or missing, continue without mounting
    print('[server] Warning: could not mount captures directory')

# NOTE: Automatic database initialization is disabled to avoid modifying an
# existing database. If you need to create the schema, run `database.init_db()`
# manually (e.g., a management script) before starting the server.

# Model load: try custom weights then fallback to yolov8n
MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), 'runs/detect/train9/weights/best.pt'),
    os.path.join(os.path.dirname(__file__), 'runs/detect/train/weights/best.pt'),
    'best.pt'
]

print("[server] Inicializando modelo de detección...")
model = None
for p in MODEL_PATHS:
    try:
        if os.path.exists(p):
            print(f"[server] Intentando cargar modelo personalizado: {p}")
            try:
                model = YOLO(p)
                model.to('cpu')
                print(f"[server] Modelo personalizado cargado: {p}")
                break
            except Exception as e:
                print(f"[server] Error cargando {p}: {e}")
    except Exception:
        pass

if model is None:
    print("[server] Cargando modelo ligero yolov8n.pt como fallback")
    model = YOLO('yolov8n.pt')
    model.to('cpu')
    print("[server] yolov8n.pt cargado")

# Mapping de clases
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

# Detección y anotación
def detect_and_annotate(image_np):
    annotated = image_np.copy()
    try:
        # increase confidence and iou slightly to reduce noisy detections
        results = model.predict(source=image_np, imgsz=640, conf=0.8, iou=0.8, max_det=50, verbose=False)
    except Exception as e:
        try:
            results = model(image_np, imgsz=640, conf=0.25, verbose=False)
        except Exception as e2:
            print("[server] Error en inferencia:", e2)
            results = None
    detected = []
    if results:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and hasattr(boxes, 'xyxy'):
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            for i, box in enumerate(xyxy):
                x1,y1,x2,y2 = [int(v) for v in box]
                cid = int(cls_ids[i]) if len(cls_ids)>i else 0
                raw = r.names.get(cid, str(cid))
                cname = normalize_class_name(raw)
                # Filtrar solo casco/chaleco
                if cname in ("casco", "sin casco", "chaleco", "sin chaleco"):
                    # seleccionar color
                    if 'sin' in cname:
                        color = (0,0,255)  # rojo para faltante
                        thickness = 4
                    else:
                        color = (0,255,0)  # verde positivo
                        thickness = 2
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
                    label = cname
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    tx1, ty1 = x1, max(y1 - th - 8, 0)
                    tx2, ty2 = x1 + tw + 8, ty1 + th + 6
                    cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0,0,0), -1)
                    cv2.putText(annotated, label, (x1 + 4, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    detected.append(label)
    # Overlay timestamp and count
    ts_text = time.strftime("%Y-%m-%d %H:%M:%S")
    status = f"{ts_text} | Detecciones: {len(detected)}"
    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(annotated, (6,6), (6 + tw + 8, 6 + th + 8), (0,0,0), -1)
    cv2.putText(annotated, status, (10, 6 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # Encode to JPEG
    ret, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        return None
    return io.BytesIO(buf.tobytes())

@app.get('/')
async def index():
    # Serve simple HTML
    html_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    return HTMLResponse('<html><body><h1>SafeBuild Web Detector</h1></body></html>')


@app.get('/dashboard')
async def dashboard():
    html_path = os.path.join(static_dir, 'dashboard.html')
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    return HTMLResponse('<html><body><h1>Dashboard no disponible</h1></body></html>')


@app.get('/manage')
async def manage_page():
    html_path = os.path.join(static_dir, 'manage.html')
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, 'r', encoding='utf-8').read())
    return HTMLResponse('<html><body><h1>Gestionar no disponible</h1></body></html>')


@app.get('/api/incidents')
async def api_list_incidents(request: Request, year: Optional[int] = None, month: Optional[int] = None):
    """Return incidents as simple log entries. Optionally filter by year and month (1-12).
    Requires no auth to view logs, but evidence paths are omitted from the public API.
    """
    rows = list_incidents()
    out = []
    for r in rows:
        try:
            ts = r[3]
        except Exception:
            ts = None
        # filter by year/month if provided
        if ts and (year or month):
            try:
                import datetime
                dt = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                if year and dt.year != int(year):
                    continue
                if month and dt.month != int(month):
                    continue
            except Exception:
                pass
        out.append({
            'id': r[0],
            'camera_name': r[1],
            'type': r[2],
            'timestamp': ts,
            'description': r[4],
            'status': r[5],
            'user_identified': r[7]
        })
    return JSONResponse(out)


@app.post('/api/incidents/{incident_id}/status')
async def api_update_incident_status(incident_id: int, request: Request):
    data = await request.json()
    status = data.get('status')
    if not status:
        raise HTTPException(status_code=400, detail='Missing status')
    # Require authenticated user to change status
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    update_incident(incident_id, status)
    return JSONResponse({'ok': True, 'id': incident_id, 'status': status})


@app.get('/api/users')
async def api_list_users(request: Request):
    # If not authenticated, do not return other users
    current = _get_current_user(request)
    if not current:
        return JSONResponse([])
    cur_id, cur_name, cur_role = current
    cur_role = str(cur_role).lower()
    rows = list_users()
    out = []
    if cur_role in ('admin', 'supervisor'):
        for r in rows:
            out.append({'id': r[0], 'username': r[1], 'role': r[2], 'qr_code': r[3]})
    else:
        # return only the current user
        for r in rows:
            if int(r[0]) == int(cur_id):
                out.append({'id': r[0], 'username': r[1], 'role': r[2], 'qr_code': r[3]})
                break
    return JSONResponse(out)


@app.post('/api/users')
async def api_create_user(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')
    qr = data.get('qr_code')
    if not username or not password:
        raise HTTPException(status_code=400, detail='username and password required')
    try:
        ok = register_user(username, password, qr)
        if not ok:
            raise HTTPException(status_code=400, detail='Could not register user (maybe duplicate)')
        return JSONResponse({'ok': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put('/api/users/{user_id}')
async def api_update_user(user_id: int, request: Request):
    data = await request.json()
    username = data.get('username')
    qr = data.get('qr_code')
    # require auth and role checks
    current = _get_current_user(request)
    if not current:
        raise HTTPException(status_code=401, detail='Authentication required')
    # Basic permission: only admin or supervisor (or the user itself) can update
    cur_id, cur_name, cur_role = current
    cur_role = str(cur_role).lower()
    if int(cur_id) != int(user_id) and cur_role not in ('admin', 'supervisor'):
        raise HTTPException(status_code=403, detail='Not allowed')
    try:
        from database import update_user
        ok = update_user(user_id, username=username, qr_code=qr)
        if not ok:
            raise HTTPException(status_code=400, detail='Update failed (maybe duplicate)')
        return JSONResponse({'ok': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/users/{user_id}')
async def api_delete_user(user_id: int, request: Request):
    current = _get_current_user(request)
    if not current:
        raise HTTPException(status_code=401, detail='Authentication required')
    cur_id, cur_name, cur_role = current
    cur_role = str(cur_role).lower()
    # Prevent deleting self and require admin role for deleting others
    if int(cur_id) == int(user_id):
        raise HTTPException(status_code=403, detail='Cannot delete yourself')
    if cur_role != 'admin':
        raise HTTPException(status_code=403, detail='Admin only')
    try:
        from database import delete_user
        ok = delete_user(user_id)
        if not ok:
            raise HTTPException(status_code=500, detail='Delete failed')
        return JSONResponse({'ok': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/download_qr/{user_id}')
async def api_download_qr(user_id: int, request: Request):
    # Permissions similar to PASADO logic
    current = _get_current_user(request)
    if not current:
        raise HTTPException(status_code=401, detail='Authentication required')
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    qr = target[2] if False else target[3]  # target tuple shape: id,username,role,qr
    # actually get qr
    try:
        qr = target[3]
    except Exception:
        qr = None

    if not qr:
        raise HTTPException(status_code=404, detail='QR not assigned')

    # permission checks
    cur_id, cur_name, cur_role = current
    try:
        cur_id = int(cur_id)
    except Exception:
        cur_id = None
    target_id = int(user_id)
    target_role = str(target[2]).lower()
    cur_role = str(cur_role).lower()

    allowed = False
    if cur_id == target_id:
        allowed = True
    elif cur_role == 'admin':
        if target_role != 'admin':
            allowed = True
    elif cur_role == 'supervisor':
        if target_role == 'guest':
            allowed = True

    if not allowed:
        raise HTTPException(status_code=403, detail='Not allowed to download this QR')

    # generate QR PNG
    if qrcode is None:
        raise HTTPException(status_code=500, detail='qrcode library not installed')
    try:
        img = qrcode.make(qr)
        bio = io.BytesIO()
        img.save(bio, 'PNG')
        bio.seek(0)
        return StreamingResponse(bio, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/login')
async def api_login(request: Request):
    data = await request.json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        raise HTTPException(status_code=400, detail='Missing credentials')
    user = authenticate_user(username, password)
    if not user:
        return JSONResponse({'ok': False, 'message': 'Invalid credentials'}, status_code=401)
    # Create session cookie
    resp = JSONResponse({'ok': True, 'user': {'id': user[0], 'username': user[1], 'role': user[2]}})
    if SERIALIZER is not None:
        token = SERIALIZER.dumps({'id': int(user[0]), 'username': user[1], 'role': user[2]})
        resp.set_cookie('safebuild_session', token, httponly=True, max_age=24*3600)
    return resp


@app.get('/api/me')
async def api_me(request: Request):
    user = _get_current_user(request)
    if not user:
        return JSONResponse({'ok': False}, status_code=401)
    uid, uname, urole = user
    return JSONResponse({'ok': True, 'user': {'id': uid, 'username': uname, 'role': urole}})


def _get_current_user(request: Request):
    # returns tuple (id, username, role) or None
    cookie = None
    try:
        cookie = request.cookies.get('safebuild_session')
    except Exception:
        return None
    if not cookie or SERIALIZER is None:
        return None
    try:
        data = SERIALIZER.loads(cookie)
        return (data.get('id'), data.get('username'), data.get('role'))
    except Exception:
        return None


@app.get('/api/cameras')
async def api_list_cameras():
    rows = list_cameras()
    out = []
    for r in rows:
        out.append({'id': r[0], 'name': r[1], 'camera_index': r[2]})
    return JSONResponse(out)


@app.post('/api/cameras')
async def api_register_camera(request: Request):
    data = await request.json()
    name = data.get('name')
    camera_index = data.get('camera_index', 0)
    if not name:
        raise HTTPException(status_code=400, detail='Missing camera name')
    try:
        register_camera(name, int(camera_index))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({'ok': True})


@app.put('/api/cameras/{camera_id}')
async def api_update_camera(camera_id: int, request: Request):
    data = await request.json()
    name = data.get('name')
    camera_index = data.get('camera_index')
    if not name:
        raise HTTPException(status_code=400, detail='Missing camera name')
    # permission check
    current = _get_current_user(request)
    if not current:
        raise HTTPException(status_code=401, detail='Authentication required')
    cur_role = str(current[2]).lower()
    if cur_role not in ('admin', 'supervisor'):
        raise HTTPException(status_code=403, detail='Not allowed')
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('UPDATE cameras SET name=?, camera_index=? WHERE id=?', (name, camera_index, camera_id))
        conn.commit()
        conn.close()
        return JSONResponse({'ok': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/api/cameras/{camera_id}')
async def api_delete_camera(camera_id: int, request: Request):
    current = _get_current_user(request)
    if not current:
        raise HTTPException(status_code=401, detail='Authentication required')
    cur_role = str(current[2]).lower()
    if cur_role not in ('admin', 'supervisor'):
        raise HTTPException(status_code=403, detail='Not allowed')
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('DELETE FROM cameras WHERE id=?', (camera_id,))
        conn.commit()
        conn.close()
        return JSONResponse({'ok': True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/report/csv')
async def report_csv(request: Request):
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    out = generate_report(output_path='reporte_analizado.csv')
    if not os.path.exists(out):
        raise HTTPException(status_code=500, detail='Report generation failed')
    return FileResponse(out, media_type='text/csv', filename=os.path.basename(out))


@app.get('/report/xlsx')
async def report_xlsx(request: Request):
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    try:
        out = generate_report_xlsx(output_path='reporte_analizado.xlsx')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(out, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=os.path.basename(out))
async def report_pdf(request: Request):
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    try:
        out = generate_report_pdf(output_pdf='reporte_analizado.pdf')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(out, media_type='application/pdf', filename=os.path.basename(out))

@app.post('/detect')
async def detect(frame: UploadFile = File(...)):
    contents = await frame.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
    except Exception as e:
        print('[server] Error leyendo imagen:', e)
        return StreamingResponse(io.BytesIO(contents), media_type='image/jpeg')

    out = detect_and_annotate(img_np)


    # Ensure captures dir exists
    try:
        os.makedirs(CAPTURES_DIR, exist_ok=True)
    except Exception:
        pass

    # Save annotated frame for evidence
    try:
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(CAPTURES_DIR, f'capture_{ts}.jpg')
        out.seek(0)
        with open(fname, 'wb') as f:
            f.write(out.read())
    except Exception as e:
        print('[server] Error saving capture:', e)
        fname = None

    # Register a generic incident so it appears in DB (PPE_violation)
    try:
        # try to resolve user from recent QR logs/mappings
        user_name = _resolve_user_from_recent_qr(boxes_out=None, time_window_seconds=1)
        try:
            register_incident('web_demo', 'PPE_violation', 'Detección desde web demo', user_identified=user_name, evidence_path=fname)
        except Exception as e:
            print('[server] Error registering incident:', e)
        # append to incident log for quick dashboard polling
        try:
            summary = f"Falta Casco — Cámara 1 {time.strftime('%Y-%m-%d %H:%M:%S')}"
            detail = f"Falta Casco - Usuario: {user_name if user_name else 'unknown'}"
            logs = _read_incident_log() or []
            logs.insert(0, {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'summary': summary, 'detail': detail, 'user': user_name})
            _write_incident_log(logs[:100])
        except Exception:
            pass
    except Exception as e:
        print('[server] Error handling incident logging:', e)

    out.seek(0)
    return StreamingResponse(out, media_type='image/jpeg')


@app.post('/detect_json')
async def detect_json(frame: UploadFile = File(...)):
    """Return detection boxes and labels as JSON for client-side overlay drawing."""
    contents = await frame.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
    except Exception:
        return JSONResponse({'ok': False, 'error': 'invalid image'}, status_code=400)

    try:
        # run model with moderate thresholds
        results = model.predict(source=img_np, imgsz=640, conf=0.35, iou=0.5, max_det=50, verbose=False)
    except Exception:
        try:
            results = model(img_np, imgsz=640, conf=0.35, verbose=False)
        except Exception as e:
            return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)

    boxes_out = []
    if results:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and hasattr(boxes, 'xyxy'):
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in box]
                cid = int(cls_ids[i]) if len(cls_ids) > i else 0
                raw = r.names.get(cid, str(cid))
                cname = normalize_class_name(raw)
                if cname in ("casco", "sin casco", "chaleco", "sin chaleco"):
                    boxes_out.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': cname})

    # QR detection (cv2 + optional pyzbar fallback)
    qr_entry = None
    try:
        detector = cv2.QRCodeDetector()
        found = None
        points = None
        # try multi
        try:
            data_list, pts, _ = detector.detectAndDecodeMulti(img_np)
        except Exception:
            data_list = None
            pts = None

        if data_list:
            for idx, d in enumerate(data_list):
                if d:
                    found = d
                    try:
                        if pts is not None and len(pts) > idx:
                            p = pts[idx]
                            points = [[float(p[i][0]), float(p[i][1])] for i in range(p.shape[0])]
                    except Exception:
                        points = None
                    break

        if not found:
            data, pts2, _ = detector.detectAndDecode(img_np)
            if data:
                found = data
                try:
                    if pts2 is not None:
                        points = [[float(pts2[i][0]), float(pts2[i][1])] for i in range(pts2.shape[0])]
                except Exception:
                    points = None

        # pyzbar fallback
        if not found and pyzbar_decode is not None:
            try:
                pil = Image.fromarray(img_np[:, :, ::-1])
                zs = pyzbar_decode(pil)
                if zs:
                    z = zs[0]
                    found = z.data.decode('utf-8') if z.data else None
                    try:
                        points = [[float(p.x), float(p.y)] for p in z.polygon]
                    except Exception:
                        points = None
            except Exception:
                pass

        if found:
            h, w = img_np.shape[:2]
            # try to resolve username for this QR
            try:
                user_name = _find_user_by_qr_value(found)
            except Exception:
                user_name = None
            # fallback: try direct DB user qr_code match
            if not user_name:
                try:
                    rows = list_users()
                    for r in rows:
                        try:
                            qr_code = r[3]
                        except Exception:
                            qr_code = None
                        if qr_code and str(qr_code) == str(found):
                            user_name = r[1]
                            break
                except Exception:
                    pass
            qr_entry = {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'ts_epoch': time.time(), 'qr': found, 'user': user_name, 'points': points, 'image_w': w, 'image_h': h}
            logs = _read_qr_log() or []
            logs.insert(0, qr_entry)
            _write_qr_log(logs[:50])
    except Exception:
        qr_entry = None

    # If PPE violation was detected, save annotated frame and register incident
    try:
        has_violation = any(('sin' in (b.get('label') or '').lower()) for b in boxes_out)
        if has_violation:
            try:
                # attempt to resolve user via recent QR or mappings
                user_name = None
                # prefer the QR we just detected in this frame
                if qr_entry:
                    try:
                        user_name = _find_user_by_qr_value(qr_entry.get('qr'))
                        if not user_name:
                            # fallback: try resolving from recent logs with proximity/time
                            user_name = _resolve_user_from_recent_qr(boxes_out=boxes_out, time_window_seconds=1)
                    except Exception:
                        user_name = None
                else:
                    user_name = _resolve_user_from_recent_qr(boxes_out=boxes_out, time_window_seconds=1)

                annotated_io = detect_and_annotate(img_np)
                fname = None
                if annotated_io is not None:
                    try:
                        os.makedirs(CAPTURES_DIR, exist_ok=True)
                    except Exception:
                        pass
                    tsf = time.strftime('%Y%m%d_%H%M%S')
                    fname = os.path.join(CAPTURES_DIR, f'capture_{tsf}.jpg')
                    try:
                        annotated_io.seek(0)
                        with open(fname, 'wb') as f:
                            f.write(annotated_io.read())
                    except Exception:
                        fname = None

                try:
                    register_incident('web_demo', 'PPE_violation', 'Detección desde web demo', user_identified=user_name, evidence_path=fname)
                except Exception:
                    pass

                # append to incident log for quick dashboard polling
                try:
                    # summary text can be customized per label; take first missing label
                    missing_label = None
                    for b in boxes_out:
                        if 'sin' in (b.get('label') or '').lower():
                            missing_label = b.get('label')
                            break
                    lbl = missing_label or 'PPE_violation'
                    summary = f"{lbl} — Cámara 1 {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    detail = f"{lbl} - Usuario: {user_name if user_name else 'unknown'}"
                    logs = _read_incident_log() or []
                    logs.insert(0, {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'ts_epoch': time.time(), 'summary': summary, 'detail': detail, 'user': user_name})
                    _write_incident_log(logs[:100])
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    resp = {'ok': True, 'boxes': boxes_out}
    if qr_entry:
        resp['qr'] = qr_entry
    return JSONResponse(resp)


@app.post('/api/scan_qr')
async def api_scan_qr(image: UploadFile = File(...)):
    """Accept an uploaded image and try to decode QR codes using OpenCV (with pyzbar fallback)."""
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({'ok': False, 'error': 'invalid image'}, status_code=400)

        detector = cv2.QRCodeDetector()
        found = None
        points = None

        # try multi decode
        try:
            data_list, pts, _ = detector.detectAndDecodeMulti(img)
        except Exception:
            data_list = None
            pts = None

        if data_list:
            for idx, d in enumerate(data_list):
                if d:
                    found = d
                    try:
                        if pts is not None and len(pts) > idx:
                            p = pts[idx]
                            points = [[float(p[i][0]), float(p[i][1])] for i in range(p.shape[0])]
                    except Exception:
                        points = None
                    break

        # single decode fallback
        if not found:
            try:
                data, pts2, _ = detector.detectAndDecode(img)
                if data:
                    found = data
                    try:
                        if pts2 is not None:
                            points = [[float(pts2[i][0]), float(pts2[i][1])] for i in range(pts2.shape[0])]
                    except Exception:
                        points = None
            except Exception:
                pass

        # adaptive threshold attempts
        if not found:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for method in (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C):
                    try:
                        th = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, 25, 10)
                        d, p, _ = detector.detectAndDecode(th)
                        if d:
                            found = d
                            try:
                                if p is not None:
                                    points = [[float(p[i][0]), float(p[i][1])] for i in range(p.shape[0])]
                            except Exception:
                                points = None
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        # pyzbar fallback
        if not found and pyzbar_decode is not None:
            try:
                pil = Image.fromarray(img[:, :, ::-1])
                zs = pyzbar_decode(pil)
                if zs:
                    z = zs[0]
                    found = z.data.decode('utf-8') if z.data else None
                    try:
                        points = [[float(p.x), float(p.y)] for p in z.polygon]
                    except Exception:
                        points = None
            except Exception:
                pass

        if not found:
            return JSONResponse({'ok': False, 'qr': None}, status_code=404)

        # attempt to get reliable corner points
        try:
            ok, det_pts = detector.detect(img)
            if ok and det_pts is not None and len(det_pts):
                p = det_pts[0]
                points = [[float(p[i][0]), float(p[i][1])] for i in range(p.shape[0])]
        except Exception:
            pass

        h, w = img.shape[:2]
        try:
            user_name = _find_user_by_qr_value(found)
        except Exception:
            user_name = None
        # fallback: check DB users for qr_code match
        if not user_name:
            try:
                rows = list_users()
                for r in rows:
                    try:
                        qr_code = r[3]
                    except Exception:
                        qr_code = None
                    if qr_code and str(qr_code) == str(found):
                        user_name = r[1]
                        break
            except Exception:
                pass
        entry = {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'ts_epoch': time.time(), 'qr': found, 'user': user_name, 'points': points, 'image_w': w, 'image_h': h}
        logs = _read_qr_log() or []
        logs.insert(0, entry)
        _write_qr_log(logs[:50])

        return JSONResponse({'ok': True, 'qr': found, 'points': points, 'image_w': w, 'image_h': h})
    except Exception as e:
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)


@app.post('/api/logout')
async def api_logout(request: Request):
    # clear session cookie server-side
    resp = JSONResponse({'ok': True})
    resp.delete_cookie('safebuild_session')
    return resp


MAPPINGS_FILE = os.path.join(CAPTURES_DIR, 'mappings.json')
def _read_mappings():
    try:
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _write_mappings(data):
    try:
        with open(MAPPINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# QR scan log (most recent scans)
QR_LOG_FILE = os.path.join(CAPTURES_DIR, 'qr_scans.json')
def _read_qr_log():
    try:
        if os.path.exists(QR_LOG_FILE):
            with open(QR_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
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


# Incident log (most recent incidents for dashboard display)
INCIDENT_LOG_FILE = os.path.join(CAPTURES_DIR, 'incidents.json')
def _read_incident_log():
    try:
        if os.path.exists(INCIDENT_LOG_FILE):
            with open(INCIDENT_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _write_incident_log(data):
    try:
        with open(INCIDENT_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _find_user_by_qr_value(qr_value):
    """Return username if mapping exists for qr_value, else None."""
    try:
        mappings = _read_mappings() or []
        for m in mappings:
            try:
                if str(m.get('qr')) == str(qr_value):
                    # support mappings that store user_id (numeric) or user (username)
                    uid = m.get('user_id') if m.get('user_id') is not None else m.get('user')
                    if uid is not None:
                        # try numeric id first
                        try:
                            user = get_user_by_id(int(uid))
                            if user:
                                return user[1]
                        except Exception:
                            # try to find by username in users list
                            try:
                                rows = list_users()
                                for r in rows:
                                    if str(r[1]) == str(uid):
                                        return r[1]
                            except Exception:
                                pass
                # also check users table for qr_code match
                try:
                    rows = list_users()
                    for r in rows:
                        # r tuple: id, username, role, qr_code (as used elsewhere)
                        try:
                            qr_code = r[3]
                        except Exception:
                            qr_code = None
                        if qr_code and str(qr_code) == str(qr_value):
                            return r[1]
                except Exception:
                    pass
            except Exception:
                continue
    except Exception:
        pass
    return None


def _resolve_user_from_recent_qr(boxes_out=None, time_window_seconds=1):
    """
    Attempt to resolve a user name from the most recent QR logs and mappings.
    If boxes_out is provided and the most recent QR entry contains points and image size,
    perform a crude proximity check. Otherwise, prefer a time-based recent mapping.
    """
    try:
        logs = _read_qr_log() or []
        if not logs:
            return None
        now_epoch = time.time()
        for entry in logs:
            try:
                # Prefer epoch timestamp if present
                ts_epoch = entry.get('ts_epoch')
                if ts_epoch is None:
                    # try parsing string timestamp as local time
                    ts = entry.get('ts')
                    if not ts:
                        continue
                    try:
                        import datetime
                        ets = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        ts_epoch = time.mktime(ets.timetuple())
                    except Exception:
                        ts_epoch = None
                if ts_epoch is None:
                    # can't determine time, consider as candidate
                    # if the QR log already contains a resolved user, prefer that
                    username = entry.get('user') or _find_user_by_qr_value(entry.get('qr'))
                    if username:
                        return username
                    continue
                age = now_epoch - float(ts_epoch)
                if age <= float(time_window_seconds):
                    # consider this recent
                    # prefer existing resolved 'user' in the log entry
                    username = entry.get('user') or _find_user_by_qr_value(entry.get('qr'))
                    if username:
                        # further optional proximity check
                        if boxes_out and entry.get('points') and entry.get('image_w') and entry.get('image_h'):
                            try:
                                # compute center of first box
                                bx = boxes_out[0]
                                cx = (bx.get('x1') + bx.get('x2')) / 2.0
                                cy = (bx.get('y1') + bx.get('y2')) / 2.0
                                # compute qr center from points
                                pts = entry.get('points')
                                if pts and len(pts):
                                    qx = sum([p[0] for p in pts]) / len(pts)
                                    qy = sum([p[1] for p in pts]) / len(pts)
                                    # normalize distance by image diagonal
                                    w = entry.get('image_w') or 1
                                    h = entry.get('image_h') or 1
                                    diag = (w*w + h*h) ** 0.5
                                    dist = ((cx - qx)**2 + (cy - qy)**2) ** 0.5
                                    if dist <= max(0.05 * diag, 50):
                                        return username
                                    else:
                                        # even if not spatially close, accept a very recent match
                                        return username
                            except Exception:
                                return username
                        else:
                            return username
            except Exception:
                continue
    except Exception:
        pass
    return None


@app.get('/api/mappings')
async def api_list_mappings():
    return JSONResponse(_read_mappings())


@app.post('/api/mappings')
async def api_add_mapping(request: Request):
    data = await request.json()
    qr = data.get('qr')
    user_id = data.get('user_id')
    note = data.get('note')
    if not qr or not user_id:
        raise HTTPException(status_code=400, detail='qr and user_id required')
    mappings = _read_mappings()
    mappings.append({'qr': qr, 'user_id': user_id, 'note': note, 'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
    ok = _write_mappings(mappings)
    if not ok:
        raise HTTPException(status_code=500, detail='Could not save mapping')
    return JSONResponse({'ok': True})


@app.get('/api/last_qr')
async def api_last_qr():
    """Return the most recent QR scan log entry, if any."""
    logs = _read_qr_log()
    if not logs:
        return JSONResponse({'ok': False, 'entry': None})
    return JSONResponse({'ok': True, 'entry': logs[0]})


@app.get('/api/last_incidents')
async def api_last_incidents():
    """Return the most recent incident log entry, if any."""
    logs = _read_incident_log()
    if not logs:
        return JSONResponse({'ok': False, 'entry': None})
    return JSONResponse({'ok': True, 'entry': logs[0]})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
