# app.py
# SafeBuild - Web en vivo (streaming) con detección YOLO, QR y registro.
# - Video en vivo desde navegador (streamlit-webrtc)
# - Detección por frames en CPU (ultralytics)
# - Asociación QR cercano -> persona sin EPP
# - Registro mediante observers / database
# - Autenticación y descarga de QR según rol

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, cv2, numpy as np, os, time, io, qrcode
import threading
from datetime import datetime
from queue import Queue, Empty
from ultralytics import YOLO

# módulos del proyecto
from config import MODEL_PATH, DEFAULT_CONF, CAPTURES_DIR, PROJECT_ROOT
from observer import SafetyMonitorSubject, AlertLogger, IncidentRegistrar, RankingUpdater
from database import (
    init_db, authenticate_user, list_incidents, generate_report,
    list_users, register_incident, get_user_by_qr
)

# pyzbar (QR)
try:
    import pyzbar.pyzbar as pyzbar
except Exception:
    pyzbar = None

st.set_page_config(page_title="SafeBuild Monitor (Live)", layout="wide")
st.title("SafeBuild Monitor")

# Compat: algunas versiones de streamlit eliminan `experimental_rerun`.
# `streamlit-webrtc` puede llamarla; si no existe, añadimos un stub seguro
# para evitar AttributeError y seguir funcionando (no hace rerun real).
if not hasattr(st, "experimental_rerun"):
    def _experimental_rerun_stub():
        # Solo registrar para debugging; no reinicia la app en esta versión
        print("[compat] st.experimental_rerun() llamado pero no disponible en esta versión de streamlit")
        return None
    st.experimental_rerun = _experimental_rerun_stub

# -----------------------------
# Inicialización base
# -----------------------------
init_db()
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Inicialización del modelo YOLO: cargar rápidamente un modelo ligero para que la demo funcione
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loading = True
    try:
        print("[init] Cargando modelo ligero 'yolov8n.pt' para inicio rápido...")
        m = YOLO('yolov8n.pt')
        m.to('cpu')
        st.session_state.model = m
        st.session_state.model_source = 'yolov8n.pt'
        st.session_state.model_loading = False
        print("[init] yolov8n.pt cargado (modo ligero)")
    except Exception as e:
        print(f"[init] Error cargando yolov8n.pt: {e}")
        st.session_state.model = None
        st.session_state.model_loading = True

    # En background intentar cargar el modelo personalizado (best.pt) y reemplazar si es posible
    def _try_load_custom():
        try_paths = [
            MODEL_PATH,
            os.path.join(PROJECT_ROOT, 'runs/detect/train9/weights/best.pt'),
            os.path.join(PROJECT_ROOT, 'runs/detect/train/weights/best.pt'),
            os.path.join(PROJECT_ROOT, 'best.pt')
        ]
        for p in try_paths:
            try:
                if os.path.exists(p):
                    print(f"[background] Intentando cargar modelo personalizado desde: {p}")
                    try:
                        m2 = YOLO(p)
                        m2.to('cpu')
                        st.session_state.model = m2
                        st.session_state.model_source = p
                        print(f"[background] Modelo personalizado cargado desde: {p}")
                        break
                    except Exception as e2:
                        print(f"[background] Error cargando {p}: {e2}")
            except Exception as e:
                print(f"[background] Error examinando ruta {p}: {e}")
        st.session_state.model_loading = False

    threading.Thread(target=_try_load_custom, daemon=True).start()
    st.info("Modelo ligero cargado para inicio rápido; se intentará cargar modelo personalizado en segundo plano.")

if "subject" not in st.session_state:
    subj = SafetyMonitorSubject()
    st.session_state.event_queue = Queue()
    st.session_state.alert_messages = []

    class StreamlitAlertLogger(AlertLogger):
        def update(self, subject_obj, event_data):
            ts = event_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            cam = event_data.get("camera", "webcam")
            alert_type = event_data.get("alert_type", "")
            user = event_data.get("user_identified", "unknown")
            msg = f"[{ts}] {cam}: {alert_type or 'Evento detectado'} - {user}"
            last = st.session_state.alert_messages[-1] if st.session_state.alert_messages else None
            if last != msg:
                st.session_state.alert_messages.append(msg)

    logger = StreamlitAlertLogger()
    subj.attach(logger)
    subj.attach(IncidentRegistrar())
    subj.attach(RankingUpdater({}))
    st.session_state.subject = subj

# Detección habilitada por defecto (permite desactivar inferencia para ver sólo video)
if "detection_enabled" not in st.session_state:
    st.session_state.detection_enabled = True

# -----------------------------
# Utilidades detección
# -----------------------------
def normalize_class_name(n):
    n = n.lower()
    mapping = {
        "helmet": "casco",
        "not_helmet": "sin casco",
        "reflective": "chaleco",
        "not_reflective": "sin chaleco",
        # Mantener compatibilidad con nombres anteriores
    }
    n2 = n.replace("-", "_").replace(" ", "_")
    return mapping.get(n, mapping.get(n2, n))

def detect_qr_codes(frame):
    found = []
    if pyzbar is None:
        return found, frame
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objs = pyzbar.decode(gray)
        for o in objs:
            try:
                data = o.data.decode("utf-8")
            except Exception:
                continue
            x, y, w, h = o.rect
            cx, cy = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            found.append((data, cx, cy))
    except Exception:
        pass
    return found, frame

def draw_box(frame, bbox, label, color=(255,0,0)):
    x1,y1,x2,y2 = bbox
    # Rectangle border
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    if label:
        # Draw filled rectangle behind text for better visibility
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx1, ty1 = x1, max(y1 - th - 8, 0)
        tx2, ty2 = x1 + tw + 8, ty1 + th + 6
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0,0,0), -1)
        cv2.putText(frame, label, (x1 + 4, ty2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_frame(frame_bgr, model, camera="webcam", imgsz=640, conf=0.25):
    # Asegurar que el frame sea válido
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, {}

    # Hacer una copia segura del frame
    try:
        annotated = frame_bgr.copy()
    except Exception as e:
        print(f"Error al copiar frame: {e}")
        return frame_bgr, {}

    # Detectar QR
    qr_list, annotated = detect_qr_codes(annotated)
    qr_map = {f"qr_{i}": d for i,(d,_,_) in enumerate(qr_list,1)}
    if qr_list:
        qr_map["any"] = qr_list[0][0]

    detected_classes = []
    try:
        if model:
            # Configurar parámetros de detección optimizados
            try:
                results = model.predict(
                    source=frame_bgr,
                    imgsz=imgsz,  # resolución dinámica para acelerar inferencia cuando se pida
                    conf=conf,   # Umbral de confianza dinámico
                    iou=0.45,    # IOU threshold
                    max_det=50,  # Máximo de detecciones
                    verbose=False
                )
            except Exception as _e:
                # Fallback: usar llamada directa (algunas versiones de ultralytics)
                try:
                    results = model(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)
                except Exception as e:
                    print("Error en inferencia YOLO (predict & call):", e)
                    results = None
            if results:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None and hasattr(boxes, "xyxy"):
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    dets = []
                    for i, box in enumerate(xyxy):
                        x1,y1,x2,y2 = [int(v) for v in box]
                        cid = int(cls_ids[i]) if len(cls_ids)>i else 0
                        raw_name = r.names.get(cid, str(cid))
                        cname = normalize_class_name(raw_name)
                        # Filtrar solo clases relevantes (casco / chaleco y sus negativos)
                        if cname in ("casco", "sin casco", "chaleco", "sin chaleco"):
                            dets.append({"class": cname, "bbox": (x1,y1,x2,y2), "center": ((x1+x2)//2,(y1+y2)//2)})
                    
                    # Agrupar por proximidad usando clustering simple
                    persons = {}
                    next_id = 1
                    for d in dets:
                        assigned = None
                        for pid, info in persons.items():
                            avgx = np.mean([c[0] for c in info["centers"]])
                            avgy = np.mean([c[1] for c in info["centers"]])
                            if ((avgx-d["center"][0])**2+(avgy-d["center"][1])**2)**0.5 < 140:
                                assigned = pid
                                break
                        if not assigned:
                            assigned = next_id
                            persons[assigned] = {"classes": set(), "centers": [], "bboxes": [], "qr_assigned": None}
                            next_id += 1
                        persons[assigned]["classes"].add(d["class"])
                        persons[assigned]["centers"].append(d["center"])
                        persons[assigned]["bboxes"].append(d["bbox"])

                    # Asociar QRs con personas basado en distancia
                    for qr_data, qr_x, qr_y in qr_list:
                        best_pid = None
                        best_dist = float('inf')
                        for pid, info in persons.items():
                            if info["centers"]:
                                avgx = np.mean([c[0] for c in info["centers"]])
                                avgy = np.mean([c[1] for c in info["centers"]])
                                dist = ((avgx-qr_x)**2 + (avgy-qr_y)**2)**0.5
                                if dist < best_dist and dist < 200:
                                    best_dist = dist
                                    best_pid = pid
                        if best_pid is not None:
                            persons[best_pid]["qr_assigned"] = qr_data
                            qr_map[f"qr_{best_pid}"] = qr_data
                            
                    # Si hay QRs pero no se asociaron a ninguna persona específica,
                    # asignar el primero como "any" y a todas las personas sin QR
                    if qr_list and not any(info["qr_assigned"] for info in persons.values()):
                        first_qr = qr_list[0][0]
                        qr_map["any"] = first_qr
                        for pid, info in persons.items():
                            if not info["qr_assigned"]:
                                qr_map[f"qr_{pid}"] = first_qr
                                info["qr_assigned"] = first_qr

                    # Dibujar boxes y determinar colores según EPP
                    for pid, info in persons.items():
                        # Comprobar EPP - no debe tener elementos negativos
                        has_casco = any("casco" in c and "sin" not in c for c in info["classes"])
                        has_chaleco = any("chaleco" in c and "sin" not in c for c in info["classes"])
                        
                        # Colores más visibles:
                        # - Verde (0,255,0): Todo OK
                        # - Amarillo (0,255,255): Falta EPP
                        # - Rojo (0,0,255): Falta todo EPP
                        if has_casco and has_chaleco:
                            color = (0,255,0)  # Verde
                        elif has_casco or has_chaleco:
                            color = (0,255,255)  # Amarillo
                        else:
                            color = (0,0,255)  # Rojo
                        
                        # Dibujar box combinado para la persona
                        bx = [b for b in info["bboxes"]]
                        if bx:
                            x1 = min(b[0] for b in bx)
                            y1 = min(b[1] for b in bx)
                            x2 = max(b[2] for b in bx)
                            y2 = max(b[3] for b in bx)
                            
                            # Texto adicional si hay QR asignado
                            label = f"Persona {pid}"
                            if info["qr_assigned"]:
                                user = get_user_by_qr(info["qr_assigned"])
                                if user:
                                    label += f" - {user[1]}"
                            
                            draw_box(annotated, (x1,y1,x2,y2), label, color)
                            
                            # Mostrar estado EPP
                            epp_status = []
                            if not has_casco:
                                epp_status.append("Sin casco")
                            if not has_chaleco:
                                epp_status.append("Sin chaleco")
                            if epp_status:
                                status_text = " + ".join(epp_status)
                                cv2.putText(annotated, status_text, 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, color, 2)

                    # Actualizar detected_classes con sufijos de persona
                    detected_classes = []
                    for pid, info in persons.items():
                        for c in info["classes"]:
                            detected_classes.append(f"{c}_{pid}")
                        if info["qr_assigned"]:
                            detected_classes.append(f"qr_{pid}")
    except Exception as e:
        print("Error YOLO:", e)
    # Overlay general status (timestamp + detections count)
    try:
        ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        det_count = len(detected_classes) if detected_classes else 0
        status = f"{ts_text} | Detecciones: {det_count}"
        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (6,6), (6 + tw + 8, 6 + th + 8), (0,0,0), -1)
        cv2.putText(annotated, status, (10, 6 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    except Exception:
        pass
    return annotated, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"classes_detected":detected_classes,"qr_map":qr_map,"camera":camera}

# -----------------------------
# Procesador de video
# -----------------------------
class DetectorProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()  # Importante: llamar al constructor padre
        self.model = st.session_state.get("model", None)
        self.queue = st.session_state.get("event_queue", None)
        self.frame_count = 0
        self.last_debug = time.time()
        self.last_log = time.time()
        self.last_detection = None
        self.processing_enabled = True
        
        # Verificar y configurar modelo
        if self.model is None:
            print("⚠️ ADVERTENCIA: Modelo no disponible en DetectorProcessor (se está cargando en background)")
            # No recargar de forma síncrona aquí para evitar bloquear el hilo de vídeo.
            # El loader en background actualizará `st.session_state.model` cuando esté listo.
        else:
            print("✅ Modelo cargado correctamente en DetectorProcessor")
        
        # Configurar el modelo para máximo rendimiento y detección
        if self.model:
            self.model.conf = 0.15  # Umbral de confianza más bajo para detectar más
            self.model.iou = 0.45   # IOU threshold
            self.model.max_det = 50  # Máximo de detecciones
            self.model.agnostic = True  # NMS agnóstico a la clase
            print("✅ Configuración del modelo optimizada para detección")
            if hasattr(self.model, "names"):
                print("📋 Clases disponibles:", self.model.names)
            # Ajustes para modo rápido/ligero según la fuente del modelo
            # Si el loader estableció model_source y es el fallback 'yolov8n.pt', usar modo ligero
            source = st.session_state.get('model_source')
            if source == 'yolov8n.pt':
                self.infer_every = 6
                self.imgsz = 320
                self.conf = 0.25
            else:
                self.infer_every = 3
                self.imgsz = 640
                self.conf = 0.25
        else:
            # Si no hay modelo, mantener modo muy ligero para no bloquear
            self.infer_every = 10
            self.imgsz = 320
            self.conf = 0.25
            
    def recv(self, frame: av.VideoFrame):
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            # Si la detección está desactivada desde UI, devolver frame sin procesar
            if not st.session_state.get('detection_enabled', True):
                try:
                    return frame
                except Exception:
                    return frame
            # Re-check session model and its configuration
            session_model = st.session_state.get("model", None)
            if session_model is not None and session_model != self.model:
                self.model = session_model
                self.model.conf = 0.15  # Reset confidence threshold on model change
                print("✨ Modelo actualizado desde session_state")
                if hasattr(self.model, "names"):
                    print("📋 Clases disponibles:", self.model.names)

            if self.model is None:
                if self.frame_count % 30 == 0:  # Avisar cada 30 frames
                    print("⚠️ Modelo no disponible")
                return frame
            
            # Procesar cada N-ésimo frame para ahorrar CPU/GPU
            if self.frame_count % self.infer_every != 0:
                # No inferir en este frame; devolver rápidamente el frame original
                try:
                    return frame
                except Exception:
                    return frame

            annotated, payload = process_frame(img, self.model, imgsz=self.imgsz, conf=self.conf)
            
            # Debug: imprimir dimensiones del frame
            if self.frame_count % 100 == 0:
                h, w = img.shape[:2]
                print(f"Dimensiones del frame: {w}x{h}")
            
            # Verificar si hay detecciones
            if payload.get("classes_detected"):
                self.last_detection = time.time()
                current_time = time.time()
                
                # Enviar a la cola de eventos si hay detecciones
                if self.queue:
                    if current_time - self.last_log > 0.5:
                        self.queue.put(payload)
                        self.last_log = current_time
                        
                # Debug: mostrar detecciones
                if self.frame_count % 30 == 0:  # Cada 30 frames
                    print(f"Detecciones: {payload['classes_detected']}")
                    if payload.get('qr_map'):
                        print(f"QR: {payload['qr_map']}")
            else:
                # Debug periódico si no hay detecciones
                current_time = time.time()
                if current_time - self.last_debug > 5.0:  # Cada 5 segundos
                    print("👁 Procesando frames pero sin detecciones")
                    self.last_debug = current_time
            
            # Asegurar que el frame anotado se convierte correctamente
            try:
                result_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                return result_frame
            except Exception as e:
                print(f"Error en conversión de frame: {e}")
                return frame
                
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            return frame
        
    def on_ended(self):
        # Limpieza al terminar
        print("Detector finalizado")

# -----------------------------
# Interfaz Streamlit
# -----------------------------
with st.sidebar:
    st.header("Acceso / Controles")
    # Detección siempre activada por defecto (modo demo para cliente)
    st.session_state.detection_enabled = True

    if "user" not in st.session_state:
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        if st.button("Iniciar sesión"):
            user = authenticate_user(u, p)
            if user:
                st.session_state.user = user
                st.success(f"Bienvenido {user[1]}")
            else:
                st.error("Credenciales inválidas")
    else:
        st.success(f"Conectado: {st.session_state.user[1]} ({st.session_state.user[2]})")
        if st.button("Cerrar sesión"):
            try:
                del st.session_state["user"]
                st.info("Sesión cerrada. Actualice la página si es necesario.")
            except Exception:
                pass

    st.markdown("---")
    # Establecer confianza fija en 0.9 (90%)
    st.session_state.confidence = 0.9

    # Solo mostrar la sección de usuarios si está autenticado
    if "user" in st.session_state:
        st.markdown("---")
        st.subheader("Usuarios / QR")
        try:
            users = list_users()
        except Exception:
            users = []
        
        if users:
            user_map = {u[1]: u for u in users}
            sel_user = st.selectbox("Seleccionar usuario", ["(ninguno)"] + list(user_map.keys()))
            
            if sel_user != "(ninguno)":
                target = user_map[sel_user]
                current = st.session_state.get("user")
                can_dl = False
                if current:
                    role, uname = current[2], current[1]
                    # Administrador: puede descargar su propio QR y los de supervisores/clients,
                    # pero NO puede descargar QR de otros administradores.
                    if role == "admin":
                        if target[2] != "admin" or target[1] == uname:
                            can_dl = True
                    elif role == "supervisor" and (target[2] == "client" or target[1] == uname):
                        can_dl = True
                    elif role == "client" and target[1] == uname:
                        can_dl = True
                if can_dl:
                    payload = f"{target[1]}|{target[0]}"
                    qr_img = qrcode.make(payload)
                    buf = io.BytesIO()
                    qr_img.save(buf)
                    buf.seek(0)
                    # Añadir key único para evitar StreamlitDuplicateElementId cuando se muestra repetidamente
                    st.download_button("Descargar QR", data=buf, file_name=f"QR_{target[1]}.png", mime="image/png", key=f"download_qr_{target[0]}_{target[1]}")
                else:
                    st.warning("No tienes permiso para este QR.")
    st.markdown("---")
    # Botón para reintentar cargar el modelo personalizado (best.pt)
    if st.button("Reintentar cargar modelo personalizado (best.pt)"):
        def _reload_custom():
            st.session_state.model_loading = True
            # Intentar cargar desde rutas conocidas (igual que al inicio)
            try_paths = [
                MODEL_PATH,
                os.path.join(PROJECT_ROOT, 'runs/detect/train9/weights/best.pt'),
                os.path.join(PROJECT_ROOT, 'runs/detect/train/weights/best.pt'),
                os.path.join(PROJECT_ROOT, 'best.pt')
            ]
            loaded = False
            for p in try_paths:
                try:
                    if os.path.exists(p):
                        print(f"[reload] Intentando cargar modelo desde: {p}")
                        m = YOLO(p)
                        m.to('cpu')
                        st.session_state.model = m
                        st.session_state.model_source = p
                        loaded = True
                        print(f"[reload] Modelo cargado desde: {p}")
                        break
                except Exception as e:
                    print(f"[reload] Error cargando {p}: {e}")
            if not loaded:
                print("[reload] No se pudo cargar un modelo personalizado.")
                st.session_state.model = None
            st.session_state.model_loading = False
        threading.Thread(target=_reload_custom, daemon=True).start()
        

# -----------------------------
# Layout principal
# -----------------------------
col1, col2 = st.columns([4,1])  # Aumentar proporción de la columna del video

with col1:
    st.header("Cámara en vivo")
    if "user" not in st.session_state:
        st.warning("⚠️ Debe iniciar sesión para ver las detecciones de EPP y QR")
        st.write("La cámara estará disponible pero sin detecciones activas.")
    
    # Contenedor para el video con CSS personalizado y máximo tamaño
    video_container = st.container()
    video_container.markdown("""
        <style>
            /* Hacer el contenedor de video lo más grande posible */
            .stVideo {
                position: relative !important;
                width: 100% !important;
                height: calc(100vh - 100px) !important;
                max-width: none !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            /* Asegurar que el video llene el contenedor */
            .stVideo > video,
            div[data-testid="stVideo"] video,
            .component-frame video {
                width: 100% !important;
                height: 100% !important;
                object-fit: cover !important;
                max-height: none !important;
            }
            
            /* Maximizar contenedores padres */
            .streamlit-expanderContent {
                width: 100% !important;
                max-width: none !important;
            }
            
            div.element-container {
                width: 100% !important;
                max-width: none !important;
            }
            
            /* Ocultar elementos innecesarios para maximizar espacio */
            .stVideo button {
                opacity: 0.7;
                transition: opacity 0.2s;
            }
            
            .stVideo button:hover {
                opacity: 1;
            }
            
            /* Ajustar layout general */
            section.main > div {
                max-width: none !important;
                padding: 0 !important;
            }
            
            [data-testid="column"] {
                padding: 0 !important;
                margin: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    with video_container:
        st.write("Presiona Start y permite acceso a la cámara.")
        # Ensure streamlit module object in sys.modules has experimental_rerun stub
        try:
            import sys
            if 'streamlit' in sys.modules:
                mod = sys.modules['streamlit']
                if not hasattr(mod, 'experimental_rerun'):
                    mod.experimental_rerun = st.experimental_rerun
            # Also try to patch streamlit_webrtc internal reference if present
            if 'streamlit_webrtc.component' in sys.modules:
                comp = sys.modules['streamlit_webrtc.component']
                if hasattr(comp, 'st') and not hasattr(comp.st, 'experimental_rerun'):
                    comp.st.experimental_rerun = st.experimental_rerun
        except Exception:
            pass

        ctx = webrtc_streamer(
            key="safebuild-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=DetectorProcessor,
            media_stream_constraints={
                # Solicitar el acceso por defecto al dispositivo de cámara (mínimas restricciones)
                "video": True,
                "audio": False
            },
            # Restaurar servidores STUN públicos para asegurar negociación WebRTC funcional
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ],
                "bundlePolicy": "max-bundle",
                "iceCandidatePoolSize": 0
            },
            video_frame_callback=None,
            async_processing=True
        )
    if ctx and ctx.state.playing:
        st.info("Transmisión activa.")
    else:
        st.warning("Inactiva o en espera de permiso.")

    # Mostrar estado de carga del modelo si aún está en background
    if st.session_state.get("model_loading"):
        st.info("🔄 Modelo de detección cargándose en segundo plano. Las detecciones aparecerán cuando termine la carga.")

# consumir cola
queue_obj = st.session_state.get("event_queue")
if queue_obj:
    for _ in range(20):
        try:
            ev = queue_obj.get_nowait()
        except Empty:
            break
        try:
            st.session_state.subject.detect_event(
                ev.get("classes_detected", []),
                ev.get("camera", "webcam"),
                user_identified=ev.get("qr_map"),
                evidence_path=None,
            )
        except Exception as e:
            print("Error procesando evento:", e)

with col2:
    st.header("Alertas / Acciones")
    rec = st.session_state.alert_messages[-40:]
    if rec:
        for m in reversed(rec):
            st.write(m)
    else:
        st.write("Sin alertas.")
    st.markdown("---")
    if st.button("Generar reporte (CSV)"):
        try:
            out = os.path.join(PROJECT_ROOT, f"reporte_{int(time.time())}.csv")
            p = generate_report(output_path=out)
            st.success(f"Reporte generado: {p}")
        except Exception as e:
            st.error(f"Error generando reporte: {e}")

st.caption("SafeBuild Web en vivo. Permita cámara y espere detección.")
