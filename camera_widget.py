# camera_widget.py - ACTUALIZADO
import cv2
import sys
import time
import os
import logging
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
from datetime import datetime
try:
    import pyzbar.pyzbar as pyzbar
except Exception:
    pyzbar = None
from config import MODEL_PATH, DEFAULT_CONF, ALERT_COOLDOWN, QR_COOLDOWN
from config import CAPTURES_DIR
import numpy as np
from observer import SafetyMonitorSubject
from database import get_user_by_qr

logging.basicConfig(
    filename='incidentes.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CameraWidget(QWidget):
    _available_indices = None
    _next_index_idx = 0

    def __init__(self, title, camera_index=None, subject=None, log_widget=None, ranking_counter=None):
        super().__init__()
        self.title = title
        self.camera_index = camera_index
        self.subject = subject
        self.log_widget = log_widget
        self.ranking_counter = ranking_counter
        self.ultima_alerta = 0
        self.cap = None
        self.model = None
        self.backend = None
        self.has_camera = False
        self.last_qr_detection = {}
        try:
            self.qr_cooldown = int(QR_COOLDOWN)
        except Exception:
            self.qr_cooldown = 5  # segundos entre detecciones de QR (fallback)

        # Configurar UI
        self.setup_ui()

        # Cargar modelo YOLO (fallback seguro)
        try:
            if MODEL_PATH and os.path.exists(MODEL_PATH):
                self.model = YOLO(MODEL_PATH)
                logging.info(f"Modelo cargado para {self.title}: {MODEL_PATH}")
            else:
                logging.warning(f"Modelo no encontrado en {MODEL_PATH}. Intentando cargar yolov8n.pt para {self.title}")
                try:
                    self.model = YOLO('yolov8n.pt')
                except Exception:
                    logging.warning('No fue posible cargar un modelo YOLO. Se operará sin detección.')
                    self.model = None
        except Exception as e:
            logging.error(f"Error al cargar modelo YOLO para {self.title}: {str(e)}")
            self.model = None

        # Inicializar cámara si tiene índice asignado
        if self.camera_index is not None:
            self.initialize_camera()
            if self.cap and self.cap.isOpened():
                self.has_camera = True
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(30)
            else:
                self.show_no_signal()
        else:
            self.has_camera = False
            self.show_no_signal()

        # Timer de reintento para cámaras que deberían tener señal
        if self.camera_index is not None:
            self.recheck_timer = QTimer()
            self.recheck_timer.timeout.connect(self.recheck_camera)
            self.recheck_timer.start(5000)

    def setup_ui(self):
        """Configura los elementos de la interfaz de usuario."""
        self.label = QLabel()
        self.label.setStyleSheet("background-color: black; color: white;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(320, 240)

        self.titleLabel = QLabel(self.title)
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.titleLabel.setFont(QFont("Arial", 14, QFont.Bold))
        self.titleLabel.setStyleSheet("color: white; background-color: #444; padding: 5px;")

        self.alertaTimerLabel = QLabel()
        self.alertaTimerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alertaTimerLabel.setStyleSheet("color: #FFA500; font-size: 10pt;")

        layout = QVBoxLayout()
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.label)
        layout.addWidget(self.alertaTimerLabel)
        self.setLayout(layout)

    @classmethod
    def detect_available_cameras(cls, prefer_external=True, max_index=10):
        """Detecta cámaras disponibles. Retorna lista de índices funcionales."""
        available = []
        backends = [None, cv2.CAP_DSHOW] if sys.platform.startswith('win') else [None, cv2.CAP_V4L2]

        for idx in range(max_index):
            for backend in backends:
                cap = None
                try:
                    if backend is None:
                        cap = cv2.VideoCapture(idx)
                    else:
                        cap = cv2.VideoCapture(idx, backend)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available.append(idx)
                            logging.info(f"Cámara detectada en índice {idx}")
                        cap.release()
                        break
                except Exception as e:
                    if cap:
                        cap.release()
                    logging.error(f"Error detectando cámara {idx}: {str(e)}")

        cls._available_indices = available
        cls._next_index_idx = 0
        print(f"Cámaras disponibles: {available}")
        return available

    def initialize_camera(self):
        """Inicializa la cámara solo si tiene índice asignado."""
        if self.camera_index is None:
            self.show_no_signal()
            return

        backends = [None, cv2.CAP_DSHOW] if sys.platform.startswith('win') else [None, cv2.CAP_V4L2]

        for backend in backends:
            try:
                if backend is None:
                    self.cap = cv2.VideoCapture(self.camera_index)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.backend = backend
                    logging.info(f"Cámara {self.title} inicializada en índice {self.camera_index}")
                    return
                else:
                    if self.cap:
                        self.cap.release()
            except Exception as e:
                logging.error(f"Error al inicializar cámara {self.camera_index}: {str(e)}")
                if self.cap:
                    self.cap.release()

        logging.error(f"No se pudo inicializar cámara {self.title} en índice {self.camera_index}")
        self.cap = None
        self.show_no_signal()

    def detect_qr_codes(self, frame):
        detected_users = []
        try:
            # Detectar QR solo si pyzbar está disponible
            detected_objects = []
            if pyzbar is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_objects = pyzbar.decode(gray)
                except Exception:
                    detected_objects = []
            else:
                detected_objects = []

            for obj in detected_objects:
                qr_data = obj.data.decode('utf-8')
                current_time = time.time()

                # Verificar cooldown para este QR
                if qr_data in self.last_qr_detection:
                    if current_time - self.last_qr_detection[qr_data] < self.qr_cooldown:
                        continue

                self.last_qr_detection[qr_data] = current_time

                # Buscar usuario en la base de datos
                user = get_user_by_qr(qr_data)
                user_info = qr_data
                if user:
                    user_info = f"{user[1]} ({user[2]})"

                # Obtener bounding box/centro para emparejar luego
                rect = getattr(obj, 'rect', None)
                cx = cy = None
                x = y = w = h = None
                if rect:
                    try:
                        x, y, w, h = rect
                        cx = int(x + w / 2)
                        cy = int(y + h / 2)
                        # Dibujar rect
                        try:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        except Exception:
                            pass
                    except Exception:
                        x = y = w = h = None

                # polygon fallback
                if (cx is None or cy is None) and hasattr(obj, 'polygon'):
                    try:
                        pts = np.array([[p.x, p.y] for p in obj.polygon], dtype=np.int32)
                        M = cv2.moments(pts)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                    except Exception:
                        pass

                detected_users.append((user_info, cx, cy, (x, y, w, h) if rect else None))

                # Mostrar información del usuario si tenemos coords
                try:
                    if cx is not None and cy is not None:
                        cv2.putText(frame, user_info, (max(5, cx - 50), max(20, cy - 10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

                # En vez de escribir directamente al widget (causa ruido), enviar un evento INFO
                # al Subject para que el AlertLogger lo procese y aplique deduplicación.
                try:
                    if self.subject:
                        ev = {
                            'alert_type': '',
                            'camera': self.title,
                            'severity': 0,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'user_identified': user_info,
                            'description': f'QR detectado - {user_info}',
                            'evidence_path': None,
                            'classes_detected': []
                        }
                        # notify will dedupe identical INFO events
                        self.subject.notify(ev)
                except Exception:
                    # Fallback: si falla Subject, mantener la escritura directa para compatibilidad
                    if self.log_widget:
                        self.log_widget.append(f"<span style='color: cyan;'>[QR] {self.title}: Usuario identificado - {user_info}</span>")

        except Exception as e:
            logging.error(f"Error en detección QR {self.title}: {str(e)}")

        return detected_users, frame

    def update_frame(self):
        """Actualiza el frame con detección YOLO y QR."""
        if not self.has_camera or not self.cap or not self.cap.isOpened():
            self.show_no_signal()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.show_no_signal()
            return

        try:
            # Detectar códigos QR (retorna lista de (user_info, cx, cy, rect))
            qr_map_by_pos, frame = self.detect_qr_codes(frame)

            # Procesar con YOLO si tiene modelo (manejo de errores)
            annotated_frame = frame
            clases_detectadas = []
            if self.model is not None:
                try:
                    results = self.model.predict(source=frame, imgsz=416, conf=DEFAULT_CONF, verbose=False)
                    if results and len(results) > 0:
                        res0 = results[0]
                        if hasattr(res0, 'plot'):
                            try:
                                annotated_frame = res0.plot()
                            except Exception:
                                annotated_frame = frame

                        # Extraer cajas y clases y agrupar por persona (clustering simple)
                        try:
                            boxes = getattr(res0, 'boxes', None)
                            clases_detectadas = []
                            persons = {}

                            if boxes is not None and hasattr(boxes, 'xyxy') and hasattr(boxes, 'cls'):
                                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy.numpy()
                                clase_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls.numpy()

                                flat = []
                                for i, box in enumerate(xyxy):
                                    x1, y1, x2, y2 = [int(v) for v in box]
                                    cx = int((x1 + x2) / 2)
                                    cy = int((y1 + y2) / 2)
                                    cname = res0.names[int(clase_ids[i])]
                                    flat.append({'class': cname, 'cx': cx, 'cy': cy, 'bbox': (x1, y1, x2, y2)})

                                next_pid = 1
                                for det in flat:
                                    assigned = None
                                    for pid, info in persons.items():
                                        avgx = int(sum([c[0] for c in info['centers']]) / len(info['centers']))
                                        avgy = int(sum([c[1] for c in info['centers']]) / len(info['centers']))
                                        dist = ((avgx - det['cx']) ** 2 + (avgy - det['cy']) ** 2) ** 0.5
                                        if dist < 120:
                                            assigned = pid
                                            break

                                    if assigned is None:
                                        assigned = next_pid
                                        persons[assigned] = {'classes': set(), 'centers': [], 'bboxes': []}
                                        next_pid += 1

                                    persons[assigned]['classes'].add(det['class'])
                                    persons[assigned]['centers'].append((det['cx'], det['cy']))
                                    persons[assigned]['bboxes'].append(det['bbox'])

                            # Matchear QRs por proximidad y asignar usuarios
                            qr_mapping = {}
                            
                            # Primera pasada: asignar QRs a la persona más cercana
                            for qr in qr_map_by_pos:
                                user_info, qx, qy, rect = qr
                                if qx is None or qy is None:
                                    continue
                                
                                best_pid = None
                                best_dist = float('inf')
                                
                                # Encontrar la persona más cercana al QR
                                for pid, info in persons.items():
                                    avgx = int(sum([c[0] for c in info['centers']]) / len(info['centers']))
                                    avgy = int(sum([c[1] for c in info['centers']]) / len(info['centers']))
                                    dist = ((avgx - qx) ** 2 + (avgy - qy) ** 2) ** 0.5
                                    if dist < best_dist and dist < 200:  # Ajustar umbral según necesidad
                                        best_dist = dist
                                        best_pid = pid
                                
                                # Asignar QR a la persona más cercana y a personas cercanas
                                if best_pid is not None:
                                    # Asignar al más cercano
                                    qr_mapping[f'qr_{best_pid}'] = user_info
                                    
                                    # Asignar a otras personas cercanas
                                    for pid, info in persons.items():
                                        if pid != best_pid:
                                            avgx = int(sum([c[0] for c in info['centers']]) / len(info['centers']))
                                            avgy = int(sum([c[1] for c in info['centers']]) / len(info['centers']))
                                            dist = ((avgx - qx) ** 2 + (avgy - qy) ** 2) ** 0.5
                                            if dist < 400:  # Umbral más amplio para personas cercanas
                                                qr_mapping[f'qr_{pid}'] = user_info
                
                            # QRが検出された場合は必ずグローバルマッピングに追加
                            if qr_map_by_pos:
                                # 最も新しいQRを使用
                                qr_mapping['any'] = qr_map_by_pos[0][0]  
                                # すべての人物に対してQRを関連付け
                                for pid in persons.keys():
                                    if f'qr_{pid}' not in qr_mapping:
                                        qr_mapping[f'qr_{pid}'] = qr_map_by_pos[0][0]

                            # Construir clases_detectadas con sufijos por persona
                            clases_detectadas = []
                            for pid, info in persons.items():
                                for c in info['classes']:
                                    clases_detectadas.append(f"{c}_{pid}")
                                if f'qr_{pid}' in qr_mapping:
                                    clases_detectadas.append(f"qr_{pid}")

                            # Fallback: si no agrupó, intentar mantener lista plana
                            if not clases_detectadas and boxes is not None:
                                try:
                                    clase_ids = boxes.cls
                                    clases_detectadas = [res0.names[int(c)] for c in clase_ids]
                                except Exception:
                                    clases_detectadas = []

                            # Notificar eventos de seguridad usando qr_mapping para identificar usuarios por persona
                            if self.subject and clases_detectadas:
                                missing_any = False
                                for pid, info in persons.items():
                                    if 'helmet' not in info['classes']:
                                        missing_any = True
                                    if not (('reflective' in info['classes']) or ('vest' in info['classes'])):
                                        missing_any = True

                                evidence_path = None
                                try:
                                    if missing_any:
                                        # limitar la frecuencia de notificación por cámara para reducir ruido
                                        nowt = time.time()
                                        if nowt - getattr(self, 'ultima_alerta', 0) < 2.0:
                                            # Skip notifying too-frequent alerts for the same camera
                                            pass
                                        else:
                                            if not os.path.exists(CAPTURES_DIR):
                                                os.makedirs(CAPTURES_DIR, exist_ok=True)
                                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                            filename = f"{self.title.replace(' ', '_')}_{timestamp}.jpg"
                                            evidence_path = os.path.join(CAPTURES_DIR, filename)
                                            cv2.imwrite(evidence_path, annotated_frame)
                                            # Pasar el mapping (p.e. {'qr_1':'nombre','qr_2':'otro','any':'...'} )
                                            self.subject.detect_event(clases_detectadas, self.title, qr_mapping, evidence_path)
                                            # actualizar última notificación para esta cámara
                                            try:
                                                self.ultima_alerta = nowt
                                            except Exception:
                                                self.ultima_alerta = time.time()
                                except Exception as e:
                                    logging.error(f"No se pudo guardar evidencia: {e}")
                        except Exception:
                            # Si hay cualquier fallo en la lógica de agrupación, asegurar que clases_detectadas sea lista vacía
                            clases_detectadas = []
                except Exception as e:
                    logging.error(f"Error predict YOLO en {self.title}: {str(e)}")
                    annotated_frame = frame

            # Agregar timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(annotated_frame, timestamp, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Mostrar frame
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            logging.error(f"Error en procesamiento de frame en {self.title}: {str(e)}")
            self.show_no_signal()

    def show_no_signal(self):
        """Muestra mensaje de 'SIN SEÑAL'."""
        if hasattr(self, 'label'):
            self.label.setText("SIN SEÑAL")
            self.label.setStyleSheet("background-color: #222; color: #FF4444; font-size: 22px;")
        
        if hasattr(self, 'alertaTimerLabel'):
            self.alertaTimerLabel.setText("")

    def recheck_camera(self):
        """Reintenta conexión para cámaras que deberían tener señal."""
        if self.camera_index is not None and (not self.cap or not self.cap.isOpened()):
            self.initialize_camera()
            if self.cap and self.cap.isOpened():
                self.has_camera = True
                self.recheck_timer.stop()
                if hasattr(self, 'label'):
                    self.label.setText("")
                    self.label.setStyleSheet("background-color: black; color: white;")

    def closeEvent(self, event):
        """Limpieza al cerrar el widget."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'recheck_timer') and self.recheck_timer.isActive():
            self.recheck_timer.stop()
        event.accept()