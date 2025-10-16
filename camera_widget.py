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

    def __init__(self, title, camera_index=None, subject: SafetyMonitorSubject = None, log_widget=None, ranking_counter=None):
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
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(320, 240)

        self.titleLabel = QLabel(self.title)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setFont(QFont("Arial", 14, QFont.Bold))
        self.titleLabel.setStyleSheet("color: white; background-color: #444; padding: 5px;")

        self.alertaTimerLabel = QLabel()
        self.alertaTimerLabel.setAlignment(Qt.AlignCenter)
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
        """Detecta códigos QR en el frame y retorna información de usuarios."""
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
                if user:
                    user_info = f"{user[1]} ({user[2]})"
                    detected_users.append(user_info)
                    
                    # Dibujar rectángulo alrededor del QR (manejo robusto de puntos)
                    points = getattr(obj, 'polygon', None)
                    if points and len(points) > 4:
                        try:
                            pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                            hull = cv2.convexHull(pts)
                            cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
                        except Exception:
                            pass
                    else:
                        rect = getattr(obj, 'rect', None)
                        if rect:
                            try:
                                x, y, w, h = rect
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            except Exception:
                                pass
                    
                    # Mostrar información del usuario (si existen coordenadas)
                    try:
                        cv2.putText(frame, user_info, (x, max(20, y - 10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception:
                        # Si por alguna razón no tenemos coords, no fallar
                        pass
                    
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
            # Detectar códigos QR
            detected_users, frame = self.detect_qr_codes(frame)
            user_identified = ", ".join(detected_users) if detected_users else None

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

                        # Extraer clases detectadas de forma segura
                        try:
                            boxes = getattr(res0, 'boxes', None)
                            if boxes is not None and hasattr(boxes, 'cls'):
                                clase_ids = boxes.cls
                                clases_detectadas = [res0.names[int(c)] for c in clase_ids]
                        except Exception:
                            clases_detectadas = []

                        # Notificar eventos de seguridad
                        if self.subject and clases_detectadas:
                            # Check for missing EPP: no helmet or no vest/reflective
                            missing_helmet = 'helmet' not in clases_detectadas
                            missing_vest = not (('reflective' in clases_detectadas) or ('vest' in clases_detectadas))
                            evidence_path = None
                            try:
                                if missing_helmet or missing_vest:
                                    if not os.path.exists(CAPTURES_DIR):
                                        os.makedirs(CAPTURES_DIR, exist_ok=True)
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    filename = f"{self.title.replace(' ', '_')}_{timestamp}.jpg"
                                    evidence_path = os.path.join(CAPTURES_DIR, filename)
                                    # Write annotated_frame (BGR) as JPEG
                                    cv2.imwrite(evidence_path, annotated_frame)
                            except Exception as e:
                                logging.error(f"No se pudo guardar evidencia: {e}")

                            # Pass evidence_path (may be None) to detect_event
                            self.subject.detect_event(clases_detectadas, self.title, user_identified, evidence_path)
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