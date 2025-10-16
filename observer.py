# observer.py - ACTUALIZADO
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from config import DEFAULT_CONF, ALERT_COOLDOWN
from database import register_incident
from datetime import datetime
import time
class Subject(ABC):
    @abstractmethod
    def attach(self, observer: 'Observer') -> None:
        pass

    @abstractmethod
    def detach(self, observer: 'Observer') -> None:
        pass

    @abstractmethod
    def notify(self, event_data: dict) -> None:
        pass


class SafetyMonitorSubject(Subject):
    """
    Subject para monitoreo de seguridad: Notifica eventos como alertas EPP/intrusión.
    Instancias independientes mantienen su propio cooldown y lista de observers.
    """
    def __init__(self):
        self._state = {}
        self._observers: List['Observer'] = []
        self._last_alert_time: float = 0

    def attach(self, observer: 'Observer') -> None:
        print(f"SafetyMonitorSubject: Attached observer {type(observer).__name__}.")
        self._observers.append(observer)

    def detach(self, observer: 'Observer') -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event_data: dict) -> None:
        ahora = time.time()
        if ahora - self._last_alert_time < ALERT_COOLDOWN:
            print('Cooldown activo: omitiendo notificación')
            return  # Cooldown
        self._last_alert_time = ahora
        
        print(f"SafetyMonitorSubject: Notifying observers with event: {event_data}")
        for observer in list(self._observers):
            try:
                observer.update(self, event_data)
            except Exception as e:
                print(f"Error notifying observer {type(observer).__name__}: {e}")

    def detect_event(self, classes_detected: list, camera_name: str, user_identified: str = None, evidence_path: str = None) -> None:
        """
        Lógica de negocio: Detecta y notifica si falta EPP o intrusión.
        """
        casco = 'helmet' in classes_detected
        chaleco = 'reflective' in classes_detected or 'vest' in classes_detected
        intrusion = any(c.startswith('intrusion') or c.startswith('not_') for c in classes_detected)

        user_info = f" - Usuario: {user_identified}" if user_identified else ""

        if not (casco and chaleco):
            severity = self.calculate_william_fine(prob=0.8, exp=3, cons=10)
            event_data = {
                'alert_type': 'Falta EPP', 
                'camera': camera_name, 
                'severity': severity, 
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_identified': user_identified,
                'description': f"Falta equipo de protección{user_info}",
                'evidence_path': evidence_path
            }
            self.notify(event_data)
        elif intrusion:
            severity = self.calculate_william_fine(prob=0.6, exp=2, cons=7)
            event_data = {
                'alert_type': 'Intrusión', 
                'camera': camera_name, 
                'severity': severity, 
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_identified': user_identified,
                'description': f"Intrusión detectada{user_info}",
                'evidence_path': evidence_path
            }
            self.notify(event_data)

    def calculate_william_fine(self, prob: float, exp: int, cons: int) -> int:
        """Método William Fine para priorizar (Prob x Exp x Cons)."""
        return int(prob * exp * cons)

class Observer(ABC):
    @abstractmethod
    def update(self, subject: Subject, event_data: dict) -> None:
        pass

class AlertLogger(Observer):
    """Observer: Loggea alertas en GUI o consola."""
    def __init__(self, log_widget=None):
        self.log_widget = log_widget

    def update(self, subject: Subject, event_data: dict) -> None:
        color = 'yellow' if event_data['severity'] < 15 else 'red'
        user_info = f" - Usuario: {event_data['user_identified']}" if event_data.get('user_identified') else ""
        msg = f"[{event_data['timestamp']}] {event_data['camera']}: {event_data['alert_type']} (Severidad: {event_data['severity']}){user_info}"
        
        if self.log_widget:
            self.log_widget.append(f"<span style='color: {color};'>🚨 {msg}</span>")
        else:
            print(f"ALERTA: {msg}")

class IncidentRegistrar(Observer):
    """Observer: Registra en DB si severidad > umbral."""
    def update(self, subject: Subject, event_data: dict) -> None:
        try:
            if event_data.get('severity', 0) > 5:  # Umbral configurable
                register_incident(
                    event_data['camera'], 
                    event_data['alert_type'], 
                    event_data['description'],
                    event_data.get('user_identified'),
                    event_data.get('evidence_path')
                )
                print(f"Incidente registrado en DB: {event_data['alert_type']}")
        except Exception as e:
            print(f"Error registrando incidente: {e}")

class RankingUpdater(Observer):
    """Observer: Actualiza ranking de cámaras por incidentes."""
    def __init__(self, ranking_counter: dict):
        self.ranking_counter = ranking_counter

    def update(self, subject: Subject, event_data: dict) -> None:
        camera_name = event_data['camera']
        self.ranking_counter[camera_name] = self.ranking_counter.get(camera_name, 0) + 1
        print(f"Ranking actualizado: {camera_name} = {self.ranking_counter[camera_name]} alertas.")