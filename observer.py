# observer.py - ACTUALIZADO
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config import DEFAULT_CONF, ALERT_COOLDOWN
from database import register_incident
from datetime import datetime
import time
import threading
from typing import Tuple


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
        # track recent events to avoid duplicate notifications (key -> last_time)
        self._recent_event_times: Dict[tuple, float] = {}
        # pending events waiting for possible QR identification nearby
        self._pending_events: Dict[tuple, dict] = {}
        self._pending_lock = threading.Lock()
        # seconds to wait before emitting an alert with unknown user to allow QR to appear
        # Se incrementa a 4s para dar tiempo a que aparezca un QR y reducir 'unknown' frecuentes
        self.PENDING_WINDOW = 4.0
        self._last_known_qr: Dict[str, tuple] = {}

    def attach(self, observer: 'Observer') -> None:
        print(f"SafetyMonitorSubject: Attached observer {type(observer).__name__}.")
        self._observers.append(observer)

    def detach(self, observer: 'Observer') -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event_data: dict) -> None:
        ahora = time.time()
        # NOTE: removed global ALERT_COOLDOWN blocking here to allow per-event
        # deduplication logic below (_recent_event_times) to control repeats.
        # Global cooldown caused legitimate alerts to be suppressed across
        # different events and cameras. We keep _last_alert_time for logging
        # purposes but do not block notifications globally.
        self._last_alert_time = ahora

        # Si se trata de un evento INFO con identificación de usuario, actualizar last_known_qr
        # y, si existen eventos pendientes para la misma cámara, flusharlos reemplazando 'unknown'.
        if (not event_data.get('alert_type')) and event_data.get('user_identified') and event_data.get('user_identified') != 'unknown':
            try:
                cam = event_data.get('camera')
                if cam:
                    self._last_known_qr[cam] = (event_data.get('user_identified'), ahora)
                # Flush pending events for this camera
                with self._pending_lock:
                    keys_to_flush = [k for k in list(self._pending_events.keys()) if k[0] == cam]
                    for k in keys_to_flush:
                        pending_ev = self._pending_events.pop(k, None)
                        if not pending_ev:
                            continue
                        # Update pending event with identified user
                        pending_ev['user_identified'] = event_data.get('user_identified')
                        # improve description if it contained unknown
                        try:
                            pending_ev['description'] = pending_ev.get('description', '').replace('Usuario: unknown', f'Usuario identificado: {event_data.get("user_identified")}')
                        except Exception:
                            pass
                        # Notify observers directly for these flushed events
                        for observer in list(self._observers):
                            try:
                                observer.update(self, pending_ev)
                            except Exception as e:
                                print(f"Error notifying observer during pending flush {type(observer).__name__}: {e}")
            except Exception:
                pass

        # イベントの重複チェックをより厳密に行う
        key = (event_data.get('alert_type'), event_data.get('camera'), 
               event_data.get('user_identified'), 
               ','.join(sorted(event_data.get('classes_detected', []))))  # クラスの検出状態も含める
        last = self._recent_event_times.get(key)
        DUPLICATE_WINDOW = 3.0  # Mostrar como máximo cada 3 segundos por evento
        if last and (ahora - last) < DUPLICATE_WINDOW:
            # 直近の通知をスキップ
            print(f"重複イベントをスキップ: {key}")
            return
        # 通知時刻を記録
        self._recent_event_times[key] = ahora

    # Notify observers (debug prints removed to avoid console encoding issues)
        for observer in list(self._observers):
            try:
                observer.update(self, event_data)
            except Exception as e:
                print(f"Error notifying observer {type(observer).__name__}: {e}")

    def detect_event(self, classes_detected: List[str], camera_name: str,
                     user_identified: Optional[Dict[str, str]] = None,
                     evidence_path: Optional[str] = None) -> None:
        """
        Lógica de negocio: Detecta y notifica si falta EPP o intrusión.
        - `classes_detected` espera items con posición: e.g. 'helmet_1', 'vest_1', 'qr_head_1', 'qr_body_2'
        - `user_identified` es un dict opcional que mapea ubicacion->qr_value: {'qr_1': 'User1', 'qr_2': 'User2', 'any': 'GlobalUser'}
        """
        # Update last known QR for this camera if we have a valid QR
        current_time = time.time()
        if isinstance(user_identified, dict) and user_identified:
            qr_value = user_identified.get('any')
            if qr_value:
                self._last_known_qr[camera_name] = (qr_value, current_time)
        # Normalize user_identified: support dict mapping OR a simple string
        qr_data = {}
        if isinstance(user_identified, dict):
            # Ensure we have both specific mappings and 'any' fallback
            qr_data = dict(user_identified)  # make a copy
            if 'any' in qr_data and not any(k.startswith('qr_') for k in qr_data):
                # If we have an 'any' value but no specific mappings, create them
                for i in range(1, 10):  # reasonable limit
                    if f'qr_{i}' not in qr_data:
                        qr_data[f'qr_{i}'] = qr_data['any']
        elif isinstance(user_identified, str) and user_identified.strip():
            # treat the provided string as a user identifier (e.g., 'admin (admin)')
            qr_data = {'any': user_identified}
            # Also create specific mappings for any potential person
            for i in range(1, 10):  # reasonable limit
                qr_data[f'qr_{i}'] = user_identified

        # Agrupar detecciones por persona (por índice numérico en los nombres)
        people = self._group_detections(classes_detected, qr_data)

        # If grouping produced no per-person records (detector returns global class names
        # without suffixes), synthesize a single person from presence/absence of classes
        if not people:
            has_helmet = 'helmet' in classes_detected
            has_vest = ('vest' in classes_detected) or ('reflective' in classes_detected)
            user_id = None
            # if qr_data has a string under 'any', use it
            if isinstance(qr_data, dict):
                if 'any' in qr_data:
                    user_id = qr_data.get('any')
                else:
                    # fallback to first available mapping value
                    first = next(iter(qr_data.values()), None)
                    user_id = first
            people = [{'helmet': has_helmet, 'vest': has_vest, 'user_id': user_id}]

        # Para cada persona generar alertas por falta de equipo
        for i, person in enumerate(people, 1):
            has_helmet = person.get('helmet', False)
            has_vest = person.get('vest', False)
            user_id = person.get('user_id') or None
            pid = i  # person identifier for QR mapping
            
            # Try to get user from any available source
            effective_user = user_id
            if not effective_user and isinstance(qr_data, dict):
                # First try specific mapping for this person
                if f'qr_{pid}' in qr_data:
                    effective_user = qr_data[f'qr_{pid}']
                # Then try global QR
                elif 'any' in qr_data:
                    effective_user = qr_data['any']
                # Finally check recent QR
                elif camera_name in self._last_known_qr:
                    last_qr, last_time = self._last_known_qr[camera_name]
                    if current_time - last_time <= 5.0:  # 5秒以内のQR
                        effective_user = last_qr

            # Helper to build and possibly buffer/flush event
            def _handle_alert(alert_type: str, description: str):
                severity = self.calculate_william_fine(prob=0.8, exp=3, cons=10)
                
                # Use a local copy to avoid binding issues when assigning inside
                # this nested function. We prefer a local e_user and keep
                # the outer effective_user unchanged.
                e_user = effective_user

                # QRコードが検出されている場合は即座にチェック（遅延なし）
                if not e_user and any('qr' in c.lower() for c in classes_detected):
                    # QRマッピングを即座にチェック
                    if isinstance(qr_data, dict):
                        if 'any' in qr_data:
                            e_user = qr_data['any']
                        elif qr_data:
                            e_user = next(iter(qr_data.values()))
                
                event = {
                    'alert_type': alert_type,
                    'camera': camera_name,
                    'severity': severity,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'user_identified': e_user or 'unknown',
                    'description': description.replace('unknown', e_user) if e_user else description,
                    'evidence_path': evidence_path,
                    'classes_detected': classes_detected,
                }

                # key includes camera and alert_type and a simple fingerprint of classes to reduce collisions
                fingerprint = tuple(sorted([str(x) for x in classes_detected]))
                key: Tuple = (camera_name, alert_type, fingerprint)

                if user_id:
                    # If someone is identified now, flush any pending similar event replacing unknown
                    with self._pending_lock:
                        pending = self._pending_events.pop(key, None)
                    if pending:
                        # update pending with identified user and notify
                        pending['user_identified'] = user_id
                        pending['description'] = description
                        self.notify(pending)
                        return
                    # Otherwise notify immediately for identified user
                    self.notify(event)
                else:
                    # Buffer the event for a short time to allow QR to appear
                    self._add_pending_event(key, event)

            # Check for each type of violation
            violations = []
            if not has_helmet:
                violations.append('Falta Casco')
            if not has_vest:
                violations.append('Falta Chaleco')
            
            # 各人物の違反に対して1回だけアラートを生成
            alert_sent = False  # 追跡フラグ
            for violation in violations:
                if alert_sent:
                    continue  # 既にアラートを送信済みの場合はスキップ
                if effective_user:
                    description = f"{violation} - Usuario identificado: {effective_user}"
                else:
                    description = f"{violation} - Usuario: unknown"
                _handle_alert(violation, description)
                alert_sent = True  # アラート送信済みをマーク

        # No more intrusion detection - we only care about PPE violations
        pass

    def _group_detections(self, classes_detected: List[str], qr_data=None) -> List[Dict[str, Any]]:
        """
        Agrupa las detecciones por persona usando el sufijo numérico en las etiquetas.
        Retorna una lista de dicts: {'helmet': bool, 'vest': bool, 'user_id': str}

        Ejemplo de `classes_detected` esperado:
          ['helmet_1', 'qr_1', 'vest_1', 'qr_2']
        Ejemplo de `qr_data`:
          {'qr_1': 'User1', 'qr_2': 'User2', 'any': 'GlobalUser'}
        """
        people: Dict[int, Dict[str, Any]] = {}

        for item in classes_detected:
            # Esperamos formato con guion bajo y número al final
            if '_' not in item:
                continue
            base, idx = item.rsplit('_', 1)
            if not idx.isdigit():
                continue
            pid = int(idx)
            if pid not in people:
                people[pid] = {'helmet': False, 'vest': False, 'user_id': None}

            # base puede incluir ubicacion, ej 'qr_head' o 'qr_body' o simplemente 'qr'
            if base == 'helmet':
                people[pid]['helmet'] = True
            elif base in ('vest', 'reflective'):
                people[pid]['vest'] = True
            elif base.startswith('qr'):
                # Preferencia: si qr_data contiene una clave específica para este qr (p.e. 'qr_1'), usarla
                key = f"qr_{pid}"
                if isinstance(qr_data, dict) and key in qr_data:
                    people[pid]['user_id'] = qr_data.get(key)
                else:
                    # determinar si el tag contiene la palabra head/body y qr_data tiene esos keys
                    if 'head' in base and isinstance(qr_data, dict) and 'head' in qr_data:
                        people[pid]['user_id'] = qr_data.get('head')
                    elif 'body' in base and isinstance(qr_data, dict) and 'body' in qr_data:
                        people[pid]['user_id'] = qr_data.get('body')
                    else:
                        # si no hay posición en el nombre, intentar asignar el primer valor disponible
                        if isinstance(qr_data, dict) and qr_data:
                            # preferir key='any' o la primera disponible
                            if 'any' in qr_data:
                                people[pid]['user_id'] = qr_data.get('any')
                            else:
                                first = next(iter(qr_data.values()), None)
                                people[pid]['user_id'] = first

        # Devolver lista ordenada por pid para determinismo
        return [people[k] for k in sorted(people.keys())]

    def calculate_william_fine(self, prob: float, exp: int, cons: int) -> int:
        """Método William Fine para priorizar (Prob x Exp x Cons)."""
        return int(prob * exp * cons)

    def _add_pending_event(self, key: tuple, event: dict) -> None:
        """Agrega un evento a la cola de pendientes y lanza un timer para su envío si no llega identificación."""
        ahora = time.time()
        with self._pending_lock:
            # if there is already a pending event, update timestamp to the latest
            existing = self._pending_events.get(key)
            if existing:
                # extend its timestamp (keep earliest timestamp but replace evidence if newer)
                existing['last_seen'] = ahora
                # we don't overwrite user_identified
                return

            event_copy = dict(event)
            event_copy['last_seen'] = ahora
            self._pending_events[key] = event_copy

        # spawn a background thread to flush after PENDING_WINDOW if still pending
        def _flush_later(k: tuple, waited: float):
            time.sleep(waited)
            with self._pending_lock:
                ev = self._pending_events.pop(k, None)
            if ev:
                # final notification (user_identified likely 'unknown')
                # map alert_type names to user-friendly labels
                self.notify(ev)

        t = threading.Thread(target=_flush_later, args=(key, self.PENDING_WINDOW), daemon=True)
        t.start()


class Observer(ABC):
    @abstractmethod
    def update(self, subject: Subject, event_data: dict) -> None:
        pass


class AlertLogger(Observer):
    """Observer: Loggea alertas en GUI o consola."""

    def __init__(self, log_widget=None):
        self.log_widget = log_widget
        self._last_message = {}

    def update(self, subject: Subject, event_data: dict) -> None:
        timestamp = event_data.get('timestamp')
        camera = event_data.get('camera')
        alert_type = event_data.get('alert_type', '')
        user_ident = event_data.get('user_identified')

        # メッセージの重複チェック用のキー
        msg_key = f"{camera}:{alert_type}:{user_ident}"
        current_time = time.time()

        # 2秒以内の同一メッセージはスキップ
        if msg_key in self._last_message:
            if current_time - self._last_message[msg_key] < 2.0:
                return

        # アラートメッセージの生成
        msg = None
        color = None
        if alert_type:
            msg = f"[{timestamp}] {camera}: {alert_type} - Usuario: {user_ident if user_ident != 'unknown' else 'unknown'}"
            color = 'yellow' if event_data.get('severity', 0) < 15 else 'red'
        elif user_ident and user_ident != 'unknown':
            msg = f"[{timestamp}] {camera}: Usuario identificado - {user_ident}"
            color = 'cyan'

        # メッセージの表示
        if msg and self.log_widget:
            color_tag = f"<span style='color: {color};'>"
            if color == 'cyan':
                color_tag += '[INFO] '
            self.log_widget.append(f"{color_tag}{msg}</span>")
        
        # 最終表示時刻を更新
        self._last_message[msg_key] = current_time

        # アラーム音の再生（違反アラートのみ）
        if alert_type.startswith('Falta'):
            def _play_beep():
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    print('\a', end='')
            try:
                t = threading.Thread(target=_play_beep, daemon=True)
                t.start()
            except Exception:
                pass


class IncidentRegistrar(Observer):
    """Observer: Registra en DB si severidad > umbral."""

    def update(self, subject: Subject, event_data: dict) -> None:
        try:
            if event_data.get('severity', 0) > 5:  # Umbral configurable
                register_incident(
                    event_data.get('camera'),
                    event_data.get('alert_type'),
                    event_data.get('description'),
                    event_data.get('user_identified'),
                    event_data.get('evidence_path')
                )
                print(f"Incidente registrado en DB: {event_data.get('alert_type')}")
        except Exception as e:
            print(f"Error registrando incidente: {e}")


class RankingUpdater(Observer):
    """Observer: Actualiza ranking de cámaras por incidentes."""

    def __init__(self, ranking_counter: dict):
        self.ranking_counter = ranking_counter

    def update(self, subject: Subject, event_data: dict) -> None:
        camera_name = event_data.get('camera')
        self.ranking_counter[camera_name] = self.ranking_counter.get(camera_name, 0) + 1
        print(f"Ranking actualizado: {camera_name} = {self.ranking_counter[camera_name]} alertas.")