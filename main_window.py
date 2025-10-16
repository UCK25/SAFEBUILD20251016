# main_window.py - ACTUALIZADO
import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QVBoxLayout, QTextEdit, QHBoxLayout, 
                             QLabel, QWidget, QPushButton, QInputDialog, QDialog, QLineEdit,
                             QFormLayout, QDialogButtonBox, QMessageBox, QSizePolicy, QScrollArea,
                             QTableWidget, QTableWidgetItem, QComboBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from datetime import datetime
import os
from database import init_db, list_cameras, register_camera, authenticate_user, generate_report, register_user, list_users
from database import delete_user, update_user_role, update_password
from observer import SafetyMonitorSubject, AlertLogger, IncidentRegistrar, RankingUpdater
from camera_widget import CameraWidget
from config import PROJECT_ROOT
from database import list_incidents
from PyQt5.QtGui import QPixmap
from config import CAPTURES_DIR

class AuthenticationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autenticación de Usuario")
        self.setFixedSize(400, 200)  # Interfaz más grande
        
        layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Ingrese su usuario")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Ingrese su contraseña")
        self.password_input.setEchoMode(QLineEdit.Password)
        
        layout.addRow("Usuario:", self.username_input)
        layout.addRow("Contraseña:", self.password_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.authenticate)
        buttons.rejected.connect(self.reject)
        
        layout.addRow(buttons)
        self.setLayout(layout)
    
    def authenticate(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if username and password:
            user = authenticate_user(username, password)
            if user:
                # user is tuple (id, username, role)
                QMessageBox.information(self, "Éxito", f"Usuario {user[1]} autenticado exitosamente!")
                # store user info for caller
                self.authenticated_user = user
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Credenciales incorrectas!")
        else:
            QMessageBox.warning(self, "Error", "Por favor complete todos los campos!")

class UserManagementDialog(QDialog):
    def __init__(self, parent=None, current_user=None):
        super().__init__(parent)
        self.setWindowTitle("Gestión de Usuarios")
        self.setMinimumSize(700, 420)

        self.current_user = current_user

        layout = QVBoxLayout()

        # Table for users
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['ID','Usuario','Rol','QR'])
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        # Improve table and header readability on dark theme
        self.table.setStyleSheet(
            "QHeaderView::section { background-color: #444; color: #FFAA33; font-weight: bold; }"
            "QTableWidget { background-color: #222; color: white; gridline-color: #555; }"
            "QTableWidget::item:selected { background-color: #666; color: white; }"
        )

        # Form to edit/create
        form_layout = QFormLayout()
        self.id_label = QLabel('')
        self.username_input = QLineEdit()
        # role selection should be from predefined roles to avoid typos
        self.role_input = QComboBox()
        self.role_input.addItems(['supervisor', 'admin', 'guest'])
        self.qr_input = QLineEdit()
        form_layout.addRow('ID:', self.id_label)
        form_layout.addRow('Usuario:', self.username_input)
        form_layout.addRow('Rol:', self.role_input)
        form_layout.addRow('QR:', self.qr_input)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton('Agregar')
        self.add_btn.clicked.connect(self.add_user)
        self.update_btn = QPushButton('Actualizar')
        self.update_btn.clicked.connect(self.update_user)
        self.delete_btn = QPushButton('Eliminar')
        self.delete_btn.clicked.connect(self.delete_user)
        self.reset_btn = QPushButton('Resetear Contraseña')
        self.reset_btn.clicked.connect(self.reset_selected_password)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.reset_btn)

        layout.addWidget(QLabel('Usuarios Registrados:'))
        layout.addWidget(self.table)
        layout.addLayout(form_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.load_users()
        self.apply_permissions()
    
    def load_users(self):
        users = list_users()
        self.table.setRowCount(len(users))
        for r, user in enumerate(users):
            self.table.setItem(r, 0, QTableWidgetItem(str(user[0])))
            self.table.setItem(r, 1, QTableWidgetItem(user[1]))
            self.table.setItem(r, 2, QTableWidgetItem(user[2]))
            self.table.setItem(r, 3, QTableWidgetItem(user[3] or ''))
            # If there is a current_user, make that row non-selectable to prevent editing self
            try:
                if self.current_user and int(self.current_user[0]) == int(user[0]):
                    for c in range(self.table.columnCount()):
                        it = self.table.item(r, c)
                        if it is not None:
                            it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
            except Exception:
                pass
    
    def add_user(self):
        username = self.username_input.text().strip()
        role = self.role_input.currentText().strip() or 'supervisor'
        qr = self.qr_input.text().strip() or None
        # For new user we require a password (prompt)
        pwd, ok = QInputDialog.getText(self, 'Password', 'Contraseña para el nuevo usuario:', QLineEdit.Password)
        if not ok or not pwd:
            QMessageBox.warning(self, 'Error', 'Contraseña requerida para nuevo usuario')
            return
        if register_user(username, pwd, qr):
            # set role
            users = list_users()
            for u in users:
                if u[1] == username:
                    update_user_role(u[0], role)
            QMessageBox.information(self, 'OK', 'Usuario agregado')
            self.load_users()
        else:
            QMessageBox.warning(self, 'Error', 'No se pudo agregar el usuario (dup?)')

    def on_row_selected(self):
        sel = self.table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        uid = self.table.item(row, 0).text()
        self.id_label.setText(uid)
        self.username_input.setText(self.table.item(row,1).text())
        # set combo to the current role if present
        role_text = self.table.item(row,2).text()
        idx = self.role_input.findText(role_text)
        if idx >= 0:
            self.role_input.setCurrentIndex(idx)
        else:
            # if role not present, add it temporarily and select
            self.role_input.addItem(role_text)
            self.role_input.setCurrentIndex(self.role_input.count()-1)
        self.qr_input.setText(self.table.item(row,3).text())

    def update_user(self):
        uid = self.id_label.text()
        if not uid:
            QMessageBox.warning(self, 'Seleccionar', 'Seleccione un usuario')
            return
        username = self.username_input.text().strip()
        role = self.role_input.currentText().strip()
        qr = self.qr_input.text().strip() or None
        from database import update_user
        ok = update_user(int(uid), username=username, qr_code=qr)
        if ok:
            update_user_role(int(uid), role)
            QMessageBox.information(self, 'OK', 'Usuario actualizado')
            self.load_users()
        else:
            QMessageBox.warning(self, 'Error', 'No se pudo actualizar (dup?)')

    def delete_user(self):
        uid = self.id_label.text()
        if not uid:
            QMessageBox.warning(self, 'Seleccionar', 'Seleccione un usuario')
            return
        from database import delete_user
        if delete_user(int(uid)):
            QMessageBox.information(self, 'OK', 'Usuario eliminado')
            self.load_users()
            self.id_label.setText('')
            self.username_input.clear()
            # reset role_input to default first entry if present
            try:
                if isinstance(self.role_input, QComboBox) and self.role_input.count() > 0:
                    self.role_input.setCurrentIndex(0)
            except Exception:
                pass
            self.qr_input.clear()
        else:
            QMessageBox.warning(self, 'Error', 'No se pudo eliminar')

    def apply_permissions(self):
        # Only admin and supervisor can perform user CRUD
        role = None
        if self.current_user:
            role = self.current_user[2]
        allowed = (str(role).lower() in ('admin', 'supervisor'))
        # Buttons
        self.delete_btn.setEnabled(allowed)
        self.reset_btn.setEnabled(allowed)
        self.add_btn.setEnabled(allowed)
        self.update_btn.setEnabled(allowed)
        # Inputs
        self.username_input.setEnabled(allowed)
        self.role_input.setEnabled(allowed)
        self.qr_input.setEnabled(allowed)
        # Do not show intrusive message on init; caller should guard opening the dialog.

    def get_selected_user_id(self):
        # Prefer explicit id_label if a row was selected
        if self.id_label.text():
            try:
                return int(self.id_label.text())
            except Exception:
                pass
        # Fallback to table selection
        sel = self.table.selectedItems()
        if not sel:
            QMessageBox.warning(self, 'Seleccionar', 'Seleccione la fila del usuario en la lista.')
            return None
        row = sel[0].row()
        try:
            return int(self.table.item(row, 0).text())
        except Exception:
            QMessageBox.warning(self, 'Error', 'No se pudo determinar el ID seleccionado.')
            return None

    def reset_selected_password(self):
        uid = self.get_selected_user_id()
        if uid is None:
            return
        # Ask for new password
        new_pass, ok = QInputDialog.getText(self, 'Reset Password', 'Nueva contraseña:', QLineEdit.Password)
        if ok and new_pass:
            from database import reset_user_password
            actor = None
            if self.current_user:
                actor = self.current_user[1]
            if reset_user_password(actor, uid, new_pass):
                QMessageBox.information(self, 'OK', 'Contraseña actualizada y registrada en auditoría')
                self.load_users()
            else:
                QMessageBox.warning(self, 'Error', 'No se pudo actualizar la contraseña')

    def change_selected_role(self):
        uid = self.get_selected_user_id()
        if uid is None:
            return
        # Ask for role (admin/supervisor/guest)
        roles = ['admin', 'supervisor', 'guest']
        role, ok = QInputDialog.getItem(self, 'Cambiar Rol', 'Seleccionar rol:', roles, 0, False)
        if ok and role:
            from database import update_user_role
            if update_user_role(uid, role):
                QMessageBox.information(self, 'OK', 'Rol actualizado')
                self.load_users()
            else:
                QMessageBox.warning(self, 'Error', 'No se pudo actualizar el rol')


class CameraManagementDialog(QDialog):
    def __init__(self, parent=None, current_user=None):
        super().__init__(parent)
        self.setWindowTitle('Gestión de Cámaras')
        self.setMinimumSize(650, 420)
        self.current_user = current_user

        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['ID','Nombre','Índice'])
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        # Style table for dark theme and readable headers (match users table)
        self.table.setStyleSheet(
            "QHeaderView::section { background-color: #444; color: #FFAA33; font-weight: bold; }"
            "QTableWidget { background-color: #222; color: white; gridline-color: #555; }"
            "QTableWidget::item:selected { background-color: #666; color: white; }"
        )

        form = QFormLayout()
        self.cam_id_label = QLabel('')
        self.cam_name = QLineEdit()
        self.cam_index = QLineEdit()
        form.addRow('ID:', self.cam_id_label)
        form.addRow('Nombre:', self.cam_name)
        form.addRow('Índice:', self.cam_index)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton('Agregar')
        self.add_btn.clicked.connect(self.add_camera)
        self.update_btn = QPushButton('Actualizar')
        self.update_btn.clicked.connect(self.edit_selected_camera)
        self.delete_btn = QPushButton('Eliminar')
        self.delete_btn.clicked.connect(self.delete_selected_camera)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.delete_btn)

        layout.addWidget(QLabel('Cámaras Registradas:'))
        layout.addWidget(self.table)
        layout.addLayout(form)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.load_cameras()
        self.apply_permissions()

    def apply_permissions(self):
        role = None
        if self.current_user:
            role = self.current_user[2]
        allowed = (str(role).lower() in ('admin', 'supervisor'))
        self.delete_btn.setEnabled(allowed)
        # update_btn exists (rename of edit)
        try:
            self.update_btn.setEnabled(allowed)
        except Exception:
            pass
        self.cam_name.setEnabled(allowed)
        self.cam_index.setEnabled(allowed)
        # Do not show intrusive message on init; caller should guard opening the dialog.

    def load_cameras(self):
        cams = list_cameras()
        self.table.setRowCount(len(cams))
        for r, c in enumerate(cams):
            self.table.setItem(r, 0, QTableWidgetItem(str(c[0])))
            self.table.setItem(r, 1, QTableWidgetItem(c[1]))
            self.table.setItem(r, 2, QTableWidgetItem(str(c[2]) if c[2] is not None else ''))
        try:
            # make columns fit content for readability
            self.table.resizeColumnsToContents()
        except Exception:
            pass

    def add_camera(self):
        name = self.cam_name.text().strip() or f'cámara {self.table.rowCount()+1}'
        try:
            idx = int(self.cam_index.text()) if self.cam_index.text().strip() != '' else None
        except Exception:
            idx = None
        try:
            register_camera(name, idx, None)
            QMessageBox.information(self, 'OK', 'Cámara agregada')
            self.load_cameras()
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'No se pudo agregar la cámara: {e}')

    def get_selected_camera_id(self):
        sel = self.table.selectedItems()
        if not sel:
            QMessageBox.warning(self, 'Seleccionar', 'Seleccione la fila de la cámara en la lista.')
            return None
        row = sel[0].row()
        return int(self.table.item(row,0).text())

    def on_row_selected(self):
        sel = self.table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        self.cam_id_label.setText(self.table.item(row,0).text())
        self.cam_name.setText(self.table.item(row,1).text())
        self.cam_index.setText(self.table.item(row,2).text())

    def delete_selected_camera(self):
        cid = self.get_selected_camera_id()
        if cid is None:
            return
        try:
            import sqlite3
            from config import DB_PATH
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute('DELETE FROM cameras WHERE id=?', (cid,))
            conn.commit()
            conn.close()
            QMessageBox.information(self, 'OK', 'Cámara eliminada')
            self.load_cameras()
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'No se pudo eliminar la cámara: {e}')

    def edit_selected_camera(self):
        cid = self.get_selected_camera_id()
        if cid is None:
            return
        name = self.cam_name.text().strip()
        try:
            idx = int(self.cam_index.text()) if self.cam_index.text().strip() != '' else None
        except Exception:
            idx = None
        try:
            import sqlite3
            from config import DB_PATH
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute('UPDATE cameras SET name=?, camera_index=? WHERE id=?', (name, idx, cid))
            conn.commit()
            conn.close()
            QMessageBox.information(self, 'OK', 'Cámara actualizada')
            self.load_cameras()
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'No se pudo actualizar la cámara: {e}')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SafeBuild AI - Sistema de Monitoreo de Seguridad")
        self.setStyleSheet("background-color: #333; color: white;")
        
        # Hacer la ventana más grande
        self.setMinimumSize(1200, 800)

        init_db()  # Inicializar DB

        # Panel derecho: estado de cámaras (historial) y panel de alertas con evidencia
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("background-color: black; color: white; font-family: Consolas; font-size: 10pt;")
        self.status_log.setMinimumHeight(180)

        # Alert / evidence display area
        self.alert_text = QTextEdit()
        self.alert_text.setReadOnly(True)
        self.alert_text.setStyleSheet("background-color: #111; color: orange; font-family: Consolas; font-size: 10pt;")
        self.alert_image = QLabel()
        self.alert_image.setFixedSize(320, 240)
        self.alert_image.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.alert_image.setAlignment(Qt.AlignCenter)

        # Subject y Observers (Patrón Observer)
        self.subject = SafetyMonitorSubject()
        # Keep alert logger but also create a small adapter to update the alert_text/alert_image
        self.alert_logger = AlertLogger(self.alert_text)
        self.incident_registrar = IncidentRegistrar()
        self.subject.attach(self.alert_logger)
        self.subject.attach(self.incident_registrar)

        # Attach a GUI observer to update image and status area
        class GUIAlertObserver:
            def __init__(self, mainwin):
                self.mainwin = mainwin
            def update(self, subject, event_data):
                # update alert text
                user = event_data.get('user_identified') or 'unknown'
                msg = f"[{event_data['timestamp']}] {event_data['camera']}: {event_data['alert_type']} - {user}"
                self.mainwin.alert_text.append(msg)
                ev = event_data.get('evidence_path')
                if ev and ev != '':
                    try:
                        pix = QPixmap(ev)
                        if not pix.isNull():
                            scaled = pix.scaled(self.mainwin.alert_image.width(), self.mainwin.alert_image.height(), Qt.KeepAspectRatio)
                            self.mainwin.alert_image.setPixmap(scaled)
                        else:
                            self.mainwin.alert_image.setText('No image')
                    except Exception:
                        self.mainwin.alert_image.setText('Error loading')
                else:
                    self.mainwin.alert_image.setText('No evidence')

        self.gui_alert_observer = GUIAlertObserver(self)
        self.subject.attach(self.gui_alert_observer)

        # Detectar cámaras disponibles
        available_cameras = CameraWidget.detect_available_cameras(prefer_external=True)
        print(f"Cámaras disponibles detectadas: {available_cameras}")

        # Inicializar ranking_counter
        db_cameras = list_cameras()
        self.ranking_counter = {name: 0 for _, name, _ in db_cameras} if db_cameras else {}

        # Si no hay cámaras en la DB, registrar las por defecto
        if not db_cameras:
            default_camera_names = ["cámara 1", "cámara 2", "cámara 3", "cámara 4"]
            
            for i, name in enumerate(default_camera_names):
                camera_index = available_cameras[i] if i < len(available_cameras) else None
                register_camera(name, camera_index)
                self.ranking_counter[name] = 0

        # Recargar cámaras desde DB
        cameras = list_cameras()
        
        self.ranking_updater = RankingUpdater(self.ranking_counter)
        self.subject.attach(self.ranking_updater)

        # current authenticated user (id, username, role) or None
        self.current_user = None

        self.layout = QGridLayout()
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.camera_widgets = []
        
        # Asignar cámaras disponibles
        for i, (pos, (cam_id, name, index)) in enumerate(zip(positions, cameras[:4])):
            camera_index = available_cameras[i] if i < len(available_cameras) else None
            print(f"Asignando {name} con índice {camera_index}")
            
            cam_widget = CameraWidget(name, camera_index, self.subject, 
                                    log_widget=self.alert_text, ranking_counter=self.ranking_counter)
            self.camera_widgets.append(cam_widget)
            self.layout.addWidget(cam_widget, *pos)

        # Panel de ranking y botones
        self.ranking_label = QLabel()
        self.ranking_label.setStyleSheet("color: orange; font-size: 12pt; font-weight: bold;")
        self.ranking_label.setMinimumHeight(100)

        # Botones para CU
        button_style = """
            QPushButton {
                background-color: #555;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: 2px solid #777;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """
        
        login_btn = QPushButton("🔐 Autenticar Usuario (CU-003)")
        login_btn.setStyleSheet(button_style)
        login_btn.clicked.connect(self.authenticate_dialog)
        
        user_btn = QPushButton("👥 Gestión de Usuarios")
        user_btn.setStyleSheet(button_style)
        user_btn.clicked.connect(self.manage_users)
        # Initially disabled until an authorized user logs in
        user_btn.setEnabled(False)
        
        report_btn = QPushButton("📊 Generar Reporte CSV (CU-022)")
        report_btn.setStyleSheet(button_style)
        report_btn.clicked.connect(self.generate_report_dialog)

        cams_btn = QPushButton("📷 Gestión de Cámaras")
        cams_btn.setStyleSheet(button_style)
        cams_btn.clicked.connect(self.manage_cameras)
        cams_btn.setEnabled(False)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("=== ESTADO DE LAS CÁMARAS ==="))
        right_panel.addWidget(self.status_log)
        right_panel.addWidget(QLabel("--- ALERTAS / EVIDENCIA ---"))
        right_panel.addWidget(self.alert_text)
        right_panel.addWidget(self.alert_image)
        right_panel.addWidget(QLabel("📊 Ranking de Cámaras:"))
        right_panel.addWidget(self.ranking_label)
        right_panel.addWidget(login_btn)
        right_panel.addWidget(user_btn)
        right_panel.addWidget(cams_btn)
        right_panel.addWidget(report_btn)

        # keep reference for enabling/disabling based on role
        self.user_btn = user_btn
        self.cams_btn = cams_btn

        main_layout = QHBoxLayout()
        main_layout.addLayout(self.layout, 3)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

        # Timer para ranking
        self.timer_ranking = QTimer()
        self.timer_ranking.timeout.connect(self.update_ranking)
        self.timer_ranking.start(5000)  # Actualizar cada 5 segundos

        # Mostrar información de cámaras asignadas (se muestra en status_log)
        self.show_camera_assignments()

        # Actualizar ranking inicial
        self.update_ranking()

    def show_camera_assignments(self):
        """Muestra información sobre las cámaras asignadas en el log"""
        # Clear and write the camera assignment summary to the status_log
        try:
            self.status_log.clear()
            self.status_log.append("=== ASIGNACIÓN DE CÁMARAS ===")
            for widget in self.camera_widgets:
                status = "CONECTADA" if widget.cap and widget.cap.isOpened() else "SIN SEÑAL"
                self.status_log.append(f"{widget.title}: Índice {widget.camera_index} - {status}")
            self.status_log.append("=============================")
        except Exception:
            pass

    def authenticate_dialog(self):
        dialog = AuthenticationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            auth_user = getattr(dialog, 'authenticated_user', None)
            if auth_user:
                self.current_user = auth_user
                self.alert_text.append(f"<span style='color: green;'>✅ Usuario {auth_user[1]} autenticado (rol: {auth_user[2]}).</span>")
                # enable user management only for admin or supervisor
                if str(auth_user[2]).lower() in ('admin', 'supervisor'):
                    self.user_btn.setEnabled(True)
                    try:
                        self.cams_btn.setEnabled(True)
                    except Exception:
                        pass
                else:
                    self.user_btn.setEnabled(False)
                    try:
                        self.cams_btn.setEnabled(False)
                    except Exception:
                        pass
            else:
                self.alert_text.append("<span style='color: red;'>⚠️ Error al obtener información de usuario autenticado.</span>")

    def manage_users(self):
        # Guard: only admin/supervisor can open user management
        role = None
        if self.current_user:
            role = str(self.current_user[2]).lower()
        if role in ('admin', 'supervisor'):
            dialog = UserManagementDialog(self, current_user=self.current_user)
            dialog.exec_()
        else:
            QMessageBox.warning(self, 'Permisos', 'No tienes permisos para gestionar usuarios.')

    def manage_cameras(self):
        # Guard: only admin/supervisor can open camera management
        role = None
        if self.current_user:
            role = str(self.current_user[2]).lower()
        if role in ('admin', 'supervisor'):
            dialog = CameraManagementDialog(self, current_user=self.current_user)
            dialog.exec_()
        else:
            QMessageBox.warning(self, 'Permisos', 'No tienes permisos para gestionar cámaras.')

    def generate_report_dialog(self):
        path = generate_report(os.path.join(PROJECT_ROOT, 'report.csv'))
        self.alert_text.append(f"<span style='color: blue;'>📄 Reporte generado: {path}</span>")

    def update_ranking(self):
        if not hasattr(self, 'ranking_counter'):
            return
        
        sorted_ranking = sorted(self.ranking_counter.items(), key=lambda x: x[1], reverse=True)
        ranking_text = "RANKING DE CÁMARAS:<br>"
        for i, (name, count) in enumerate(sorted_ranking, 1):
            color = "#FF4444" if i == 1 else "#FFAA00" if i == 2 else "#44FF44"
            ranking_text += f"<span style='color: {color};'>{i}. {name}: {count} alertas</span><br>"
        self.ranking_label.setText(ranking_text)

    def closeEvent(self, event):
        """Limpieza al cerrar la ventana principal"""
        for widget in self.camera_widgets:
            widget.close()
        event.accept()