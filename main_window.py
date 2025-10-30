# main_window.py - ACTUALIZADO
import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QVBoxLayout, QTextEdit, QHBoxLayout, 
                             QLabel, QWidget, QPushButton, QInputDialog, QDialog, QLineEdit,
                             QFormLayout, QDialogButtonBox, QMessageBox, QSizePolicy, QScrollArea,
                             QTableWidget, QTableWidgetItem, QComboBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QFont
from datetime import datetime
import os
from database import init_db, list_cameras, register_camera, authenticate_user, generate_report, register_user, list_users, generate_report_xlsx
from database import delete_user, update_user_role, update_password, get_user_by_id, reset_user_password
from observer import SafetyMonitorSubject, AlertLogger, IncidentRegistrar, RankingUpdater
from camera_widget import CameraWidget
from config import PROJECT_ROOT
from database import list_incidents
from PyQt5.QtGui import QPixmap
from config import CAPTURES_DIR
# Modal positioning offsets (adjust X/Y to move dialogs)
MODAL_OFFSET_X = 660
MODAL_OFFSET_Y = 170


def route_message(widget, msg, level='info', title=None, offset_x=MODAL_OFFSET_X, offset_y=MODAL_OFFSET_Y):
    """Display messages using QMessageBox positioned near the bottom-right of the primary screen.
    Use offset_x/offset_y to adjust the exact placement (user can tune these constants).
    """
    msgBox = QMessageBox(widget)
    msgBox.setText(msg)
    msgBox.setWindowTitle(title or ('Info' if level == 'info' else 'Error'))

    if level == 'info':
        msgBox.setIcon(QMessageBox.Information)
    else:
        msgBox.setIcon(QMessageBox.Warning)

    # Use primary screen available geometry to avoid taskbar area
    screen = QApplication.primaryScreen()
    if screen:
        geom = screen.availableGeometry()
        size = msgBox.sizeHint()
        x = geom.x() + geom.width() - size.width() - offset_x
        y = geom.y() + geom.height() - size.height() - offset_y
        msgBox.move(x, y)
    else:
        # Fallback: center on parent
        msgBox.move(widget.mapToGlobal(QPoint(max(0, widget.width() - msgBox.width()), max(0, widget.height() - msgBox.height()))))

    msgBox.exec_()


def smart_info(widget, title, msg):
    route_message(widget, msg, 'info', title)


def smart_warn(widget, title, msg):
    route_message(widget, msg, 'warning', title)

class AuthenticationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autenticación de Usuario")
        self.setFixedSize(400, 150)
        
        # Position at bottom right
        if parent:
            pos = parent.mapToGlobal(QPoint(parent.width() - self.width() - 660,
                                          parent.height() - self.height() - 170))
            self.move(pos)
        
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
                self.authenticated_user = user
                self.accept()
            else:
                msg = "Credenciales incorrectas!"
                route_message(self, msg, 'warning', 'Error')
        else:
            msg = "Por favor complete todos los campos!"
            route_message(self, msg, 'warning', 'Error')

class UserManagementDialog(QDialog):
    def __init__(self, parent=None, current_user=None, guest_mode=False):
        super().__init__(parent)
        self.setWindowTitle("Gestión de Usuarios")
        self.setMinimumSize(650, 120)
        self.guest_mode = guest_mode
        
        # Position at bottom right
        if parent:
            pos = parent.mapToGlobal(QPoint(parent.width() - self.width() - 537,
                                          parent.height() - self.height() - 310))
            self.move(pos)

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
        self.download_btn = QPushButton('Descargar QR')
        self.download_btn.clicked.connect(self.download_selected_qr)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.download_btn)

        layout.addWidget(QLabel('Usuarios Registrados:'))
        layout.addWidget(self.table)
        layout.addLayout(form_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.load_users()
        self.apply_permissions()
    
    def load_users(self):
        users = list_users()
        selected_row = None
        
        # Filter users based on role permissions
        visible_users = []
        if self.current_user:
            current_role = str(self.current_user[2]).lower()
            current_id = int(self.current_user[0])
            
            if current_role in ('admin', 'supervisor'):
                # Admin and supervisor can see all users
                visible_users = users
            else:
                # Guests can only see their own information
                visible_users = [u for u in users if int(u[0]) == current_id]
        else:
            visible_users = []
            
        self.table.setRowCount(len(visible_users))
        for r, user in enumerate(visible_users):
            self.table.setItem(r, 0, QTableWidgetItem(str(user[0])))
            self.table.setItem(r, 1, QTableWidgetItem(user[1]))
            self.table.setItem(r, 2, QTableWidgetItem(user[2]))
            qr = user[3] or ''
            # QR display logic
            display_qr = '*' * len(qr) if qr else ''
            self.table.setItem(r, 3, QTableWidgetItem(display_qr))
        # If current user present, select their row so they see their data immediately
        try:
            if selected_row is not None:
                self.table.selectRow(selected_row)
                # ensure row widgets populated
                sel_items = self.table.selectedItems()
                if sel_items:
                    self.on_row_selected()
        except Exception:
            pass
    
    def add_user(self):
        username = self.username_input.text().strip()
        role = self.role_input.currentText().strip() or 'supervisor'
        qr = self.qr_input.text().strip() or None
        # For new user we require a password (prompt)
        pwd, ok = QInputDialog.getText(self, 'Password', 'Contraseña para el nuevo usuario:', QLineEdit.Password)
        if not ok or not pwd:
            msg = 'Contraseña requerida para nuevo usuario'
            smart_warn(self, 'Error', msg)
            return
        if register_user(username, pwd, qr):
            # set role
            users = list_users()
            for u in users:
                if u[1] == username:
                    update_user_role(u[0], role)
            msg = 'Usuario agregado'
            smart_info(self, 'OK', msg)
            self.load_users()
        else:
            msg = 'No se pudo agregar el usuario (dup?)'
            smart_warn(self, 'Error', msg)

    def on_row_selected(self):
        sel = self.table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        item0 = self.table.item(row, 0)
        item1 = self.table.item(row, 1)
        item2 = self.table.item(row, 2)
        item3 = self.table.item(row, 3)
        if not item0 or not item1 or not item2 or not item3:
            return
        uid = item0.text()
        self.id_label.setText(uid)
        self.username_input.setText(item1.text())
        # set combo to the current role if present
        role_text = item2.text()
        idx = self.role_input.findText(role_text)
        if idx >= 0:
            self.role_input.setCurrentIndex(idx)
        else:
            # if role not present, add it temporarily and select
            self.role_input.addItem(role_text)
            self.role_input.setCurrentIndex(self.role_input.count()-1)
        self.qr_input.setText(item3.text())
        # Enable download button - permissions will be checked when clicked
        self.download_btn.setEnabled(True)

    def update_user(self):
        if not self.current_user:
            smart_warn(self, 'Error', 'Debe autenticarse primero')
            return
            
        uid = self.id_label.text()
        if not uid:
            smart_warn(self, 'Seleccionar', 'Seleccione la fila del usuario en la lista.')
            return
            
        # Check permissions
        user = get_user_by_id(int(uid))
        if not user:
            smart_warn(self, 'Error', 'Usuario no encontrado')
            return
            
        current_role = str(self.current_user[2]).lower()
        target_role = str(user[2]).lower()
        current_id = int(self.current_user[0])
        target_id = int(uid)
        
        username = self.username_input.text().strip()
        new_role = self.role_input.currentText().strip()
        qr = self.qr_input.text().strip() or None

        # Permission checks for role changes
        if new_role != target_role:  # Role is being changed
            if current_id == target_id:
                smart_warn(self, 'Error', 'No puede cambiar su propio rol')
                return
            elif current_role == 'admin':
                if target_role == 'admin':
                    smart_warn(self, 'Error', 'Los administradores no pueden cambiar el rol de otros administradores')
                    return
            elif current_role == 'supervisor':
                smart_warn(self, 'Error', 'Los supervisores no pueden cambiar roles')
                return
            else:
                smart_warn(self, 'Error', 'No tiene permisos para cambiar roles')
                return

        # General update permissions
        if current_role == 'admin':
            if target_role == 'admin' and current_id != target_id:
                smart_warn(self, 'Error', 'Los administradores no pueden modificar otros administradores')
                return
        elif current_role == 'supervisor':
            if target_role in ('admin', 'supervisor') and current_id != target_id:
                smart_warn(self, 'Error', 'Los supervisores solo pueden modificar usuarios visitantes')
                return
        else:
            smart_warn(self, 'Error', 'No tiene permisos para modificar usuarios')
            return
            
        from database import update_user
        ok = update_user(int(uid), username=username, qr_code=qr)
        if ok:
            update_user_role(int(uid), new_role)
            msg = 'Usuario actualizado'
            smart_info(self, 'OK', msg)
            self.load_users()
        else:
            msg = 'No se pudo actualizar (dup?)'
            smart_warn(self, 'Error', msg)

    def delete_user(self):
        if not self.current_user:
            smart_warn(self, 'Error', 'Debe autenticarse primero')
            return
            
        uid = self.id_label.text()
        if not uid:
            smart_warn(self, 'Seleccionar', 'Seleccione la fila del usuario en la lista.')
            return
            
        # Prevent self-deletion
        try:
            if int(uid) == int(self.current_user[0]):
                smart_warn(self, 'Error', 'No puede eliminar su propio usuario')
                return
        except (TypeError, ValueError):
            smart_warn(self, 'Error', 'ID de usuario inválido')
            return
            
        # Check permissions
        user = get_user_by_id(int(uid))
        if not user:
            smart_warn(self, 'Error', 'Usuario no encontrado')
            return
            
        current_role = str(self.current_user[2]).lower()
        target_role = str(user[2]).lower()
        
        if current_role == 'admin':
            if target_role == 'admin':
                smart_warn(self, 'Error', 'Los administradores no pueden eliminar otros administradores')
                return
        elif current_role == 'supervisor':
            if target_role in ('admin', 'supervisor'):
                smart_warn(self, 'Error', 'Los supervisores solo pueden eliminar usuarios visitantes')
                return
                
        from database import delete_user
        if delete_user(int(uid)):
            msg = 'Usuario eliminado'
            smart_info(self, 'OK', msg)
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
            msg = 'No se pudo eliminar'
            smart_warn(self, 'Error', msg)

    def apply_permissions(self):
        if not self.current_user:
            # No user authenticated: disable all controls but do not show modal warnings
            self.delete_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.add_btn.setEnabled(False)
            self.update_btn.setEnabled(False)
            self.username_input.setEnabled(False)
            self.role_input.setEnabled(False)
            self.qr_input.setEnabled(False)
            return

        current_role = str(self.current_user[2]).lower()
        
        # Non-admin users: show only their own information
        if current_role != 'admin':
            self.add_btn.hide()
            if current_role == 'guest':
                # Guest mode: minimal view
                self.delete_btn.hide()
                self.reset_btn.hide()
                self.update_btn.hide()
                self.username_input.hide()
                self.role_input.hide()
                self.qr_input.hide()
            return

        # For admin and supervisor, enable based on selected user
        selected_id = self.get_selected_user_id()
        if selected_id:
            selected_user = get_user_by_id(selected_id)
            if selected_user:
                selected_role = str(selected_user[2]).lower()
                
                if current_role == 'admin':
                    # Admin can't modify other admins except themselves
                    if selected_role == 'admin' and int(selected_id) != int(self.current_user[0]):
                        self.delete_btn.setEnabled(False)
                        self.reset_btn.setEnabled(False)
                        self.update_btn.setEnabled(False)
                        self.username_input.setEnabled(False)
                        self.role_input.setEnabled(False)
                        self.qr_input.setEnabled(False)
                    else:
                        # Admin can modify supervisors and guests
                        self.delete_btn.setEnabled(True)
                        self.reset_btn.setEnabled(True)
                        self.update_btn.setEnabled(True)
                        self.username_input.setEnabled(True)
                        self.role_input.setEnabled(True)
                        self.qr_input.setEnabled(True)
                    # Admin can always add new users
                    self.add_btn.setEnabled(True)
                    
                elif current_role == 'supervisor':
                    # Supervisor can't modify admins or other supervisors except themselves
                    if (selected_role == 'admin' or 
                        (selected_role == 'supervisor' and int(selected_id) != int(self.current_user[0]))):
                        self.delete_btn.setEnabled(False)
                        self.reset_btn.setEnabled(False)
                        self.update_btn.setEnabled(False)
                        self.username_input.setEnabled(False)
                        self.role_input.setEnabled(False)
                        self.qr_input.setEnabled(False)
                    else:
                        # Supervisor can modify guests and themselves
                        self.delete_btn.setEnabled(True)
                        self.reset_btn.setEnabled(True)
                        self.update_btn.setEnabled(True)
                        self.username_input.setEnabled(True)
                        self.role_input.setEnabled(True)
                        self.qr_input.setEnabled(True)
                    # Supervisor can add new users
                    self.add_btn.setEnabled(True)
        else:
            # No user selected - only enable add button for admin/supervisor
            self.delete_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.update_btn.setEnabled(False)
            self.username_input.setEnabled(False)
            self.role_input.setEnabled(False)
            self.qr_input.setEnabled(False)
            self.add_btn.setEnabled(True)

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
            # do not show a warning here (caller will show if needed)
            return None
        row = sel[0].row()
        item = self.table.item(row, 0)
        if not item:
            smart_warn(self, 'Error', 'No se pudo determinar el ID seleccionado.')
            return None
        try:
            return int(item.text())
        except Exception:
            smart_warn(self, 'Error', 'No se pudo determinar el ID seleccionado.')
            return None

    def reset_selected_password(self):
        if not self.current_user:
            smart_warn(self, 'Error', 'Debe autenticarse primero')
            return

        uid = self.get_selected_user_id()
        if uid is None:
            smart_warn(self, 'Seleccionar', 'Seleccione la fila del usuario en la lista.')
            return

        # Get target user info
        user = get_user_by_id(int(uid))
        if not user:
            smart_warn(self, 'Error', 'Usuario no encontrado')
            return

        current_role = str(self.current_user[2]).lower()
        target_role = str(user[2]).lower()
        current_id = int(self.current_user[0])
        target_id = int(uid)

        # Check permissions
        if current_role == 'admin':
            if target_role == 'admin' and current_id != target_id:
                smart_warn(self, 'Error', 'Los administradores no pueden cambiar la contraseña de otros administradores')
                return
        elif current_role == 'supervisor':
            if target_role in ('admin', 'supervisor'):
                smart_warn(self, 'Error', 'Los supervisores solo pueden cambiar la contraseña de visitantes')
                return
        else:
            if current_id != target_id:
                smart_warn(self, 'Error', 'Los visitantes solo pueden cambiar su propia contraseña')
                return

        # Ask for new password
        new_pass, ok = QInputDialog.getText(self, 'Reset Password', 'Nueva contraseña:', QLineEdit.Password)
        if ok and new_pass:
            from database import reset_user_password
            actor = self.current_user[1] if self.current_user else None
            if reset_user_password(actor, uid, new_pass):
                smart_info(self, 'OK', 'Contraseña actualizada y registrada en auditoría')
                self.load_users()
            else:
                smart_warn(self, 'Error', 'No se pudo actualizar la contraseña')

    def download_selected_qr(self):
        if not self.current_user:
            smart_warn(self, 'Error', 'Debe autenticarse primero')
            return
            
        uid = self.get_selected_user_id()
        if uid is None:
            smart_warn(self, 'Seleccionar', 'Seleccione la fila del usuario en la lista.')
            return
            
        user = get_user_by_id(uid)
        if not user:
            smart_warn(self, 'Error', 'Usuario no encontrado')
            return
            
        qr = user[3]
        if not qr:
            smart_info(self, 'Sin QR', 'El usuario no tiene QR asignado')
            return
            
        # Check permissions
        try:
            current_role = str(self.current_user[2]).lower()
            target_role = str(user[2]).lower()
            current_id = int(self.current_user[0])
            target_id = int(uid)
            
            # Everyone can download their own QR
            if current_id == target_id:
                allowed = True
            # Additional permissions based on role
            elif current_role == 'admin':
                if target_role != 'admin':
                    allowed = True
                else:
                    smart_warn(self, 'Error', 'Los administradores no pueden descargar QR de otros administradores')
                    return
            elif current_role == 'supervisor':
                if target_role == 'guest':
                    allowed = True
                else:
                    smart_warn(self, 'Error', 'Los supervisores solo pueden descargar QR de visitantes')
                    return
            else:  # guest
                smart_warn(self, 'Error', 'Los visitantes solo pueden descargar su propio QR')
                return
                
        except Exception:
            smart_warn(self, 'Error', 'Error al verificar permisos')
            return

        # Create QR directory if it doesn't exist
        qr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'QR')
        if not os.path.exists(qr_dir):
            os.makedirs(qr_dir)

        # Set default save path to QR directory
        default_path = os.path.join(qr_dir, f'{user[1]}_qr.png')
        fname, _ = QFileDialog.getSaveFileName(self, 'Guardar QR como', default_path, 'PNG Image (*.png)')
        if not fname:
            return

        # Generate QR image
        try:
            import qrcode
            img = qrcode.make(qr)
            with open(fname, 'wb') as f:
                img.save(f)
            smart_info(self, 'OK', f'QR guardado en {fname}')
        except Exception as e:
            smart_warn(self, 'Error', f'No se pudo guardar el QR: {e}')

    def change_selected_role(self):
        uid = self.get_selected_user_id()
        if uid is None:
            smart_warn(self, 'Seleccionar', 'Seleccione la fila del usuario en la lista.')
            return
        # Ask for role (admin/supervisor/guest)
        roles = ['admin', 'supervisor', 'guest']
        role, ok = QInputDialog.getItem(self, 'Cambiar Rol', 'Seleccionar rol:', roles, 0, False)
        if ok and role:
            from database import update_user_role
            if update_user_role(uid, role):
                smart_info(self, 'OK', 'Rol actualizado')
                self.load_users()
            else:
                smart_warn(self, 'Error', 'No se pudo actualizar el rol')


class CameraManagementDialog(QDialog):
    def __init__(self, parent=None, current_user=None):
        super().__init__(parent)
        self.setWindowTitle('Gestión de Cámaras')
        self.setMinimumSize(650, 120)
        self.current_user = current_user

        # Position at bottom right
        if parent:
            pos = parent.mapToGlobal(QPoint(parent.width() - self.width() - 537,
                                          parent.height() - self.height() - 300))
            self.move(pos)

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
            smart_info(self, 'OK', 'Cámara agregada')
            self.load_cameras()
        except Exception as e:
            smart_warn(self, 'Error', f'No se pudo agregar la cámara: {e}')

    def get_selected_camera_id(self):
        sel = self.table.selectedItems()
        if not sel:
            smart_warn(self, 'Seleccionar', 'Seleccione la fila de la cámara en la lista.')
            return None
        row = sel[0].row()
        item = self.table.item(row, 0)
        if not item:
            smart_warn(self, 'Error', 'No se pudo determinar el ID de la cámara seleccionada.')
            return None
        try:
            return int(item.text())
        except Exception:
            smart_warn(self, 'Error', 'No se pudo determinar el ID de la cámara seleccionada.')
            return None

    def on_row_selected(self):
        sel = self.table.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        item0 = self.table.item(row, 0)
        item1 = self.table.item(row, 1)
        item2 = self.table.item(row, 2)
        if not item0 or not item1 or not item2:
            return
        self.cam_id_label.setText(item0.text())
        self.cam_name.setText(item1.text())
        self.cam_index.setText(item2.text())

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
            smart_info(self, 'OK', 'Cámara eliminada')
            self.load_cameras()
        except Exception as e:
            smart_warn(self, 'Error', f'No se pudo eliminar la cámara: {e}')

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
            smart_info(self, 'OK', 'Cámara actualizada')
            self.load_cameras()
        except Exception as e:
            smart_warn(self, 'Error', f'No se pudo actualizar la cámara: {e}')

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

        # Panel de información (ocupa la posición donde estaba cámara 4)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("background-color: #111; color: #88FF88; font-family: Consolas; font-size: 10pt;")

        # We keep alert_image as an internal widget but we won't present evidence images on the right panel per requirements
        self.alert_image = QLabel()
        self.alert_image.setFixedSize(320, 240)
        self.alert_image.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.alert_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Subject y Observers (Patrón Observer)
        # Subject y Observers (Patrón Observer)
        self.subject = SafetyMonitorSubject()
        # Logger que escribe en el panel de alertas (PPE alerts only)
        self.alert_logger = AlertLogger(self.alert_text)
        self.incident_registrar = IncidentRegistrar()
        self.subject.attach(self.alert_logger)
        self.subject.attach(self.incident_registrar)

        # PPE alerts are routed via AlertLogger into `alert_text`; no extra GUI observer attached to avoid duplicates.

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

        self.grid_layout = QGridLayout()
        # Show 2 cameras by default, and 3rd camera only if 3+ cameras are available
        positions = [(0, 0), (0, 1), (1, 0)]  # Positions for up to 3 cameras
        self.camera_widgets = []
        
        # Default to showing only 2 cameras
        num_cameras_to_show = min(2, len(cameras))
        
        # Show 3rd camera only if 3 or more cameras are detected
        if len(available_cameras) >= 3:
            num_cameras_to_show = min(3, len(cameras))
            
        # Assign available cameras to the grid
        for i, (pos, cam) in enumerate(zip(positions[:num_cameras_to_show], cameras[:num_cameras_to_show])):
            cam_id, name, index = cam
            camera_index = available_cameras[i] if i < len(available_cameras) else None
            print(f"Asignando {name} con índice {camera_index}")
            # Pasar el widget de alertas como log_widget para asegurar que
            # tanto ALERTAS como INFORMACIONES (QR) se muestren en el panel derecho
            cam_widget = CameraWidget(name, camera_index, self.subject,
                                      log_widget=self.alert_text, ranking_counter=self.ranking_counter)
            self.camera_widgets.append(cam_widget)
            self.grid_layout.addWidget(cam_widget, *pos)

        # Crear contenedor para la zona donde estuvo cámara 4 (info)
        self.info_container = QWidget()
        self.info_container_layout = QVBoxLayout()
        self.info_container.setLayout(self.info_container_layout)
        # info_text se mantiene arriba y debajo se mostrarán los paneles embebidos
        self.info_container_layout.addWidget(self.info_text)
        # panel_area (vacío) — reservado para mensajes o futuras extensiones
        self.panel_area = QWidget()
        self.panel_area_layout = QVBoxLayout()
        self.panel_area.setLayout(self.panel_area_layout)
        self.info_container_layout.addWidget(self.panel_area)

        # Reservar la celda (1,1) para el contenedor de información
        self.grid_layout.addWidget(self.info_container, 1, 1)

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

        login_btn = QPushButton("🔐 Autenticar Usuario")
        login_btn.setStyleSheet(button_style)
        login_btn.clicked.connect(self.authenticate_dialog)

        user_btn = QPushButton("👥 Gestión de Usuarios")
        user_btn.setStyleSheet(button_style)
        user_btn.clicked.connect(self.manage_users)
        # Initially disabled until an authorized user logs in
        user_btn.setEnabled(False)

        self.report_btn = QPushButton("📊 Generar Reporte")
        self.report_btn.setStyleSheet(button_style)
        self.report_btn.clicked.connect(self.generate_report_dialog)

        cams_btn = QPushButton("📷 Gestión de Cámaras")
        cams_btn.setStyleSheet(button_style)
        cams_btn.clicked.connect(self.manage_cameras)
        cams_btn.setEnabled(False)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("=== ESTADO DE LAS CÁMARAS ==="))
        right_panel.addWidget(self.status_log)
        # Mostrar cuadro de avisos / alert_text como antes, debajo del estado de cámaras
        right_panel.addWidget(QLabel("--- ALERTAS / INFORMACIONES ---"))
        right_panel.addWidget(self.alert_text)
        right_panel.addWidget(self.ranking_label)

        # keep reference for enabling/disabling based on role
        self.user_btn = user_btn
        self.cams_btn = cams_btn

        # Colocar los botones en la celda (1,0) — donde antes estaba la cámara 3
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout()
        buttons_widget.setLayout(buttons_layout)
        buttons_layout.addWidget(login_btn)
        buttons_layout.addWidget(user_btn)
        buttons_layout.addWidget(cams_btn)
        # Reporte deshabilitado por defecto hasta autenticación
        self.report_btn.setEnabled(False)
        buttons_layout.addWidget(self.report_btn)
        buttons_layout.addStretch()
        # Añadir widget de botones a la cuadrícula en (1,0)
        self.grid_layout.addWidget(buttons_widget, 1, 0)

        # Estructura principal: grid (3/4) y panel derecho (1/4)
        main_layout = QHBoxLayout()
        main_layout.addLayout(self.grid_layout, 3)
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
            # Always show first 2 cameras
            for widget in self.camera_widgets[:2]:
                status = "CONECTADA" if widget.cap and widget.cap.isOpened() else "SIN SEÑAL"
                self.status_log.append(f"{widget.title}: Índice {widget.camera_index} - {status}")
            # Show 3rd camera only if 3 or more cameras are available
            if len(CameraWidget.detect_available_cameras(prefer_external=True)) >= 3 and len(self.camera_widgets) >= 3:
                widget = self.camera_widgets[2]
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
                # Show popup and add to alerts
                msg = f"Usuario {auth_user[1]} autenticado (rol: {auth_user[2]})."
                route_message(self, msg, 'info', 'Autenticación Exitosa')
                self.alert_text.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Autenticación exitosa: Usuario {auth_user[1]} (rol: {auth_user[2]})")
                # enable user management for any authenticated user (guests see limited view)
                self.user_btn.setEnabled(True)
                # enable camera management only for admin or supervisor
                try:
                    if str(auth_user[2]).lower() in ('admin', 'supervisor'):
                        self.cams_btn.setEnabled(True)
                    else:
                        self.cams_btn.setEnabled(False)
                except Exception:
                    pass
                # enable report button for authenticated users
                try:
                    self.report_btn.setEnabled(True)
                except Exception:
                    pass
            else:
                msg = "Error al obtener información de usuario autenticado."
                route_message(self, msg, 'warning', 'Error')
        else:
            msg = "Autenticación cancelada."
            route_message(self, msg, 'info', 'Autenticación')

    def manage_users(self):
        # Guard: only admin/supervisor can open user management
        role = None
        if self.current_user:
            role = str(self.current_user[2]).lower()
            if role in ('admin', 'supervisor'):
                dialog = UserManagementDialog(self, current_user=self.current_user)
                dialog.exec_()
            elif role == 'guest':
                # Guests can see their QR code in a simplified dialog
                dialog = UserManagementDialog(self, current_user=self.current_user, guest_mode=True)
                dialog.exec_()
            else:
                msg = "No tienes permisos para gestionar usuarios."
                route_message(self, msg, 'warning', 'Error de Permisos')
        else:
            msg = "Debe autenticarse primero."
            route_message(self, msg, 'warning', 'Error de Permisos')

    def manage_cameras(self):
        # Guard: only admin/supervisor can open camera management
        role = None
        if self.current_user:
            role = str(self.current_user[2]).lower()
        if role in ('admin', 'supervisor'):
            dialog = CameraManagementDialog(self, current_user=self.current_user)
            dialog.exec_()
        else:
            msg = "No tienes permisos para gestionar cámaras."
            route_message(self, msg, 'warning', 'Error de Permisos')

    def generate_report_dialog(self):
        # Solo permitir si el usuario está autenticado
        if not self.current_user:
            route_message(self, 'Debe autenticarse antes de generar reportes.', 'warning', 'Autenticación requerida')
            return
        # Pedir año y mes al usuario (mes 0 = todo el año)
        from PyQt5.QtWidgets import QInputDialog
        from datetime import datetime
        now = datetime.now()
        year, ok = QInputDialog.getInt(self, 'Generar Reporte', 'Año (ej. 2025):', now.year, 2000, 2100, 1)
        if not ok:
            return
        month_items = ['Todo el año', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        month_idx, okm = QInputDialog.getItem(self, 'Generar Reporte', 'Mes (opcional):', month_items, now.month, False)
        if not okm:
            return
        month = 0
        try:
            month = month_items.index(month_idx)
        except Exception:
            month = 0

        # Construir nombre de archivo
        if month == 0:
            out_name = f'reporte_{year}.xlsx'
        else:
            out_name = f'reporte_{year}_{month:02d}.xlsx'
        out_path = os.path.join(PROJECT_ROOT, out_name)

        try:
            # Llamar a la función que genera Excel con análisis y evidencias
            generate_report_xlsx(year=year, month=(None if month == 0 else month), output_path=out_path, time_window_minutes=60)
            route_message(self, f'Reporte generado: {out_path}', 'info', 'Reporte generado')
        except Exception as e:
            route_message(self, f'Error generando reporte: {e}', 'warning', 'Error')

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