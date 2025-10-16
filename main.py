import sys
import os
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from database import init_db
from config import PROJECT_ROOT

if __name__ == "__main__":
    # Verificar dataset y modelo
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'Safety-vest---v4-1')):
        print("Advertencia: Dataset no encontrado. Ejecuta descarga Roboflow si es necesario.")
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train8', 'weights', 'best.pt')):
        print("Advertencia: Modelo 'best.pt' no encontrado. El sistema usará fallback.")

    init_db()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())