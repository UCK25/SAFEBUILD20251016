import os

# Configuraciones globales
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Detecta dinámicamente
DATASET_PATH = os.path.join(PROJECT_ROOT, 'Safety-vest---v4-1')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train8', 'weights', 'best.pt')
DB_PATH = os.path.join(PROJECT_ROOT, 'safety_monitor.db')
CAPTURES_DIR = os.path.join(PROJECT_ROOT, 'captures')

# Umbrales por defecto (ajustables por CU-025)
DEFAULT_CONF = 0.4
ALERT_COOLDOWN = 10  # Segundos
QR_COOLDOWN = 5  # Segundos entre detecciones del mismo QR