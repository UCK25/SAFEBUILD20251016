# SafeBuild AI - Sistema de Monitoreo de Seguridad en Obras

## Instalación
1. Clona o crea la estructura de carpetas.
2. `pip install -r requirements.txt`
3. Descarga dataset: Usa Roboflow si no tienes Safety-vest---v4-1.
4. Entrena modelo: `yolo task=detect mode=train model=yolov8n.pt data=Safety-vest---v4-1/data.yaml epochs=100` (genera best.pt).
5. Ejecuta: `python main.py`

## Mapeo HU/CU
- HU-01 a HU-10: database.py (usuarios/cámaras).
- CU-012 a CU-017: camera_widget.py + observer.py.
- CU-018 a CU-026: main_window.py (listar, reportes, configs).

## Riesgos
- William Fine integrado en observer.py para severidad.
- Pruebas: Simula con videos; precisión >90%.

Referencias: Ver DOCUMENTO P4.docx.