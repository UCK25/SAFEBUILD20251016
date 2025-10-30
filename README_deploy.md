SafeBuild Web Detector — Deploy rápido

Este pequeño servidor FastAPI recibe frames desde el navegador, ejecuta la detección (casco/chaleco) con YOLO y devuelve la imagen anotada. Es una alternativa más robusta que WebRTC para presentaciones web.

Archivos añadidos:
- `web_server.py` — servidor FastAPI con endpoint `/detect` y carga del modelo.
- `static/index.html` — página de demo que captura la cámara y envía frames al servidor.
- `requirements_web.txt` — dependencias necesarias.
- `Dockerfile` — para crear una imagen desplegable.

Probar localmente (sin Docker)
1. Crear un entorno virtual y activar

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements_web.txt
python web_server.py
```

2. Abrir en el navegador (mismo equipo): http://localhost:8000/ — pulsa Start y permite la cámara.

Probar con Docker
1. Construir la imagen

```powershell
docker build -t safebuild-web-detector:latest .
```

2. Ejecutar

```powershell
docker run --rm -p 8000:8000 --name safebuild-web safebuild-web-detector:latest
```

3. Abrir http://localhost:8000/ en el equipo que tenga la cámara.

Desplegar en Render (guía rápida)
1. Subir este repositorio a GitHub.
2. En Render, crear un nuevo "Web Service" y elegir despliegue desde el repo. Seleccionar el branch.
3. En "Environment" seleccionar "Docker" y Render construirá la imagen usando el `Dockerfile` que incluimos.
4. Una vez desplegado, acceder a la URL pública (HTTPS) que proporciona Render. La página pedirá acceso a la cámara y funcionará.

Notas
- Si tienes un `best.pt` compatible en `runs/detect/train9/weights/best.pt`, `web_server` intentará cargarlo; si falla, cargará `yolov8n.pt`.
- Latencia: la demo envía un frame cada 600 ms por defecto; puedes ajustar el intervalo en `static/index.html` (setInterval).
- Seguridad: en producción limita CORS, añade autenticación y usa HTTPS.

Simplified Render (no Docker) — más fiable si Docker te falla
1. Subir este repo a GitHub (o usa tu repo existente).
2. En Render, crear un nuevo servicio web y conectar el repo (Create -> Web Service -> Connect a repository).
3. Selecciona "Python" como Environment (no Docker).
4. En Build Command pon:
	pip install -r requirements_web.txt
5. En Start Command pon:
	uvicorn web_server:app --host 0.0.0.0 --port $PORT
6. Habilita Auto Deploy si quieres que cada push actualice el servicio.

Notas:
- Render construirá e instalará las dependencias en sus servidores; no necesitas Docker local.
- Si Render falla instalando `torch`/`ultralytics` en su builder, considera usar la imagen Docker (Render soporta Docker) o usar un modelo ligero en el repo.

Uso rápido local sin venv (PowerShell)
Si no quieres usar .venv y quieres verlo ya en tu equipo, ejecuta el script `run_no_venv.ps1`:

```powershell
.\run_no_venv.ps1
```

Esto instala dependencias en tu user-site y ejecuta el servidor con `uvicorn` (usa `py -3` internamente). Si fallan instalaciones grandes (torch), verás mensajes y te indicaré cómo instalar la rueda correcta.

Si quieres, puedo:
- Ajustar el HTML (colores, textos, logo) para presentación con el cliente.
- Preparar un pequeño script para desplegar automáticamente en Render (crear servicio y variables).
