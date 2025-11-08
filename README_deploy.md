SafeBuild — deploy rápido (Flask)

Este repositorio contiene tanto una versión Streamlit (`app.py`) como dos servidores web: `web_server.py` (FastAPI/uvicorn) y `flask_server.py` (Flask). Para simplificar el despliegue en Render como "pure Flask" mantendremos `flask_server.py` como entrypoint y dejaremos los otros servidores/artefactos opcionales fuera del branch de despliegue.

Archivos relevantes para deploy Flask:
- `flask_server.py` — servidor Flask con endpoints de dashboard, detección (API) y reportes.
- `static/` — HTML/CSS/JS del dashboard.
- `captures/`, `safety_monitor.db` — datos y DB (creados/actualizados en runtime).
- `yolov8n.pt` — modelo ligero utilizado como fallback (recomendado mantener si se quiere detección).

Pasos para desplegar en Render (sin Docker)
1. Subir este repo a GitHub.
2. En Render, crear un nuevo Web Service y conectar el repo.
3. Seleccionar "Python" como environment.
4. Build Command: `pip install -r requirements.txt` (he añadido `requirements.txt` para Flask).
5. Start Command: `gunicorn flask_server:app --bind 0.0.0.0:$PORT` (o usar el `Procfile` existente).
6. Habilitar Auto Deploy si quieres despliegues automáticos por push.

Notas importantes
- Dependencias pesadas: `ultralytics` arrastra `torch` (binarios grandes). En Render's free builder la instalación de `torch` puede fallar por falta de compatibilidad/recursos. Opciones:
  - Usar Docker con una imagen base que incluya la rueda de `torch` apropiada (más fiable). (El repo incluye un `Dockerfile` que actualmente usa `requirements_web.txt` — si quieres usar Docker puedo adaptarlo para Flask.)
  - Omitir la carga del modelo en Render (desactivar detección) y usar sólo el dashboard/registro/reportes. Puedo añadir una variable de entorno `LOAD_MODEL=no` para saltar la carga de `ultralytics` en esa modalidad.

 - Recomendación de seguridad: en producción asegúrate de usar HTTPS (Render lo provee por defecto) y revisar permisos de los endpoints.

Si quieres que yo continúe, puedo:
- 作る: ストリップされたデプロイ用ブランチ（例: `deploy/flask`）を作成して、不要ファイルを除外し、`requirements.txt` と `render.yaml` を更新します。
- Docker 化: `Dockerfile` を Flask 用に更新し、`torch` を含めるオプションを提供します（必要なら）。
- モデルの有無で動作するかを切り替える環境変数を追加します。

どれを優先しますか？（すぐ作業を始めます）
