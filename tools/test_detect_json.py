# Simple test: generate a QR PNG and POST it to local /detect_json
import io, sys
try:
    import qrcode
    import requests
except Exception as e:
    print('Missing dependency:', e)
    print('Install with: pip install qrcode requests')
    sys.exit(2)

qr_val = 'TEST-QR-12345'
img = qrcode.make(qr_val).convert('RGB')
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
files = {'frame': ('test_qr.png', buf, 'image/png')}
try:
    r = requests.post('http://127.0.0.1:8000/detect_json', files=files, timeout=10)
    print('HTTP', r.status_code)
    try:
        j = r.json()
        import json
        print(json.dumps(j, indent=2, ensure_ascii=False))
    except Exception:
        print('Response text length:', len(r.text))
        print(r.text[:1000])
except Exception as e:
    print('Request error:', e)
    sys.exit(3)
