"""Invoca la API HTTP ResUNet desde una máquina local.
Uso: python invoke_api.py https://.../api/v1/segment test.jpg prediction.png
Requiere: pip install requests
"""
import sys
from pathlib import Path
import requests

if len(sys.argv) != 4:
    raise SystemExit('Uso: python invoke_api.py <api_url> <imagen.jpg> <salida.png>')
url, input_path, output_path = sys.argv[1:]
response = requests.post(url, data=Path(input_path).read_bytes(),
                         headers={'Content-Type': 'image/jpeg', 'Accept': 'image/png'},
                         timeout=30)
response.raise_for_status()
if not response.content.startswith(b'\x89PNG\r\n\x1a\n'):
    raise ValueError('La respuesta no es PNG')
Path(output_path).write_bytes(response.content)
print(f'HTTP {response.status_code} | {len(response.content)} bytes | {output_path}')
