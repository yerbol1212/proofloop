"""Local-only form for entering a Meta API key without putting it in chat."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import os
import secrets
import tempfile
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
TOKEN = secrets.token_urlsafe(32)
FORM = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ProofLoop — Meta key setup</title><style>body{font:18px system-ui;background:#f5f7fa;color:#18222f;max-width:600px;margin:12vh auto;padding:28px}main{background:white;border:1px solid #dde3eb;border-radius:18px;padding:32px}h1{font-size:28px}label{display:block;margin:26px 0 10px;font-weight:600}input{box-sizing:border-box;width:100%;font:18px system-ui;padding:14px;border:1px solid #9aa9bd;border-radius:8px}button{margin-top:18px;background:#165dca;color:white;border:0;border-radius:8px;padding:14px 24px;font:600 17px system-ui;cursor:pointer}p{line-height:1.5}small{color:#536176}</style><main><h1>Connect ProofLoop to Meta</h1><p>Paste your Meta API key below, then click <b>Save key</b>.</p><form method="post" action="/save"><input type="hidden" name="token" value="TOKEN"><label for="key">Meta API key</label><input id="key" name="key" type="password" autocomplete="off" spellcheck="false" required autofocus placeholder="Paste your Meta API key here"><button type="submit">Save key</button></form><p><small>This local page saves the key into ProofLoop’s private .env.local file. The file is excluded from GitHub.</small></p></main></html>'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def send(self, status, body):
        data = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)
    def valid_host(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'
    def do_GET(self):
        if not self.valid_host() or self.path != '/':
            return self.send(404, 'Not found')
        self.send(200, FORM.replace('TOKEN', TOKEN))
    def do_POST(self):
        expected = f'http://127.0.0.1:{self.server.server_port}'
        if not self.valid_host() or self.path != '/save' or self.headers.get('Origin') != expected:
            return self.send(403, 'Please use the local setup page.')
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError()
            data = parse_qs(self.rfile.read(size).decode(), max_num_fields=4)
            if not secrets.compare_digest(data.get('token', [''])[0], TOKEN):
                raise ValueError()
            key = data.get('key', [''])[0].strip()
            if not 8 <= len(key) <= 4096 or any(c.isspace() for c in key) or any(c in key for c in '\"\'\\\x00'):
                raise ValueError()
            dest = ROOT / '.env.local'
            lines = dest.read_text().splitlines() if dest.exists() else []
            lines = [line for line in lines if line.strip().partition('=')[0].strip() != 'META_API_KEY']
            lines.append('META_API_KEY=' + key)
            fd, name = tempfile.mkstemp(prefix='.env-key-', dir=ROOT)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, 'w') as output:
                    output.write('\n'.join(lines) + '\n')
                os.replace(name, dest)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            self.send(200, '<!doctype html><html><meta charset="utf-8"><title>Key saved</title><body style="font:20px system-ui;max-width:600px;margin:15vh auto;padding:24px"><h1>Meta key saved</h1><p>You can close this page and tell Codex <b>saved</b>. Your key was not displayed or sent to GitHub.</p></body></html>')
        except (ValueError, UnicodeError, OSError):
            self.send(400, 'Could not save. Return to the local setup page and paste the key again.')

def main():
    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        print(f'ProofLoop key setup: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
