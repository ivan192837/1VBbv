"""
Локальный сервер для страницы sklearn docs + прокси для AI-чата через OpenRouter API (Qwen).
Запуск: python server.py
Открыть в браузере: http://localhost:8080
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import traceback
import urllib.request
import ssl

# === НАСТРОЙКИ ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "qwen/qwen3.6-plus:free"  # поддерживает text + image + video
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# =================

# SSL контекст
ssl_ctx = ssl.create_default_context()


def call_openrouter(messages, images=None):
    """Отправить запрос к OpenRouter API"""

    # Если есть изображения — преобразуем последнее user-сообщение в multimodal формат
    if images:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = msg.get("content", "Что изображено на этих скриншотах?")
                new_content = [{"type": "text", "text": text}]
                for img_b64 in images:
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    })
                msg["content"] = new_content
                break

    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Sklearn Docs Assistant"
        },
        method="POST"
    )

    with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
        return json.loads(resp.read())


class ProxyHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            user_data = json.loads(body)
            messages = user_data.get("messages", [])
            images = user_data.get("images", [])
            
            # Поддержка старого формата, если вдруг отправили одно
            if "image" in user_data and user_data["image"]:
                images.append(user_data["image"])

            try:
                result = call_openrouter(messages, images)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"❌ Ошибка API: {error_msg}")
                traceback.print_exc()

                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"message": error_msg}
                }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.path = "/sklearn_docs_deepseek.html"
        super().do_GET()

    def log_message(self, format, *args):
        if "favicon" not in str(args) and "apple-touch" not in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), ProxyHandler)
    print(f"✅ Сервер запущен: http://localhost:{port}")
    print(f"   Модель: {OPENROUTER_MODEL}")
    print("   Нажмите Ctrl+C чтобы остановить")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
        server.server_close()
