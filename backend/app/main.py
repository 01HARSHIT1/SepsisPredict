"""ASGI entry point. The model is loaded by the application startup hook."""

from backend.app.api.routes import create_app

app = create_app()
