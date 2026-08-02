"""Vercel entrypoint — re-exports the FastAPI app for zero-config detection."""

from src.api.main import app

__all__ = ["app"]
