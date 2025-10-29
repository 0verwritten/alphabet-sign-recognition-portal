from fastapi import APIRouter

from app.api.routes import asl, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(asl.router)
