from fastapi import FastAPI
from app.api.routes import router
from app.config import settings

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title=settings.APP_NAME,
        description="RAG Service API for document processing and embeddings",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    # Include router with prefix
    app.include_router(
        router,
        prefix=settings.API_V1_STR
    )

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "api_version": settings.API_V1_STR
        }

    return app

# Create the application instance
app = create_app()