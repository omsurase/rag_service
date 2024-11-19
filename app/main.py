import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Modified from "main:app" to "app:app"
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        workers=1
    )