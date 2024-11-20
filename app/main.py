# main.py
import uvicorn
import logging
from app.config import settings
from app.logging_config import configure_logging

# Configure logging before running
configure_logging()

# Get a logger
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info(f"Starting application in {settings.ENVIRONMENT} environment")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=9000,
            reload=True if settings.ENVIRONMENT == "development" else False,
            workers=1,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)