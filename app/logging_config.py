# app/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def configure_logging():
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler with rotation
            RotatingFileHandler(
                logs_dir / "app.log", 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            ),
            # Console handler
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure specific loggers
    loggers = [
        'uvicorn',
        'uvicorn.access',
        'uvicorn.error',
        'app',
        'RAGEngine',
        'fastapi'
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

    # Set Pydantic to use WARNING level to reduce noise
    logging.getLogger('pydantic').setLevel(logging.WARNING)