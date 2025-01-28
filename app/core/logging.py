from loguru import logger
import sys
from pathlib import Path

def setup_logging():
    # Remove default logger
    logger.remove()
    
    # Log format for console
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Console logging
    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG",
        colorize=True,
    )
    
    # File logging
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        format=log_format,
        level="INFO",
        compression="zip",
    )
