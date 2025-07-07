import logging
import sys

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Creates and configures a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# For backward compatibility with older scripts
data_collector_logger = get_logger('data_collector') 