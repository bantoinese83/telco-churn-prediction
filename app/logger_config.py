import os
from loguru import logger


def configure_logger(
    log_file_path="logs/app_log_{time}.log",
    rotation_size="25 MB",
    log_format="{time} - {level} - {message}",
):
    """
    Configure logger with file rotation and custom format.

    Parameters:
    - log_file_path (str): Path pattern for log files.
    - rotation_size (str): Maximum size of each log file before rotation.
    - format (str): Log message format.

    Returns:
    - logger: Configured logger instance.
    """
    try:
        # Ensure the logs directory exists
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)

        # Add a rotating file handler
        logger.add(log_file_path, rotation=rotation_size, format=log_format)

        return logger
    except Exception as e:
        print(f"Error configuring logger: {e}")
        return None
