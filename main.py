import argparse
import logging
import signal
import sys
from halo import Halo

from app.fast_api import run_fast_api
from app.logger_config import configure_logger
from app.train_model import train_model


def signal_handler(sig, frame):
    logging.info("Received termination signal. Shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Churn ML Application")
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to run the application in",
        choices=["fastapi", "train"],
        required=True,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--logfile", type=str, default=None, help="Path to a file to log messages"
    )

    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = configure_logger(log_level, args.logfile)
    logger.info("Starting the Churn ML Application")

    spinner = Halo(text="Loading", spinner="dots")
    spinner.start()

    try:
        if args.mode == "fastapi":
            logger.info("Running FastAPI mode")
            run_fast_api()
        elif args.mode == "train":
            logger.info("Running training mode")
            train_model()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        spinner.stop()
        logger.info("Shutting down the Churn ML Application")


if __name__ == "__main__":
    main()
