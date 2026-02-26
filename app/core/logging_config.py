import logging
import sys


def setup_logging() -> None:
    """
    Configure application-wide logging.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured (e.g., by uvicorn), don't reconfigure.
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

import logging
import sys


def setup_logging() -> None:
    """
    Basic logging configuration for the application.
    """
    root_logger = logging.getLogger()

    if root_logger.handlers:
        # Already configured (e.g., by Uvicorn); do not reconfigure.
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

