import logging

from pathlib import Path

# Get project root (adjust number of parents if needed)
ROOT = Path(__file__).resolve().parents[2]

log_path = ROOT / "logs" / "train" / "train.log"

def setup_logger(LOG_FILE_PATH: str):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if LOG_FILE_PATH is not None:
        fh = logging.FileHandler(LOG_FILE_PATH)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


    return logger