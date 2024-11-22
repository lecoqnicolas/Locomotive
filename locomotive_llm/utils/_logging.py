import os
import logging
from pathlib import Path

def init_log(log_file="./logs", log_level="INFO"):
    log_file = Path(log_file)
    os.makedirs(log_file.parent, exist_ok=True)
    logging.basicConfig(filename=log_file, encoding='utf-8', level=log_level)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Logging {log_file} Initialised")
