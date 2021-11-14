from pathlib import Path
import json

DATA_FOLDER = Path("../../data")
CONFIG_PATH = DATA_FOLDER / "config.json"
CONFIG = None


def get_configuration() -> dict:
    global CONFIG

    if not CONFIG:
        CONFIG = json.load(CONFIG_PATH)

    return CONFIG
