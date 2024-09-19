import json


def load_config(config_path, reverse=False):
    try:
        with open(config_path) as f:
            config = json.loads(f.read())
        if reverse:
            config["from"], config["to"] = config["to"], config["from"]
    except Exception as e:
        print(f"Cannot open config file: {e}")
        exit(1)
