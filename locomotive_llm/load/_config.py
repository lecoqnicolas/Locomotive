import json


def load_config(config_path, reverse=False):
    try:
        with open(config_path) as f:
            config = json.loads(f.read())
        if reverse:
            config["from"], config["to"] = config["to"], config["from"]
        return config
    except Exception as e:
        raise Exception(f"Cannot open config file") from e
