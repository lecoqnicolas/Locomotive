import torch


def get_device(device: str|int):
    if isinstance(device, str):
        if device.lower() == "cpu":
            device = -1
        elif device.startswith("cuda"):
            device = torch.cuda.current_device()
        else:
            raise ValueError(f"Unsupported device: {device}")
    else:
        device = int(device)
    return device
