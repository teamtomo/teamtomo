import functools

import torch

AVAILABLE_DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    AVAILABLE_DEVICES.append("mps")
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("cuda")


def device_test(test_func):
    @functools.wraps(test_func)
    def run_devices(*args, **kwargs):
        for device in AVAILABLE_DEVICES:
            with torch.device(device):
                test_func(*args, **kwargs)

    return run_devices
