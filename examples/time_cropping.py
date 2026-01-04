import torch
import time
from contextlib import contextmanager
from torch_subpixel_crop import subpixel_crop_2d

@contextmanager
def timer(name: str):
    """Simple timing context manager.
    
    Parameters
    ----------
    name : str
        Name to display for this timing section.
    """
    torch.cuda.synchronize()  # Wait for GPU operations to finish
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()  # Wait for GPU operations to finish
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f} ms")


# Test setup
device = 'cuda'
batch_size = 40
n_positions = 4
image_size = 4096
patch_size = 96

# Create test data
images = torch.randn(batch_size, image_size, image_size, device=device)
positions = torch.rand(n_positions, batch_size, 2, device=device) * image_size

# Warmup (important for accurate GPU timing!)
for _ in range(5):
    _ = subpixel_crop_2d(images, positions, patch_size)

# Time it
n_runs = 100
with timer(f"subpixel_crop_2d ({n_runs} runs)"):
    for _ in range(n_runs):
        patches = subpixel_crop_2d(images, positions, patch_size)

print(f"Output shape: {patches.shape}")
