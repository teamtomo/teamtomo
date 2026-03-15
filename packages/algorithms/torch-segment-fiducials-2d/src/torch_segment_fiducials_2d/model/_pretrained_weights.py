from pathlib import Path

import pooch

# https://github.com/fatiando/pooch
GOODBOY = pooch.create(
    path=pooch.os_cache("fidder"),
    base_url="doi:10.5281/zenodo.15831134/",
    registry={"fidder.ckpt": "md5:1f76bfc13757b1a5e84e6f6d6b51c8fe"},
)


def get_latest_checkpoint() -> Path:
    """Retrieve the latest checkpoint from cache if available or download."""
    checkpoint_file = Path(GOODBOY.fetch("fidder.ckpt"))
    return checkpoint_file
