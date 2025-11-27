from importlib.metadata import version as importlib_metadata_version

from packaging import version as packaging_version


def get_torch_ver():
    torch_ver = packaging_version.parse(importlib_metadata_version('torch'))
    return torch_ver
