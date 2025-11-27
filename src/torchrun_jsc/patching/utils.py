from packaging import version
import torch


def get_torch_ver():
    torch_ver = version.parse(torch.__version__)
    return torch_ver
