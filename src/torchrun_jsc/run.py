"""
Fixed version of `torchrun` on Jülich Supercomputing Centre for PyTorch
versions ≥2.5.

To use, modify your execution like the following:

Old
```shell
torchrun [...]
# or
python -m torch.distributed.run [...]
```

New
```shell
python /path/to/torchrun_jsc/run.py [...]
# or if `torchrun_jsc` is on `PYTHONPATH`
python -m torchrun_jsc.run [...]
```

Tested for PyTorch 2.5.0, 2.6.0.
"""

import runpy
import warnings

from packaging import version
import torch

from . import parsing
from . import patching


def torch_run_main():
    runpy.run_module('torch.distributed.run', run_name='__main__')


def main():
    torch_ver = version.parse(torch.__version__)
    if (
            torch_ver.major == 2 and torch_ver.minor < 5
            or torch_ver.major < 2
    ):
        warnings.warn(
            'This version of PyTorch is not officially supported by '
            '`torchrun_jsc`. You may be able to ignore this warning.'
        )

    host, conf, is_host, local_addr = parsing.parse_args()
    is_host = patching.fix_host_check(is_host, conf, host)
    patching.fix_local_addr(is_host, host, local_addr)
    torch_run_main()


if __name__ == '__main__':
    main()
