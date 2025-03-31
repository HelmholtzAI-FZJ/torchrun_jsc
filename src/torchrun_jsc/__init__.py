"""
Fixed version of `torchrun` on Jülich Supercomputing Centre for PyTorch
versions ≥1.9. Requires Slurm usage.

To use, modify your execution like the following:

Old
```shell
torchrun [...]
# or
python -m torch.distributed.run [...]
```

New
```shell
# if `torchrun_jsc` is on `PYTHONPATH`
python -m torchrun_jsc [...]
# or if `torchrun_jsc` is `pip`-installed
torchrun_jsc [...]
```
"""

import os
import warnings

from packaging import version
import torch


def main():
    torch_ver = version.parse(torch.__version__)
    if torch_ver.major > 2 or torch_ver.major == 2 and torch_ver.minor >= 5:
        if torch_ver.major > 2:
            warnings.warn(
                'This version of PyTorch is not officially supported by '
                '`torchrun_jsc`. You may be able to ignore this warning.'
            )

        if bool(int(os.getenv('TORCHRUN_JSC_PREFER_OLD_SOLUTION', '0'))):
            from .run_old import main as run_main_old
            run_main_old()
        else:
            from .run import main as run_main
            run_main()
    elif torch_ver.major == 2 or torch_ver.major == 1 and torch_ver.minor >= 9:
        from .run_old import main as run_main_old
        run_main_old()
    else:
        raise RuntimeError(
            'This version of PyTorch is not supported by `torchrun_jsc` '
            'because it does not have the `torchrun` API implemented. '
            'Please use another launch API.'
        )
