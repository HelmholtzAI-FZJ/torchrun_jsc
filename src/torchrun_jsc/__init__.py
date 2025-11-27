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

from . import patching
from . import run


def apply_fixes(strict=True):
    is_supported_torch_version = True
    torch_ver = patching.get_torch_ver()
    if torch_ver.major > 2 or torch_ver.major == 2 and torch_ver.minor >= 5:
        if torch_ver.major > 2:
            warnings.warn(
                'This version of PyTorch is not officially supported by '
                '`torchrun_jsc`. You may be able to ignore this warning.'
            )

        if bool(int(os.getenv('TORCHRUN_JSC_PREFER_OLD_SOLUTION', '0'))):
            patching.fix_torch_run_old()
        else:
            patching.fix_torch_run()
    elif torch_ver.major == 2 or torch_ver.major == 1 and torch_ver.minor >= 9:
        patching.fix_torch_run_old()
    else:
        # Applying fixes failed due to too old PyTorch version.
        is_supported_torch_version = False

    if strict and not is_supported_torch_version:
        raise RuntimeError(
            'This version of PyTorch is not supported by `torchrun_jsc` '
            'because it does not have the `torchrun` API implemented. '
            'Please use another launch API.'
        )
    return is_supported_torch_version


def main():
    apply_fixes()
    run.torch_run_main()
