"""
Fixed version of `torchrun` on Jülich Supercomputing Centre for PyTorch
versions <2.5.

To use, modify your execution like the following:

Old
```shell
torchrun [...]
# or
python -m torch.distributed.run [...]
```

New
```shell
python /path/to/torchrun_jsc/run_old.py [...]
# or if `torchrun_jsc` is on `PYTHONPATH`
python -m torchrun_jsc.run_old [...]
```

Tested for PyTorch <2, 2.1.2, 2.4, 2.5.1, 2.6.0.
"""

from . import patching
from . import run


def main():
    patching.fix_torch_run_old()
    run.torch_run_main()


if __name__ == '__main__':
    main()
