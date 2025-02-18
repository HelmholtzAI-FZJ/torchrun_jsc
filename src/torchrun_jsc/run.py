"""
WARNING: Currently broken, do not use. Leads to indeterministic
deadlocks.

Fixed version of `torchrun` on Jülich Supercomputing Centre for PyTorch
versions ≥2. Requires Slurm usage.

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
"""

from argparse import ArgumentParser, REMAINDER
import runpy
import sys

from torch.distributed.argparse_util import check_env, env

from . import parsing
from . import patching


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--standalone', action=check_env)
    parser.add_argument(
        '--rdzv_endpoint',
        '--rdzv-endpoint',
        action=env,
        type=str,
        default='',
    )
    parser.add_argument(
        '--rdzv_conf',
        '--rdzv-conf',
        action=env,
        type=str,
        default='',
    )
    # This inconsistent ordering is adapted for easier comparison with
    # the source.
    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
    )
    parser.add_argument('other_args', nargs=REMAINDER)
    args = parser.parse_known_args()[0]

    endpoint = args.rdzv_endpoint
    host = parsing.parse_host(endpoint, args.standalone)

    conf = args.rdzv_conf
    is_host = parsing.parse_is_host(conf)

    local_addr = args.local_addr

    return host, conf, is_host, local_addr


def main():
    host, conf, is_host, local_addr = parse_args()
    is_host = patching.fix_is_host(is_host, conf)
    patching.fix_local_addr(is_host, host, local_addr)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
