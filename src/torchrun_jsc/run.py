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

from argparse import ArgumentParser
import runpy
import sys

from . import arg_patching
from . import parsing


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--rdzv_endpoint', '--rdzv-endpoint')
    parser.add_argument('--rdzv_conf', '--rdzv-conf')
    parser.add_argument('--local_addr', '--local-addr')
    args = parser.parse_known_args()[0]

    endpoint = args.rdzv_endpoint
    host = parsing.parse_host(endpoint)

    conf = args.rdzv_conf
    is_host = parsing.parse_is_host(conf)

    local_addr = args.local_addr

    return host, conf, is_host, local_addr


def fix_get_hostname(host, local_addr):
    if host and not local_addr:
        insertion_index = min(len(sys.argv), 1)
        sys.argv.insert(insertion_index, f'--local_addr={host}')


def main():
    host, conf, is_host, local_addr = parse_args()
    fix_get_hostname(host, local_addr)
    arg_patching.fix_is_host(is_host, conf)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
