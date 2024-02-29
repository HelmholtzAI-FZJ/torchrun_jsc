"""
Fixed version of `torchrun` on Jülich Supercomputing Centre for PyTorch
versions <2. Requires Slurm usage.

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
"""

from argparse import ArgumentParser
import ipaddress
import runpy
import socket

from torch.distributed.elastic.agent.server import api as sapi

from . import arg_patching
from . import parsing


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--rdzv_endpoint', '--rdzv-endpoint')
    parser.add_argument('--rdzv_conf', '--rdzv-conf')
    args = parser.parse_known_args()[0]

    endpoint = args.rdzv_endpoint
    host = parsing.parse_host(endpoint)

    conf = args.rdzv_conf
    is_host = parsing.parse_is_host(conf)

    return host, conf, is_host


def fix_torch_run(host):
    _orig_get_fq_hostname = sapi._get_fq_hostname

    if host:
        try:
            ipaddress.ip_address(host)
            is_ip = True
        except ValueError:
            is_ip = False

        if is_ip:
            def new_get_fq_hostname():
                return socket.gethostbyaddr(host)[0]
        else:
            def new_get_fq_hostname():
                return socket.getfqdn(host)
    else:
        new_get_fq_hostname = _orig_get_fq_hostname

    sapi._get_fq_hostname = new_get_fq_hostname


def main():
    host, conf, is_host = parse_args()
    arg_patching.fix_is_host(is_host, conf, 'SLURM_NODEID')
    fix_torch_run(host)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
