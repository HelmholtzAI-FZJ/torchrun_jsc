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

Tested for PyTorch <2, 2.1.2, 2.4.
"""

from argparse import ArgumentParser
import inspect
import ipaddress
import runpy
import socket
import warnings

from packaging import version
import torch
from torch.distributed.elastic.agent.server import api as sapi
from torch.distributed.elastic.rendezvous import api as rapi
from torch.distributed.elastic.utils.distributed import get_free_port

from . import arg_patching
from . import parsing


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--standalone', action='store_true')
    parser.add_argument('--rdzv_endpoint', '--rdzv-endpoint')
    parser.add_argument('--rdzv_conf', '--rdzv-conf')
    args = parser.parse_known_args()[0]

    endpoint = args.rdzv_endpoint
    host = parsing.parse_host(endpoint, args.standalone)

    conf = args.rdzv_conf
    is_host = parsing.parse_is_host(conf)

    return host, conf, is_host


def build_get_fq_hostname_fn(host):
    assert host
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    if is_ip:
        def get_fq_hostname():
            return socket.gethostbyaddr(host)[0]
    else:
        def get_fq_hostname():
            return socket.getfqdn(host)

    return get_fq_hostname


def build_rendezvous_store_info_build_fn(host):
    get_fq_hostname = build_get_fq_hostname_fn(host)

    def new_build(rank, store):
        if rank == 0:
            addr = get_fq_hostname()
            port = get_free_port()
            store.set(
                rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
                addr.encode(encoding="UTF-8"),
            )
            store.set(
                rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
                str(port).encode(encoding="UTF-8"),
            )

        addr = store.get(
            rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
        ).decode(encoding="UTF-8")
        port = int(store.get(
            rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
        ).decode(encoding="UTF-8"))
        return rapi.RendezvousStoreInfo(master_addr=addr, master_port=port)

    return new_build


def fix_torch_run(host):
    orig_get_fq_hostname = sapi._get_fq_hostname
    orig_sig = inspect.signature(orig_get_fq_hostname)

    # Do not replace the function if the number of arguments doesn't
    # match (we expect no arguments in the original version).
    if host and not orig_sig.parameters:
        new_get_fq_hostname = build_get_fq_hostname_fn(host)
    else:
        new_get_fq_hostname = orig_get_fq_hostname

    sapi._get_fq_hostname = new_get_fq_hostname


def fix_torch_run_rdvz_store_info(host):
    torch_ver = version.parse(torch.__version__)
    assert (
        torch_ver.major >= 3
        or torch_ver.major == 2 and torch_ver.minor >= 4
    ), (
        "PyTorch version is too old for applying the "
        "`RendezvousStoreInfo` patch."
    )
    if not hasattr(rapi, 'RendezvousStoreInfo'):
        warnings.warn(
            'This version of PyTorch is not officially supported by '
            '`torchrun_jsc`; will not apply `RendezvousStoreInfo` patch. You '
            'may be able to ignore this warning.'
        )
        return

    orig_build = rapi.RendezvousStoreInfo.build
    orig_sig = inspect.signature(orig_build)

    # Do not replace the function if the number of arguments doesn't
    # match (we expect two arguments in the original version).
    if host and len(orig_sig.parameters) == 2:
        new_build = build_rendezvous_store_info_build_fn(host)
    else:
        new_build = orig_build

    rapi.RendezvousStoreInfo.build = new_build


def main():
    torch_ver = version.parse(torch.__version__)
    host, conf, is_host = parse_args()
    arg_patching.fix_is_host(is_host, conf)
    fix_torch_run(host)
    # PyTorch 2.4 introduced a new `RendezvousStoreInfo` that requires
    # patching.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 4
    ):
        fix_torch_run_rdvz_store_info(host)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
