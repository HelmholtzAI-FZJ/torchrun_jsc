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

Tested for PyTorch <2, 2.1.2, 2.4, 2.6.
"""

from argparse import ArgumentParser, REMAINDER
import inspect
import ipaddress
import os
import runpy
import socket
import warnings

from packaging import version
import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.agent.server import api as sapi
from torch.distributed.elastic.rendezvous import api as rapi
from torch.distributed.elastic.rendezvous import dynamic_rendezvous
from torch.distributed.elastic.utils.distributed import get_free_port

from . import arg_patching
from . import parsing


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
    parser.add_argument('other_args', nargs=REMAINDER)
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

    torch_ver = version.parse(torch.__version__)
    if torch_ver.major >= 3 or torch_ver.major == 2 and torch_ver.minor >= 6:
        def new_build(rank, store, local_addr, server_port=None):
            if rank == 0:
                addr = local_addr or get_fq_hostname()
                port = server_port or get_free_port()
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
    elif torch_ver.major == 2 and torch_ver.minor >= 5 and torch_ver.minor < 6:
        def new_build(rank, store, local_addr):
            if rank == 0:
                addr = local_addr or get_fq_hostname()
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
    elif torch_ver.major == 2 and torch_ver.minor >= 4 and torch_ver.minor < 5:
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
    else:
        raise AssertionError(
            "PyTorch version is too old for applying the "
            "`RendezvousStoreInfo` patch."
        )

    return new_build


def build_node_desc_generator_generate_fn(host):
    get_fq_hostname = build_get_fq_hostname_fn(host)

    torch_ver = version.parse(torch.__version__)
    if torch_ver.major >= 2:
        def new_generate(self, local_addr=None):
            with self._lock:
                local_id = self._local_id

                self._local_id += 1

            return dynamic_rendezvous._NodeDesc(
                local_addr or get_fq_hostname(),
                os.getpid(),
                local_id,
            )
    elif torch_ver.major == 1 and torch_ver.minor >= 9:
        def new_generate(self):
            with self._lock:
                local_id = self._local_id

                self._local_id += 1

            return dynamic_rendezvous._NodeDesc(
                get_fq_hostname(),
                os.getpid(),
                local_id,
            )
    else:
        raise AssertionError(
            "PyTorch version is too old for applying the "
            "`_NodeDescGenerator` patch."
        )

    return new_generate


def fix_torch_run(host):
    orig_get_fq_hostname = sapi._get_fq_hostname
    orig_sig = inspect.signature(orig_get_fq_hostname)

    # Do not replace the function if the number of arguments doesn't
    # match (we expect no arguments in the original version).
    if host and not orig_sig.parameters:
        new_get_fq_hostname = build_get_fq_hostname_fn(host)
    else:
        if orig_sig.parameters:
            warnings.warn(
                'The function signature of a function that `torchrun_jsc` '
                'needs to patch has changed; will not apply `get_fq_hostname` '
                'patch. You may be able to ignore this warning.'
            )
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

    if torch_ver.major >= 3 or torch_ver.major == 2 and torch_ver.minor >= 6:
        num_orig_parameters = 4
    elif torch_ver.major == 2 and torch_ver.minor >= 5 and torch_ver.minor < 6:
        num_orig_parameters = 3
    elif torch_ver.major == 2 and torch_ver.minor >= 4 and torch_ver.minor < 5:
        num_orig_parameters = 2
    else:
        raise AssertionError(
            "PyTorch version is too old for applying the "
            "`RendezvousStoreInfo` patch."
        )

    # Do not replace the function if the number of arguments doesn't
    # match (we expect a certain number of arguments in the original
    # version).
    if host and len(orig_sig.parameters) == num_orig_parameters:
        new_build = build_rendezvous_store_info_build_fn(host)
    else:
        if len(orig_sig.parameters) != num_orig_parameters:
            warnings.warn(
                'The function signature of a function that `torchrun_jsc` '
                'needs to patch has changed; will not apply '
                '`RendezvousStoreInfo` patch. You may be able to ignore '
                'this warning.'
            )
        new_build = orig_build

    rapi.RendezvousStoreInfo.build = new_build


def fix_torch_run_node_desc_generator(is_host, host):
    torch_ver = version.parse(torch.__version__)
    # We could actually apply the patch to older versions, too, but
    # let's not bother with checking the function signature and whatnot
    # for now.
    assert (
        torch_ver.major >= 2
        or torch_ver.major == 1 and torch_ver.minor >= 9
    ), (
        "PyTorch version is too old for applying the "
        "`_NodeDescGenerator` patch."
    )

    # If we're not on the host node, don't change anything. If we did,
    # other nodes would obtain the same address as the host node, which
    # we don't want.
    if not is_host:
        return

    orig_generate = dynamic_rendezvous._NodeDescGenerator.generate
    orig_sig = inspect.signature(orig_generate)

    if torch_ver.major >= 2:
        num_orig_parameters = 2
    elif torch_ver.major == 1 and torch_ver.minor >= 9:
        num_orig_parameters = 1
    else:
        raise AssertionError(
            "PyTorch version is too old for applying the "
            "`_NodeDescGenerator` patch."
        )

    # Do not replace the function if the number of arguments doesn't
    # match (we expect a certain number of arguments in the original
    # version).
    if host and len(orig_sig.parameters) == num_orig_parameters:
        new_generate = build_node_desc_generator_generate_fn(host)
    else:
        if len(orig_sig.parameters) != num_orig_parameters:
            warnings.warn(
                'The function signature of a function that `torchrun_jsc` '
                'needs to patch has changed; will not apply '
                '`_NodeDescGenerator` patch. You may be able to ignore '
                'this warning.'
            )
        new_generate = orig_generate

    dynamic_rendezvous._NodeDescGenerator.generate = new_generate


def main():
    torch_ver = version.parse(torch.__version__)
    host, conf, is_host = parse_args()
    is_host = arg_patching.fix_is_host(is_host, conf)
    fix_torch_run(host)
    # PyTorch 2.4 introduced a new `RendezvousStoreInfo` that requires
    # patching.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 4
    ):
        fix_torch_run_rdvz_store_info(host)
    # PyTorch 2.5 started to use `_NodeDesc`s for more than just
    # logging. Since prior versions don't require this patch, we don't
    # apply it to decrease the surface of modifications.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 5
    ):
        fix_torch_run_node_desc_generator(is_host, host)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
