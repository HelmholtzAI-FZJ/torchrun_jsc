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

Tested for PyTorch <2, 2.1.2, 2.4, 2.5.1, 2.6.0.
"""

from argparse import ArgumentParser, REMAINDER
import runpy

from packaging import version
import torch
from torch.distributed.argparse_util import check_env, env

from . import arg_patching
from . import node_desc_generator_patching
from . import parsing
from . import rendezvous_store_info_patching
from . import simple_elastic_agent_patching


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


def main():
    torch_ver = version.parse(torch.__version__)
    host, conf, is_host = parse_args()
    is_host = arg_patching.fix_is_host(is_host, conf)
    # Since PyTorch 2.4, we no longer need to fix `_get_fq_hostname`.
    if (
            torch_ver.major == 2 and torch_ver.minor <= 3
            or torch_ver.major == 1 and torch_ver.minor >= 9
    ):
        simple_elastic_agent_patching.fix_torch_run_simple_elastic_agent(host)
    # PyTorch 2.4 introduced a new `RendezvousStoreInfo` that requires
    # patching.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 4
    ):
        rendezvous_store_info_patching.fix_torch_run_rendezvous_store_info(
            host,
        )
    # PyTorch 2.5 started to use `_NodeDesc`s for more than just
    # logging. Since prior versions don't require this patch, we don't
    # apply it to decrease the surface of modifications.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 5
    ):
        node_desc_generator_patching.fix_torch_run_node_desc_generator(
            is_host,
            host,
        )
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
