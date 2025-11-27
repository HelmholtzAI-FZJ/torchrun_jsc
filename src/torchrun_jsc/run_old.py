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

from . import parsing
from . import patching
from . import run


def main():
    torch_ver = patching.get_torch_ver()
    host, conf, is_host = parsing.parse_args_old()
    is_host = patching.fix_host_check(is_host, conf, host)
    # Since PyTorch 2.4, we no longer need to fix `_get_fq_hostname`.
    if (
            torch_ver.major == 2 and torch_ver.minor < 4
            or torch_ver.major == 1 and torch_ver.minor >= 9
    ):
        patching.fix_torch_run_simple_elastic_agent(host)
    # PyTorch 2.4 introduced a new `RendezvousStoreInfo` that requires
    # patching.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 4
    ):
        patching.fix_torch_run_rendezvous_store_info(
            host,
        )
    # PyTorch 2.5 started to use `_NodeDesc`s for more than just
    # logging. Since prior versions don't require this patch, we don't
    # apply it to decrease the surface of modifications.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 5
    ):
        patching.fix_torch_run_node_desc_generator(
            is_host,
            host,
        )
    run.torch_run_main()


if __name__ == '__main__':
    main()
