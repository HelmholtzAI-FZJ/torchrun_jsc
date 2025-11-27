import warnings

from . import utils
from .arg_patching import fix_local_addr
from .host_check_patching import fix_host_check
from .node_desc_generator_patching import fix_torch_run_node_desc_generator
from .rendezvous_store_info_patching import fix_torch_run_rendezvous_store_info
from .simple_elastic_agent_patching import fix_torch_run_simple_elastic_agent
from .. import parsing


def fix_torch_run():
    torch_ver = utils.get_torch_ver()
    if (
            torch_ver.major == 2 and torch_ver.minor < 5
            or torch_ver.major < 2
    ):
        warnings.warn(
            'This version of PyTorch is not officially supported by '
            '`torchrun_jsc`. You may be able to ignore this warning.'
        )

    host, conf, is_host, local_addr = parsing.parse_args()
    is_host = fix_host_check(is_host, conf, host)
    fix_local_addr(is_host, host, local_addr)


def fix_torch_run_old():
    torch_ver = utils.get_torch_ver()
    host, conf, is_host = parsing.parse_args_old()
    is_host = fix_host_check(is_host, conf, host)
    # Since PyTorch 2.4, we no longer need to fix `_get_fq_hostname`.
    if (
            torch_ver.major == 2 and torch_ver.minor < 4
            or torch_ver.major == 1 and torch_ver.minor >= 9
    ):
        fix_torch_run_simple_elastic_agent(host)
    # PyTorch 2.4 introduced a new `RendezvousStoreInfo` that requires
    # patching.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 4
    ):
        fix_torch_run_rendezvous_store_info(
            host,
        )
    # PyTorch 2.5 started to use `_NodeDesc`s for more than just
    # logging. Since prior versions don't require this patch, we don't
    # apply it to decrease the surface of modifications.
    if (
            torch_ver.major >= 3
            or torch_ver.major == 2 and torch_ver.minor >= 5
    ):
        fix_torch_run_node_desc_generator(
            is_host,
            host,
        )
