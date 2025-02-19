import os

from .arg_patching import fix_is_host, fix_local_addr
from .matches_machine_hostname_patching import (
    fix_torch_run_matches_machine_hostname,
)
from .node_desc_generator_patching import fix_torch_run_node_desc_generator
from .rendezvous_store_info_patching import fix_torch_run_rendezvous_store_info
from .simple_elastic_agent_patching import fix_torch_run_simple_elastic_agent


def fix_host_check(is_host, conf, host):
    if bool(int(os.getenv('TORCHRUN_JSC_PREFER_ARG_PATCHING', '1'))):
        is_host = fix_is_host(is_host, conf)
    else:
        new_matches_machine_hostname = \
            fix_torch_run_matches_machine_hostname()
        is_host = new_matches_machine_hostname(host)
    return is_host
