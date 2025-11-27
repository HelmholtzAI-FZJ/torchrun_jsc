import os

from .arg_patching import fix_is_host
from .matches_machine_hostname_patching import (
    fix_torch_run_matches_machine_hostname,
)


def fix_host_check(is_host, conf, host):
    if bool(int(os.getenv('TORCHRUN_JSC_PREFER_ARG_PATCHING', '1'))):
        is_host = fix_is_host(is_host, conf)
    else:
        new_matches_machine_hostname = \
            fix_torch_run_matches_machine_hostname()
        is_host = new_matches_machine_hostname(host)
    return is_host
