"""
Common patching routines for `torchrun` arguments.
"""

import os
import sys


def fix_is_host(is_host, conf):
    nodeid = os.getenv('TORCHRUN_JSC_NODE_RANK', os.getenv('SLURM_NODEID'))

    # If `is_host` was already specified in the `torchrun`
    # configuration, we won't overwrite it.
    # If we could not obtain a `nodeid`, it means we have no information
    # so we better not touch anything.
    if is_host is None and nodeid is not None:
        # Now we check ourselves if we are the host.
        is_host = int(nodeid == os.getenv('TORCHRUN_JSC_HOST_NODE_RANK', '0'))

        if not conf:
            insertion_index = min(len(sys.argv), 1)
            sys.argv.insert(insertion_index, '--rdzv_conf=')

        # Since `torchrun` only uses standard `argparse` for
        # parsing, we do not need to worry about discerning multiple
        # `--rdzv_conf` arguments (one for `torchrun`, one for the
        # script).
        for (i, arg) in enumerate(sys.argv):
            if (
                    arg.startswith('--rdzv_conf')
                    or arg.startswith('--rdzv-conf')
            ):
                # Handle specification as two arguments vs. as one
                # argument.
                if arg in ['--rdzv_conf', '--rdzv-conf']:
                    modification_index = i + 1
                    old_conf = sys.argv[modification_index]
                else:
                    modification_index = i
                    old_conf = (
                        sys.argv[modification_index].split('=', 1)[1])

                # Handle empty conf specification.
                if old_conf:
                    sys.argv[modification_index] = (
                        f'{sys.argv[modification_index]},')
                sys.argv[modification_index] = (
                    f'{sys.argv[modification_index]}'
                    f'is_host={is_host}'
                )
                break

    return is_host


def fix_local_addr(is_host, host, local_addr):
    if is_host and host and not local_addr:
        insertion_index = min(len(sys.argv), 1)
        sys.argv.insert(insertion_index, f'--local_addr={host}')
