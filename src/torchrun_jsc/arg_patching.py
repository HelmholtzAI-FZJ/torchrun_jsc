"""
Common patching routines for `torchrun` arguments.
"""

import os
import sys


def fix_is_host(is_host, conf):
    slurm_nodeid = os.getenv('SLURM_NODEID')

    # If `is_host` was already specified in the `torchrun`
    # configuration, we won't overwrite it.
    # If `SLURM_NODEID` is not set, it means we have no information so
    # we better not touch anything.
    if is_host is None and slurm_nodeid is not None:
        # Now we check ourselves if we are the host.
        is_host = int(slurm_nodeid == '0')

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
