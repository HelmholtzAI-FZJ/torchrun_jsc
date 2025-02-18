import inspect
import os
import warnings

from packaging import version
import torch
from torch.distributed.elastic.rendezvous import dynamic_rendezvous

from .. import hostname_utils


def build_node_desc_generator_generate_fn(host):
    get_fq_hostname = hostname_utils.build_get_fq_hostname_fn(host)

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
