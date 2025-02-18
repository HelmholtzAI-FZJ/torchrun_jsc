import inspect
import warnings

from torch.distributed.elastic.agent.server import api as sapi

from .. import hostname_utils


def fix_torch_run_simple_elastic_agent(host):
    orig_get_fq_hostname = sapi._get_fq_hostname
    orig_sig = inspect.signature(orig_get_fq_hostname)

    # Do not replace the function if the number of arguments doesn't
    # match (we expect no arguments in the original version).
    if host and not orig_sig.parameters:
        new_get_fq_hostname = hostname_utils.build_get_fq_hostname_fn(host)
    else:
        if orig_sig.parameters:
            warnings.warn(
                'The function signature of a function that `torchrun_jsc` '
                'needs to patch has changed; will not apply `get_fq_hostname` '
                'patch. You may be able to ignore this warning.'
            )
        new_get_fq_hostname = orig_get_fq_hostname

    sapi._get_fq_hostname = new_get_fq_hostname
