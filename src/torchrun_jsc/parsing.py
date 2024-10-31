"""
Common parsing routines for `torchrun` arguments.
"""


def _as_bool(key, value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    elif isinstance(value, str):
        if value.lower() in ['1', 'true', 't', 'yes', 'y']:
            return True
        if value.lower() in ['0', 'false', 'f', 'no', 'n']:
            return False
    raise ValueError(
        f'The rendezvous configuration option {key} does not represent a '
        f'valid boolean value.'
    )


def parse_host(rdzv_endpoint, standalone):
    if standalone:
        return 'localhost'
    host = (
        rdzv_endpoint.rsplit(':', 1)[0]
        if rdzv_endpoint
        else None
    )
    return host


def parse_is_host(rdzv_conf):
    is_host = None
    if rdzv_conf:
        confs = rdzv_conf.split(',')
        for (key, value) in map(lambda kv: kv.split('=', 1), confs):
            if key == 'is_host':
                is_host = _as_bool(key, value)
                break
    return is_host
