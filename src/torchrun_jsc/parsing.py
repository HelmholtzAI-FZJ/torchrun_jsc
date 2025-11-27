"""
Common parsing routines for `torchrun` arguments.
"""

from argparse import ArgumentParser, REMAINDER

from torch.distributed.argparse_util import check_env, env


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
    # This inconsistent ordering is adapted for easier comparison with
    # the source.
    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
    )
    parser.add_argument('other_args', nargs=REMAINDER)
    args = parser.parse_known_args()[0]

    endpoint = args.rdzv_endpoint
    host = parse_host(endpoint, args.standalone)

    conf = args.rdzv_conf
    is_host = parse_is_host(conf)

    local_addr = args.local_addr

    return host, conf, is_host, local_addr


def parse_args_old():
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
    host = parse_host(endpoint, args.standalone)

    conf = args.rdzv_conf
    is_host = parse_is_host(conf)

    return host, conf, is_host
