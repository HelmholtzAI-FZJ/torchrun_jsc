import inspect
import warnings

try:
    from torch.distributed.elastic.rendezvous import api as rapi
except (ModuleNotFoundError, ImportError):
    rapi = None
try:
    from torch.distributed.elastic.utils.distributed import get_free_port
except (ModuleNotFoundError, ImportError):
    # This is copied from PyTorch.
    from contextlib import closing
    import socket

    def _get_socket_with_port() -> socket.socket:
        addrs = socket.getaddrinfo(
            host="localhost",
            port=None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        for addr in addrs:
            family, type_, proto, _, _ = addr
            try:
                s = socket.socket(family, type_, proto)
                s.bind(("localhost", 0))
                s.listen(0)
                return s
            except OSError as e:
                s.close()
                print(f"Socket creation attempt failed: {e}")
        raise RuntimeError("Failed to create a socket")

    def get_free_port():
        sock = _get_socket_with_port()
        with closing(sock):
            return sock.getsockname()[1]


from . import utils
from .. import hostname_utils


def build_rendezvous_store_info_build_fn(host):
    get_fq_hostname = hostname_utils.build_get_fq_hostname_fn(host)

    torch_ver = utils.get_torch_ver()
    if torch_ver.major >= 3 or torch_ver.major == 2 and torch_ver.minor >= 6:
        assert rapi is not None, (
            "PyTorch version is too new for applying the "
            "`RendezvousStoreInfo` patch."
        )

        def new_build(rank, store, local_addr, server_port=None):
            if rank == 0:
                addr = local_addr or get_fq_hostname()
                port = server_port or get_free_port()
                store.set(
                    rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
                    addr.encode(encoding="UTF-8"),
                )
                store.set(
                    rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
                    str(port).encode(encoding="UTF-8"),
                )

            addr = store.get(
                rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
            ).decode(encoding="UTF-8")
            port = int(store.get(
                rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
            ).decode(encoding="UTF-8"))
            return rapi.RendezvousStoreInfo(master_addr=addr, master_port=port)
    elif torch_ver.major == 2 and torch_ver.minor >= 5 and torch_ver.minor < 6:
        # This should never raise.
        assert rapi is not None

        def new_build(rank, store, local_addr):
            if rank == 0:
                addr = local_addr or get_fq_hostname()
                port = get_free_port()
                store.set(
                    rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
                    addr.encode(encoding="UTF-8"),
                )
                store.set(
                    rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
                    str(port).encode(encoding="UTF-8"),
                )

            addr = store.get(
                rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
            ).decode(encoding="UTF-8")
            port = int(store.get(
                rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
            ).decode(encoding="UTF-8"))
            return rapi.RendezvousStoreInfo(master_addr=addr, master_port=port)
    elif torch_ver.major == 2 and torch_ver.minor >= 4 and torch_ver.minor < 5:
        # This should never raise.
        assert rapi is not None

        def new_build(rank, store):
            if rank == 0:
                addr = get_fq_hostname()
                port = get_free_port()
                store.set(
                    rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
                    addr.encode(encoding="UTF-8"),
                )
                store.set(
                    rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
                    str(port).encode(encoding="UTF-8"),
                )

            addr = store.get(
                rapi.RendezvousStoreInfo.MASTER_ADDR_KEY,
            ).decode(encoding="UTF-8")
            port = int(store.get(
                rapi.RendezvousStoreInfo.MASTER_PORT_KEY,
            ).decode(encoding="UTF-8"))
            return rapi.RendezvousStoreInfo(master_addr=addr, master_port=port)
    else:
        raise AssertionError(
            "PyTorch version is too old for applying the "
            "`RendezvousStoreInfo` patch."
        )

    return new_build


def fix_torch_run_rendezvous_store_info(host):
    torch_ver = utils.get_torch_ver()
    assert (
        torch_ver.major >= 3
        or torch_ver.major == 2 and torch_ver.minor >= 4
    ), (
        "PyTorch version is too old for applying the "
        "`RendezvousStoreInfo` patch."
    )
    if rapi is None or not hasattr(rapi, 'RendezvousStoreInfo'):
        warnings.warn(
            'This version of PyTorch is not officially supported by '
            '`torchrun_jsc`; will not apply `RendezvousStoreInfo` patch. You '
            'may be able to ignore this warning.'
        )
        return

    orig_build = rapi.RendezvousStoreInfo.build
    orig_sig = inspect.signature(orig_build)

    if torch_ver.major >= 3 or torch_ver.major == 2 and torch_ver.minor >= 6:
        num_orig_parameters = 4
    elif torch_ver.major == 2 and torch_ver.minor >= 5 and torch_ver.minor < 6:
        num_orig_parameters = 3
    elif torch_ver.major == 2 and torch_ver.minor >= 4 and torch_ver.minor < 5:
        num_orig_parameters = 2
    else:
        raise AssertionError(
            "PyTorch version is too old for applying the "
            "`RendezvousStoreInfo` patch."
        )

    # Do not replace the function if the number of arguments doesn't
    # match (we expect a certain number of arguments in the original
    # version).
    if host and len(orig_sig.parameters) == num_orig_parameters:
        new_build = build_rendezvous_store_info_build_fn(host)
    else:
        if len(orig_sig.parameters) != num_orig_parameters:
            warnings.warn(
                'The function signature of a function that `torchrun_jsc` '
                'needs to patch has changed; will not apply '
                '`RendezvousStoreInfo` patch. You may be able to ignore '
                'this warning.'
            )
        new_build = orig_build

    rapi.RendezvousStoreInfo.build = new_build
