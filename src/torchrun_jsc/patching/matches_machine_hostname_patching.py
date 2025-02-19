import fcntl
import ipaddress
import socket
import struct
from typing import List, Tuple

from torch.distributed.elastic.rendezvous import utils as rutils


# From https://stackoverflow.com/a/27494105.
def nic_ip_address(nic_name: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', nic_name[:15].encode("UTF-8"))
    )[20:24])


# Adapted from https://stackoverflow.com/a/27494105.
def nic_info() -> List[Tuple[str, str]]:
    """Return a list of tuples containing each NIC's hostname and its IPv4."""
    nics: List[Tuple[str, str]] = []
    try:
        if_nameindex = socket.if_nameindex()
    except OSError:
        return nics

    for (_, nic_name) in if_nameindex:
        try:
            ip_addr = nic_ip_address(nic_name)
        except OSError:
            continue

        hostname = socket.gethostbyaddr(ip_addr)[0]
        nics.append((hostname, ip_addr))
    return nics


def fix_torch_run_matches_machine_hostname():
    old_matches_machine_hostname = rutils._matches_machine_hostname

    def new_matches_machine_hostname(host):
        is_match = old_matches_machine_hostname(host)

        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            addr = None

        try:
            host_addr_list = socket.getaddrinfo(
                host, None, proto=socket.IPPROTO_TCP, flags=socket.AI_CANONNAME
            )
        except (ValueError, socket.gaierror):
            host_addr_list = []

        host_ip_list = [
            host_addr_info[4][0]
            for host_addr_info in host_addr_list
        ]

        host_fqdn = socket.getfqdn(host)

        if not is_match:
            for (nic_host, nic_addr) in nic_info():
                if (
                        nic_host == host
                        or nic_host == host_fqdn
                        or addr and nic_addr == str(addr)
                        or nic_addr in host_ip_list
                ):
                    return True

        return is_match

    rutils._matches_machine_hostname = new_matches_machine_hostname
    return new_matches_machine_hostname
