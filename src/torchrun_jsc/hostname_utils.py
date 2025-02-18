import ipaddress
import socket


def build_get_fq_hostname_fn(host):
    assert host
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    if is_ip:
        def get_fq_hostname():
            return socket.gethostbyaddr(host)[0]
    else:
        def get_fq_hostname():
            return socket.getfqdn(host)

    return get_fq_hostname
