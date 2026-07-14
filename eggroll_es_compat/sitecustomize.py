"""Compatibility aliases loaded in every Ray worker Python process."""

try:
    import vllm.utils
    from vllm.utils.network_utils import get_ip, get_open_port
except ImportError:
    pass
else:
    vllm.utils.get_ip = get_ip
    vllm.utils.get_open_port = get_open_port
