"""安全相关：调用方身份解析与不可信数据隔离。"""
from .auth import Principal, parse_api_keys, resolve_principal
from .untrusted import wrap_untrusted

__all__ = ["Principal", "parse_api_keys", "resolve_principal", "wrap_untrusted"]
