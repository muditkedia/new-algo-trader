"""Strict explicit SmartAPI.env parsing without environment mutation or search."""

import re
from pathlib import Path

from dotenv import dotenv_values

from algo_trader.broker import AngelOneCredentials

DEFAULT_SMARTAPI_ENV_PATH = Path(".secrets/SmartAPI.env")
_KEY_MAP = {
    "SMARTAPI_API_KEY": "api_key",
    "SMARTAPI_CLIENT_CODE": "client_code",
    "SMARTAPI_MPIN": "pin",
    "SMARTAPI_TOTP_SECRET": "totp_secret",
}
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)


def load_smartapi_credentials(
    path: Path = DEFAULT_SMARTAPI_ENV_PATH,
) -> AngelOneCredentials:
    """Load exactly four keys from the exact caller-selected file."""
    selected = Path(path)
    if not selected.exists():
        raise FileNotFoundError(f"SmartAPI credential file does not exist: {selected}")
    if not selected.is_file():
        raise ValueError(f"SmartAPI credential path is not a file: {selected}")
    try:
        source = selected.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("SmartAPI credential file cannot be read") from error
    keys = _ASSIGNMENT.findall(source)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate SmartAPI credential key(s): {', '.join(duplicates)}")
    parsed = dotenv_values(selected, interpolate=False)
    unexpected = sorted(set(parsed) - set(_KEY_MAP))
    if unexpected:
        raise ValueError(f"unexpected SmartAPI credential key(s): {', '.join(unexpected)}")
    missing = [key for key in _KEY_MAP if key not in parsed]
    if missing:
        raise ValueError(f"missing SmartAPI credential key(s): {', '.join(missing)}")
    normalized: dict[str, str] = {}
    for source_key, target_key in _KEY_MAP.items():
        value = parsed[source_key]
        if value is None or not value.strip():
            raise ValueError(f"SmartAPI credential key must be non-blank: {source_key}")
        normalized[target_key] = value.strip()
    return AngelOneCredentials(**normalized)
