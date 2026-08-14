"""Explicit read-only SmartAPI connectivity smoke check."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from algo_trader.broker import AngelOneBroker, AngelOneInstrumentMaster
from algo_trader.runtime.credentials import load_smartapi_credentials
from algo_trader.runtime.models import RuntimeConfig, RuntimeConnectivityReport


def run_smartapi_connectivity_check(
    *,
    config: RuntimeConfig,
    instrument_master: AngelOneInstrumentMaster,
    checked_at: datetime,
    quote_symbol: str | None = None,
    broker_factory: Callable[[AngelOneInstrumentMaster], AngelOneBroker] = AngelOneBroker,
    credentials_loader: Callable[..., object] = load_smartapi_credentials,
) -> RuntimeConnectivityReport:
    """Authenticate, perform only read operations, and explicitly log out."""
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("checked_at must be timezone-aware")
    credentials = credentials_loader(config.credential_path)
    broker = broker_factory(instrument_master)
    session = None
    try:
        session = broker.authenticate(credentials, checked_at)
        broker.get_funds()
        broker.list_positions()
        broker.list_orders()
        quote_ok = None
        if quote_symbol is not None:
            instrument_master.resolve(quote_symbol)
            broker.get_ltp(quote_symbol, checked_at)
            quote_ok = True
        return RuntimeConnectivityReport(
            authenticated=True,
            client_code=session.client_code,
            sdk_version=session.sdk_version,
            funds_read_ok=True,
            positions_read_ok=True,
            orders_read_ok=True,
            quote_read_ok=quote_ok,
            checked_at=checked_at,
        )
    finally:
        if session is not None:
            broker.logout(session)
