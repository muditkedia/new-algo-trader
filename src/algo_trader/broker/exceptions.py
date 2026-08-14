"""Safe broker integration exception hierarchy."""


class BrokerError(RuntimeError):
    """Base error for the external broker boundary."""


class BrokerAuthenticationError(BrokerError):
    """Authentication or explicit session-refresh failure."""


class BrokerApiError(BrokerError):
    """A broker operation returned a declared failure."""


class BrokerSystemicError(BrokerApiError):
    """A shared broker/API failure for which repeating equivalent calls is unsafe."""


class BrokerDataError(BrokerError):
    """A broker response could not be normalized safely."""


class BrokerInstrumentError(BrokerError):
    """Instrument-master resolution failed or was ambiguous."""


class BrokerAmbiguousStateError(BrokerError):
    """Broker evidence matched more than one possible order state."""
