from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from algo_trader import (
    ExitReason,
    Fill,
    MLScore,
    ProtectiveExitSpec,
    Side,
    Signal,
    SignalStatus,
    Trade,
)

NOW = datetime(2025, 1, 2, 9, 20, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def make_signal(status: SignalStatus = SignalStatus.GENERATED) -> Signal:
    return Signal(
        strategy_id="opening-range",
        strategy_version="1.0.0",
        symbol="RELIANCE",
        timestamp=NOW,
        side=Side.LONG,
        strategy_parameters={"lookback": 20},
        feature_snapshot={"atr": 12.5},
        status=status,
    )


def make_ml_score(recommended_notional: int = 75_000) -> MLScore:
    return MLScore(
        model_version="meta-1",
        quality_score=0.72,
        calibrated_probability=0.64,
        predicted_net_return=0.004,
        recommended_notional=recommended_notional,
    )


def make_trade(
    *,
    status: SignalStatus,
    is_shadow: bool,
    target_notional: int = 75_000,
    entry_timestamp: datetime | None = None,
) -> Trade:
    return Trade(
        signal=make_signal(status),
        ml_score=make_ml_score(),
        target_notional=target_notional,
        entry_fill=Fill(
            timestamp=entry_timestamp or NOW + timedelta(minutes=5),
            price=Decimal("2500.00"),
            quantity=30,
            slippage_per_unit=Decimal("0.25"),
            is_simulated=is_shadow,
        ),
        exit_fill=Fill(
            timestamp=NOW + timedelta(minutes=35),
            price=Decimal("2520.00"),
            quantity=30,
            slippage_per_unit=Decimal("0.30"),
            is_simulated=is_shadow,
        ),
        gross_pnl=Decimal("600.00"),
        total_costs=Decimal("50.00"),
        net_pnl=Decimal("550.00"),
        mfe_return=Decimal("0.010"),
        mae_return=Decimal("-0.0024"),
        exit_reason=ExitReason.TARGET_REACHED,
        is_shadow=is_shadow,
    )


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (Signal, {"timestamp": datetime(2025, 1, 2, 9, 20)}),
        (
            Fill,
            {
                "timestamp": datetime(2025, 1, 2, 9, 25),
                "price": Decimal("100"),
                "quantity": 1,
                "is_simulated": True,
            },
        ),
    ],
)
def test_timezone_naive_datetime_is_rejected(model: type, kwargs: dict) -> None:
    if model is Signal:
        kwargs = {
            "strategy_id": "test",
            "strategy_version": "1",
            "symbol": "ABC",
            "side": Side.LONG,
            **kwargs,
        }

    with pytest.raises(ValidationError, match="timezone-aware"):
        model(**kwargs)


def test_timezone_aware_timestamp_is_accepted() -> None:
    signal = make_signal()

    assert signal.timestamp == NOW
    assert signal.timestamp.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("notional", [50_000, 100_000])
def test_ml_notional_inclusive_bounds_are_valid(notional: int) -> None:
    assert make_ml_score(notional).recommended_notional == notional


@pytest.mark.parametrize("notional", [49_999, 100_001])
def test_ml_notional_outside_bounds_is_rejected(notional: int) -> None:
    with pytest.raises(ValidationError):
        make_ml_score(notional)


def test_ml_notional_must_use_five_thousand_increment() -> None:
    with pytest.raises(ValidationError, match="increments of 5000"):
        make_ml_score(52_500)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_calibrated_probability_outside_unit_interval_is_rejected(probability: float) -> None:
    with pytest.raises(ValidationError):
        MLScore(
            model_version="meta-1",
            quality_score=0.5,
            calibrated_probability=probability,
            predicted_net_return=0.001,
            recommended_notional=50_000,
        )


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_calibrated_probability_inclusive_bounds_are_valid(probability: float) -> None:
    score = MLScore(
        model_version="meta-1",
        quality_score=0.5,
        calibrated_probability=probability,
        predicted_net_return=0.001,
        recommended_notional=50_000,
    )

    assert score.calibrated_probability == probability


@pytest.mark.parametrize("quality_score", [-0.01, 1.01])
def test_quality_score_outside_unit_interval_is_rejected(quality_score: float) -> None:
    with pytest.raises(ValidationError):
        MLScore(
            model_version="meta-1",
            quality_score=quality_score,
            calibrated_probability=0.5,
            predicted_net_return=0.005,
            recommended_notional=50_000,
        )


@pytest.mark.parametrize("quality_score", [0.0, 1.0])
def test_quality_score_inclusive_bounds_are_valid(quality_score: float) -> None:
    score = MLScore(
        model_version="meta-1",
        quality_score=quality_score,
        calibrated_probability=0.5,
        predicted_net_return=0.005,
        recommended_notional=50_000,
    )

    assert score.quality_score == quality_score


def test_enums_have_stable_string_values() -> None:
    assert Side.LONG.value == "LONG"
    assert Side.SHORT.value == "SHORT"
    assert SignalStatus.CAPACITY_REJECTED.value == "CAPACITY_REJECTED"
    assert ExitReason.TIME_EXIT.value == "TIME_EXIT"
    assert ExitReason.STRATEGY_EXIT.value == "STRATEGY_EXIT"


def test_construct_normal_executed_trade() -> None:
    trade = make_trade(status=SignalStatus.EXECUTED, is_shadow=False)

    assert trade.signal.status is SignalStatus.EXECUTED
    assert trade.net_pnl == Decimal("550.00")
    assert trade.mfe_return == Decimal("0.010")
    assert trade.mae_return == Decimal("-0.0024")
    assert not trade.is_shadow


def test_construct_capacity_rejected_shadow_trade() -> None:
    trade = make_trade(status=SignalStatus.CAPACITY_REJECTED, is_shadow=True)

    assert trade.signal.status is SignalStatus.CAPACITY_REJECTED
    assert trade.entry_fill.is_simulated
    assert trade.is_shadow


def test_trade_preserves_original_ml_score() -> None:
    ml_score = make_ml_score()
    trade = make_trade(status=SignalStatus.EXECUTED, is_shadow=False).model_copy(
        update={"ml_score": ml_score}
    )

    assert trade.ml_score is ml_score
    assert trade.ml_score.model_dump() == ml_score.model_dump()


@pytest.mark.parametrize(
    ("field", "value"),
    [("mfe_return", Decimal("-0.001")), ("mae_return", Decimal("0.001"))],
)
def test_trade_excursion_return_sign_is_validated(field: str, value: Decimal) -> None:
    trade_data = make_trade(status=SignalStatus.EXECUTED, is_shadow=False).model_dump()
    trade_data[field] = value

    with pytest.raises(ValidationError):
        Trade.model_validate(trade_data)


def test_target_notional_must_match_ml_recommendation() -> None:
    with pytest.raises(ValidationError, match="target_notional must equal"):
        make_trade(
            status=SignalStatus.EXECUTED,
            is_shadow=False,
            target_notional=80_000,
        )


@pytest.mark.parametrize("target_notional", [49_999, 52_500, 100_001])
def test_target_notional_uses_valid_notional_bucket(target_notional: int) -> None:
    with pytest.raises(ValidationError):
        make_trade(
            status=SignalStatus.EXECUTED,
            is_shadow=False,
            target_notional=target_notional,
        )


def test_trade_derives_notional_and_returns_from_fills_and_pnl() -> None:
    trade = make_trade(status=SignalStatus.EXECUTED, is_shadow=False)

    assert trade.actual_entry_notional == Decimal("75000.00")
    assert trade.gross_return == Decimal("0.008")
    assert trade.net_return == Decimal("0.007333333333333333333333333333")
    assert "actual_entry_notional" not in type(trade).model_fields
    assert "gross_return" not in type(trade).model_fields
    assert "net_return" not in type(trade).model_fields


def test_entry_fill_cannot_precede_signal() -> None:
    with pytest.raises(ValidationError, match="entry fill timestamp cannot precede"):
        make_trade(
            status=SignalStatus.EXECUTED,
            is_shadow=False,
            entry_timestamp=NOW - timedelta(microseconds=1),
        )


def test_signal_snapshots_are_detached_and_immutable() -> None:
    parameters = {"periods": [5, 20]}
    signal = Signal(
        strategy_id="test",
        strategy_version="1",
        symbol="ABC",
        timestamp=NOW,
        side=Side.SHORT,
        strategy_parameters=parameters,
        feature_snapshot={},
    )
    parameters["periods"].append(50)

    assert signal.strategy_parameters["periods"] == (5, 20)
    with pytest.raises(TypeError):
        signal.strategy_parameters["new"] = 1  # type: ignore[index]


def test_protective_exit_requires_at_least_one_price() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        ProtectiveExitSpec()


@pytest.mark.parametrize("field", ["stop_price", "target_price"])
@pytest.mark.parametrize("price", [Decimal("0"), Decimal("-1")])
def test_protective_exit_prices_must_be_positive(field: str, price: Decimal) -> None:
    with pytest.raises(ValidationError):
        ProtectiveExitSpec(**{field: price})


def test_fill_rejects_negative_slippage() -> None:
    with pytest.raises(ValidationError):
        Fill(
            timestamp=NOW,
            price=Decimal("100"),
            quantity=1,
            slippage_per_unit=Decimal("-0.01"),
            is_simulated=True,
        )
