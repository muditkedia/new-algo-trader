from decimal import Decimal

import pytest

from algo_trader.execution import (
    ExecutionAction,
    FixedBasisPointsSlippage,
    NoSlippage,
    SlippageModel,
)


@pytest.mark.parametrize("action", list(ExecutionAction))
def test_no_slippage_leaves_price_unchanged(action: ExecutionAction) -> None:
    model = NoSlippage()

    assert isinstance(model, SlippageModel)
    assert model.apply(Decimal("100"), action) == Decimal("100")


def test_fixed_basis_points_worsens_buy_upward() -> None:
    model = FixedBasisPointsSlippage(Decimal("10"))

    assert model.apply(Decimal("100"), ExecutionAction.BUY) == Decimal("100.10")


def test_fixed_basis_points_worsens_sell_downward() -> None:
    model = FixedBasisPointsSlippage(Decimal("10"))

    assert model.apply(Decimal("100"), ExecutionAction.SELL) == Decimal("99.90")


@pytest.mark.parametrize("action", list(ExecutionAction))
def test_zero_basis_points_leaves_price_unchanged(action: ExecutionAction) -> None:
    model = FixedBasisPointsSlippage(Decimal("0"))

    assert model.apply(Decimal("100"), action) == Decimal("100")


def test_negative_basis_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FixedBasisPointsSlippage(Decimal("-0.01"))
