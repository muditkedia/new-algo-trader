"""Explicit, authorization-gated Strategy 1 optimizer entrypoint.

This command never runs as part of an ordinary development or OOS backtest. The
caller supplies an evaluator factory because market-data/backtest composition is
environment-specific; the existing optimizer still validates every evaluation
range against the OOS registry before the evaluator is called.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from algo_trader.ml import StrategyOptimizerConfig, optimize_strategy_parameters
from algo_trader.oos import OOSRegistry
from algo_trader.research import canonical_fingerprint
from algo_trader.strategies import LiquidityShockReclaimStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--evaluator-factory",
        required=True,
        help="Explicit module:callable factory returning a StrategyParameterEvaluator.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_evaluator(factory_reference: str):
    if factory_reference.count(":") != 1:
        raise ValueError("--evaluator-factory must use module:callable syntax")
    module_name, attribute_name = factory_reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    evaluator = factory()
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError("evaluator factory must return a StrategyParameterEvaluator")
    return evaluator


def validate_strategy1_config(config: StrategyOptimizerConfig) -> None:
    strategy = LiquidityShockReclaimStrategy()
    if config.strategy_id != strategy.strategy_id:
        raise ValueError("optimizer config strategy_id is not Strategy 1")
    if dict(config.baseline_parameters) != dict(strategy.parameters):
        raise ValueError(
            "optimizer baseline_parameters must exactly match the canonical Strategy 1 config"
        )


def main() -> None:
    args = parse_args()
    config = StrategyOptimizerConfig.model_validate_json(
        args.config.read_text(encoding="utf-8")
    )
    validate_strategy1_config(config)
    evaluator = load_evaluator(args.evaluator_factory)
    if args.output.exists():
        raise FileExistsError(f"optimizer output already exists: {args.output}")
    with OOSRegistry(args.registry) as registry:
        result = optimize_strategy_parameters(config, evaluator, registry)
    payload = {
        "strategy_config_fingerprint": canonical_fingerprint(
            dict(config.baseline_parameters)
        ),
        "optimizer_config": config.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)


if __name__ == "__main__":
    main()
