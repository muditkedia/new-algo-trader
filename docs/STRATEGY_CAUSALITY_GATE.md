# Strategy Causality Gate

The Strategy Causality Gate is the mandatory behavioral lookahead-prevention check for
future production strategies. It tests the earliest research boundary directly: canonical
candles into immutable strategy `Signal` values. It does not inspect source text or blacklist
particular APIs.

## Candle information timing

Historical candles are five-minute, start-stamped candles. A candle stamped 09:15 contains
activity through approximately 09:20, so its completed OHLCV is available at 09:20. The gate
uses only `algo_trader.data.bar_available_at(bar_start, 5)` for that calculation, preserves
timezone-aware timestamps, and operates on actual supplied timestamps. It neither assumes
contiguous rows nor synthesizes missing candles.

For each tested prefix, a signal timestamp must be no earlier than the first supplied candle's
availability and no later than the final candle's availability. A signal also undergoes a
knowledge-prefix check: the exact signal must already exist when the strategy receives only
the leading rows whose completed values were available by that signal's timestamp. Thus, a
strategy cannot use the completed 09:20-start candle to claim a 09:22 decision because that
candle is unavailable until 09:25.

## Behavioral prefix invariance

Historical/batch `generate_signals(candles)` returns the complete deterministic sequence of
decisions available within the supplied completed-candle history. Its output is cumulative,
not "latest signal only." If `P` is any tested prefix and `F` is future data, signals already
observable from `P` must be exactly the same after evaluating `P + F`.

The gate exhaustively evaluates every prefix from `max(1, warmup_bars)` through the full input.
For each consecutive extension, it filters the longer output to the earlier prefix's information
cutoff and compares that ordered sequence with the complete earlier output. Exact `Signal`
semantics are compared, including identity, side, timestamp, status, parameter snapshot, feature
snapshot, length, and order. The gate never sorts, repairs, or reconstructs output.

Every semantic prefix is evaluated at least twice. The two outputs must match exactly, and the
full prefix is evaluated again after all shorter prefixes to detect hidden mutable state or
cross-prefix accumulation. Strategy parameters and every supplied prefix must remain
semantically unchanged. Output must be an actual `list[Signal]`; metadata must match the
strategy and single-symbol frame; status must be `GENERATED`; timestamps must be nondecreasing;
and exact duplicate signals are rejected. Distinct signals at the same timestamp are allowed
when their order is deterministic and prefix-stable.

These checks behaviorally expose common lookahead patterns such as negative shifts or next-row
access, centered windows, full-frame extrema/means/ranks/normalization, retroactive signal
insertion or deletion, historical snapshot mutation, hidden strategy state, and nondeterministic
evaluation.

## Scope and required workflow

Passing establishes only that the tested strategy implementation is behaviorally causal on the
supplied candle frame. It does not prove profitability, statistical robustness, intellectual
strategy identity, OOS scope, universe/survivorship correctness, execution semantics, cost
accounting, or allocator timing. The gate does not consume OOS data, scan production market
data, train ML models, run backtests, contact brokers, or write files.

Every future production strategy implementation must follow this sequence:

```text
strategy implemented
    -> unit tests
    -> Strategy Causality Gate PASS
    -> only then research/backtesting/OOS/optimizer work
```

That strategy's tests must explicitly call `assert_strategy_prefix_invariant(...)` on a
deterministic representative fixture or history slice that exceeds warmup, spans enough
prefixes, includes a session boundary where relevant, and exercises its major branches. The
architecture-level harness passing its own adversarial tests does not certify any future
strategy. Strategy 1's research runner additionally locates a real signal in development or
explicitly `TRAINING_ALLOWED` history, includes post-signal rows, and fails closed unless the
existing gate compares a nontrivial number of prefixes with at least one signal. This is
research causality evidence, not a claim of profitability or production certification.
