# Integration Repair Contracts

Research review output is atomic and intentionally small. The directory
`results/research/<scope>/UPLOAD_FOR_REVIEW/` contains exactly
`strategy1_research_history.json` and `strategy1_research_master.xlsx`. Canonical per-run
results, reports, manifests, Parquet tables, figures, margin snapshots, and model artifacts
remain internal evidence. A manifest-bearing corrupt run fails cumulative generation; an
explicit `.tmp` or manifest-less run is reported as incomplete.

Runs are written under `<run_id>.tmp`, validated, and atomically renamed before an OOS result
may be registered `TESTED`. The report can carry the deterministic canonical result fingerprint
before registry mutation. Market-data subsets, strategy configuration, margin evidence, model
identity, costs, slippage, environment versions, runner bytes, and Git state contribute explicit
provenance and the run-input fingerprint.

Strategy 1 v1.1.0 uses one frozen configuration object for behavior and serialized parameters.
Its research path scores a signal before recommended notional and quantity are constructed.
Only a checksum-verified, compatible model with explicit evaluation evidence may become active;
otherwise the bootstrap scorer is recorded visibly. Candidate training sources are restricted to
development and OOS ranges already marked `TRAINING_ALLOWED`; sealed or tested-unapproved data
cannot be selected.

Generic research adapters own only strategy-specific factories and request/exit composition.
The historical backtester, allocator, costs, reporting arithmetic, and OOS registry remain their
existing independent owners.

Runtime has an explicit composition root and no default LIVE mode. The scheduled production
lifecycle includes completed five-minute candle evaluation, market-data health before entries,
the 15:30 market-close safety check, stream startup, explicit trading-calendar input, and an
independent broker-funds ceiling for LIVE. PAPER execution assumptions, including adverse
slippage, are part of `RuntimeConfig` and its fingerprint. The pure R-multiple stop-state update
is shared with the frozen historical exit policy; secondary exit detail distinguishes initial,
breakeven, profit-lock, trailing, hard-target, and time outcomes while preserving `ExitReason`.
