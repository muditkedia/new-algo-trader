# Community Component Gate

## 1. Gate purpose

The Community Component Gate is an offline technical dependency check, not legal
advice. It reconciles direct declarations, the checked-in component inventory,
installed versions, safe imports or executables, project evidence, and locally
available package-license metadata. It neither chooses trading behavior nor upgrades
packages.

## 2. Runtime community components

The table records the environment validated for this architecture review. Installed
versions may change within the declared constraints; the gate reports the exact local
version and never queries for a newer release.

| Distribution | Category | Role | Installed version at gate run | Declared constraint | License/status | Project evidence |
|---|---|---|---:|---|---|---|
| APScheduler | Runtime community | Runtime per-date scheduling | 3.11.3 | `>=3.11.3,<4` | MIT | `src/algo_trader/runtime/scheduler.py` |
| duckdb | Runtime community | Market-data queries, OOS registry, Runtime persistence | 1.5.5 | `>=1,<2` | MIT | `src/algo_trader/data/market_data.py`; `src/algo_trader/oos/registry.py`; `src/algo_trader/runtime/state.py` |
| lightgbm | Runtime community | Trade Meta-Model classifier/regressor | 4.7.0 | `>=4.7,<5` | MIT | `src/algo_trader/ml/meta_model.py` |
| logzero | Runtime community | Official SmartAPI SDK support dependency | 1.7.0 | `>=1.7,<2` | MIT | `src/algo_trader/broker/angel_one.py` |
| matplotlib | Runtime community | Deterministic PNG reporting | 3.11.1 | `>=3.11,<4` | Matplotlib license | `src/algo_trader/reporting/outputs.py` |
| openpyxl | Runtime community | Excel derivative reports/native charts | 3.1.5 | `>=3.1,<4` | MIT | `src/algo_trader/reporting/outputs.py` |
| optuna | Runtime community | Sparse multi-objective Strategy Optimizer | 4.9.0 | `>=4.9,<5` | MIT | `src/algo_trader/ml/optimizer.py` |
| polars | Runtime community | Canonical in-memory/tabular data | 1.43.2 | `>=1,<2` | MIT | `src/algo_trader/data/market_data.py` |
| pydantic | Runtime community | Immutable validated models | 2.13.4 | `>=2,<3` | MIT | `src/algo_trader/domain.py` |
| pyarrow | Runtime community | Parquet/Arrow interoperability | 25.0.1 | `>=25,<26` | Apache-2.0 | `src/algo_trader/data/market_data.py` |
| pyotp | Runtime community | Deterministic SmartAPI TOTP | 2.10.0 | `>=2.10,<3` | MIT | `src/algo_trader/broker/angel_one.py` |
| python-dotenv | Runtime community | Explicit `SmartAPI.env` parsing | 1.2.2 | `>=1.2.2,<2` | BSD-3-Clause | `src/algo_trader/runtime/credentials.py` |
| scikit-learn | Runtime community | Probability calibration / LogisticRegression | 1.9.0 | `>=1.9,<2` | BSD-3-Clause | `src/algo_trader/ml/meta_model.py` |
| TA-Lib | Runtime community | Technical-indicator primitives | 0.7.1 | `>=0.7,<0.8` | BSD-2-Clause; local license file | `src/algo_trader/indicators/talib_indicators.py` |
| tzdata | Runtime community | ZoneInfo database fallback, especially on Windows | 2026.3 | `>=2025.2` | Apache-2.0 | `src/algo_trader/data/timing.py` |
| websocket-client | Runtime community | Official SmartAPI websocket dependency | 1.9.0 | `>=1.9,<2` | Apache-2.0 | `src/algo_trader/broker/market_data.py` |

License labels summarize installed metadata or an identified local license file; they
do not replace review of the applicable license text.

## 3. Angel One vendor SDK exception

| Distribution | Category | Role | Installed version at gate run | Declared constraint | License/status | Project evidence |
|---|---|---|---:|---|---|---|
| smartapi-python | Runtime vendor | Official Angel One Broker SDK boundary | 1.5.5 | `==1.5.5` | Vendor SDK / license not explicitly declared | `src/algo_trader/broker/angel_one.py`; `src/algo_trader/broker/market_data.py` |

`smartapi-python==1.5.5` is the required vendor integration. Its public source
availability does not itself establish an open-source license, and the project must
not infer redistribution rights from an unspecified license. External redistribution
or productization requires a separate review of the currently applicable Angel One
terms and rights before it occurs. The manifest records this prerequisite as
`REVIEW_REQUIRED_BEFORE_EXTERNAL_DISTRIBUTION`. This is an external legal/commercial
prerequisite, not an unresolved software-architecture defect. The accepted technical
integration and explicit vendor exception pass the technical gate without relabeling
the SDK as permissive open source; this statement is not legal advice.

## 4. Development tools

| Distribution | Category | Role | Installed version at gate run | Declared constraint | License/status | Project evidence |
|---|---|---|---:|---|---|---|
| pytest | Dev tool | Automated architecture/unit tests | 9.1.1 | `>=9,<10` | MIT | `tests/test_domain.py` |
| Ruff | Dev tool | Lint/import/style gate | 0.16.3 | `>=0.16,<0.17` | MIT | `pyproject.toml` |

These tools are declared only in `project.optional-dependencies.dev`, not in normal
production dependencies. Ruff is validated through its bounded `ruff --version`
command rather than an assumed import API.

## 5. Direct vs transitive dependency policy

The manifest contains owned direct runtime components and development tools only.
Ordinary transitives such as Requests, urllib3, certifi, charset-normalizer, idna, and
NumPy remain transitive unless project production code begins importing them as an
approved architectural dependency. An unexplained direct production import fails the
gate; an incidentally installed package does not become a direct dependency.

## 6. Conditional/reference-only components

QuantStats remains a reporting reference only. MLflow remains conditional on the
project's own artifact tracking proving insufficient. mlfinpy remains a selected
technique/reference only. None is installed or declared for the current architecture.

No core trading-engine dependency exists on VectorBT, Nautilus, LEAN, Backtrader,
Backtesting.py, bt, or OpenAlgo. The platform deliberately uses a thin custom
domain/execution architecture over small mature libraries.

## 7. Gate commands

From the project root, using the project environment:

```text
python -m pip check
python scripts/community_component_gate.py
pytest
ruff check .
```

The script is read-only, uses local metadata, performs no network request, reads no
secret or production-data file, authenticates to no broker, and writes no artifact.

## 8. Current gate result

Validated locally on 2026-08-14: 17 runtime components and two development tools
reconciled successfully. The report explicitly retained the SmartAPI vendor-license
exception and ended `COMMUNITY COMPONENT GATE: PASS`. `python -m pip check` also
reported no broken requirements.
