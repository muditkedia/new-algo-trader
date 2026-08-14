"""Offline technical gate for New Algo Trader's direct third-party components."""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("config/community_components.toml")
CATEGORIES = {"RUNTIME_COMMUNITY", "RUNTIME_VENDOR", "DEV_TOOL"}
EXPECTED_RUNTIME = frozenset(
    {
        "apscheduler",
        "duckdb",
        "lightgbm",
        "logzero",
        "matplotlib",
        "openpyxl",
        "optuna",
        "polars",
        "pydantic",
        "pyarrow",
        "pyotp",
        "python-dotenv",
        "scikit-learn",
        "smartapi-python",
        "ta-lib",
        "tzdata",
        "websocket-client",
    }
)
EXPECTED_DEV = frozenset({"pytest", "ruff"})
FORBIDDEN_DIRECT = frozenset(
    normalize
    for normalize in (
        "quantstats",
        "mlflow",
        "mlfinpy",
        "vectorbt",
        "backtrader",
        "backtesting",
        "bt",
        "nautilus-trader",
        "openalgo",
        "xgboost",
        "tensorflow",
        "torch",
        "keras",
        "ray",
        "dask",
        "redis",
        "celery",
        "kafka",
    )
)
NONTERMINAL_LICENSE_STATUSES = {"", "UNKNOWN", "UNREVIEWED", "MISSING"}


@dataclass(frozen=True, slots=True)
class Component:
    distribution_name: str
    category: str
    role: str
    project_evidence: tuple[str, ...]
    license_status: str
    license_metadata_note: str
    import_name: str | None = None
    executable_name: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_distribution_name(self.distribution_name)


@dataclass(frozen=True, slots=True)
class LicenseEvidence:
    metadata_summary: str
    local_license_file: bool


@dataclass(frozen=True, slots=True)
class ComponentResult:
    component: Component
    installed_version: str
    declared: bool
    usable: bool
    license_ok: bool
    evidence_ok: bool
    version_ok: bool
    details: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(
            (
                self.declared,
                self.usable,
                self.license_ok,
                self.evidence_ok,
                self.version_ok,
            )
        )


@dataclass(frozen=True, slots=True)
class GateReport:
    component_results: tuple[ComponentResult, ...]
    global_issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.global_issues and all(
            result.passed for result in self.component_results
        )

    def render(self) -> str:
        lines = [
            "New Algo Trader Community Component Gate",
            "distribution | installed | category | declared | usable | license | "
            "evidence | version | result",
        ]
        for result in self.component_results:
            lines.append(
                " | ".join(
                    (
                        result.component.distribution_name,
                        result.installed_version,
                        result.component.category,
                        _mark(result.declared),
                        _mark(result.usable),
                        result.component.license_status
                        if result.license_ok
                        else "FAIL",
                        _mark(result.evidence_ok),
                        _mark(result.version_ok),
                        "PASS" if result.passed else "FAIL",
                    )
                )
            )
            for detail in result.details:
                lines.append(f"  {result.component.distribution_name}: {detail}")
        for issue in self.global_issues:
            lines.append(f"CHECK: {issue} | FAIL")
        if any(
            result.component.normalized_name == "smartapi-python"
            for result in self.component_results
        ):
            lines.append(
                "SmartAPI vendor exception: VENDOR_LICENSE_NOT_EXPLICITLY_DECLARED"
            )
        lines.append(
            f"COMMUNITY COMPONENT GATE: {'PASS' if self.passed else 'FAIL'}"
        )
        return "\n".join(lines)


def normalize_distribution_name(name: str) -> str:
    """Apply the PyPA case/hyphen/underscore normalization used by this gate."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def dependency_name(specification: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", specification)
    if match is None:
        raise ValueError(f"invalid dependency specification: {specification!r}")
    return normalize_distribution_name(match.group(1))


def load_toml(path: Path, access_log: list[Path] | None = None) -> dict[str, object]:
    if access_log is not None:
        access_log.append(path.resolve())
    with path.open("rb") as stream:
        return tomllib.load(stream)


def load_manifest(
    path: Path, access_log: list[Path] | None = None
) -> tuple[Component, ...]:
    raw = load_toml(path, access_log)
    if raw.get("manifest_version") != 1:
        raise ValueError("community component manifest_version must be 1")
    rows = raw.get("component")
    if not isinstance(rows, list):
        raise ValueError("community component manifest requires [[component]] rows")
    components = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("component row must be a TOML table")
        evidence = row.get("project_evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(value, str) and value for value in evidence
        ):
            raise ValueError("component project_evidence must be a string array")
        component = Component(
            distribution_name=_required_string(row, "distribution_name"),
            import_name=_optional_string(row, "import_name"),
            executable_name=_optional_string(row, "executable_name"),
            category=_required_string(row, "category"),
            role=_required_string(row, "role"),
            project_evidence=tuple(evidence),
            license_status=_required_string(row, "license_status"),
            license_metadata_note=_required_string(row, "license_metadata_note"),
        )
        if component.category not in CATEGORIES:
            raise ValueError(f"invalid component category: {component.category}")
        if (component.import_name is None) == (component.executable_name is None):
            raise ValueError("component requires exactly one import_name or executable_name")
        components.append(component)
    return tuple(components)


def project_dependency_specs(project: dict[str, object]) -> tuple[str, ...]:
    project_table = project.get("project")
    if not isinstance(project_table, dict):
        raise ValueError("pyproject.toml requires [project]")
    dependencies = project_table.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) for value in dependencies
    ):
        raise ValueError("project.dependencies must be a string array")
    return tuple(dependencies)


def dev_dependency_specs(project: dict[str, object]) -> tuple[str, ...]:
    project_table = project.get("project")
    optional = (
        project_table.get("optional-dependencies")
        if isinstance(project_table, dict)
        else None
    )
    dev = optional.get("dev") if isinstance(optional, dict) else None
    if not isinstance(dev, list) or not all(isinstance(value, str) for value in dev):
        return ()
    return tuple(dev)


def inspect_license(distribution: metadata.Distribution) -> LicenseEvidence:
    values = []
    for key in ("License-Expression", "License"):
        value = distribution.metadata.get(key)
        if value:
            values.append(str(value))
    values.extend(
        value
        for value in (distribution.metadata.get_all("Classifier") or ())
        if str(value).startswith("License ::")
    )
    combined = " ".join(values).lower()
    if "matplotlib" in combined or "python software foundation" in combined:
        summary = "MATPLOTLIB/PSF"
    elif "mit" in combined:
        summary = "MIT"
    elif "apache" in combined:
        summary = "APACHE"
    elif "bsd" in combined:
        summary = "BSD"
    elif combined:
        summary = "OTHER_DECLARED"
    else:
        summary = "NO_METADATA_FIELDS"
    local_license = any(
        Path(str(item)).name.lower().startswith(("license", "licence", "copying"))
        for item in (distribution.files or ())
    )
    return LicenseEvidence(summary, local_license)


def license_is_acceptable(
    component: Component, evidence: LicenseEvidence
) -> bool:
    if component.category == "RUNTIME_VENDOR":
        return (
            component.normalized_name == "smartapi-python"
            and component.license_status == "VENDOR_LICENSE_NOT_EXPLICITLY_DECLARED"
        )
    if component.license_status.upper() in NONTERMINAL_LICENSE_STATUSES:
        return False
    status = component.license_status.upper()
    expected_metadata = {
        "MIT": "MIT",
        "APACHE-2.0": "APACHE",
        "BSD-2-CLAUSE": "BSD",
        "BSD-3-CLAUSE": "BSD",
        "MATPLOTLIB-LICENSE": "MATPLOTLIB/PSF",
    }
    expected = expected_metadata.get(status)
    if expected is None:
        return False
    if evidence.metadata_summary == expected:
        return True
    return evidence.metadata_summary == "NO_METADATA_FIELDS" and evidence.local_license_file


def executable_version(executable_name: str) -> tuple[bool, str]:
    executable = shutil.which(executable_name)
    if executable is None:
        suffix = ".exe" if sys.platform == "win32" else ""
        adjacent = Path(sys.executable).resolve().parent / f"{executable_name}{suffix}"
        executable = str(adjacent) if adjacent.is_file() else None
    if executable is None:
        return False, "executable not found"
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, type(error).__name__
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return completed.returncode == 0, output[0] if output else "no version output"


def scan_production_imports(root: Path, access_log: list[Path] | None = None) -> set[str]:
    imports: set[str] = set()
    source_root = root / "src" / "algo_trader"
    for path in sorted(source_root.rglob("*.py")):
        if access_log is not None:
            access_log.append(path.resolve())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".", 1)[0])
    return {
        name
        for name in imports
        if name not in sys.stdlib_module_names and name != "algo_trader"
    }


def run_gate(
    root: Path = ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    access_log: list[Path] | None = None,
    importer: Callable[[str], object] = importlib.import_module,
    distribution_provider: Callable[[str], metadata.Distribution] = metadata.distribution,
) -> GateReport:
    root = root.resolve()
    manifest_file = manifest_path if manifest_path.is_absolute() else root / manifest_path
    project = load_toml(root / "pyproject.toml", access_log)
    components = load_manifest(manifest_file, access_log)
    runtime_specs = project_dependency_specs(project)
    dev_specs = dev_dependency_specs(project)
    runtime_names = tuple(dependency_name(spec) for spec in runtime_specs)
    dev_names = tuple(dependency_name(spec) for spec in dev_specs)
    inventory_names = tuple(component.normalized_name for component in components)
    inventory_runtime = {
        component.normalized_name
        for component in components
        if component.category != "DEV_TOOL"
    }
    inventory_dev = {
        component.normalized_name
        for component in components
        if component.category == "DEV_TOOL"
    }
    issues = []
    _require_unique(runtime_names, "project runtime dependencies", issues)
    _require_unique(dev_names, "project dev dependencies", issues)
    _require_unique(inventory_names, "component manifest", issues)
    if set(runtime_names) != EXPECTED_RUNTIME:
        issues.append("project runtime dependency set differs from approved inventory")
    if inventory_runtime != EXPECTED_RUNTIME:
        issues.append("manifest runtime component set differs from approved inventory")
    if set(dev_names) != EXPECTED_DEV or inventory_dev != EXPECTED_DEV:
        issues.append("pytest/Ruff dev-tool declarations are incomplete or drifted")
    if set(runtime_names) != inventory_runtime:
        issues.append("pyproject runtime dependencies and manifest do not reconcile")
    if EXPECTED_DEV & set(runtime_names):
        issues.append("pytest/Ruff must not be production runtime dependencies")
    forbidden = set(runtime_names) & FORBIDDEN_DIRECT
    if forbidden:
        issues.append("forbidden direct dependencies: " + ",".join(sorted(forbidden)))
    smart_specs = [
        spec for spec in runtime_specs if dependency_name(spec) == "smartapi-python"
    ]
    if smart_specs != ["smartapi-python==1.5.5"]:
        issues.append("smartapi-python must remain exactly pinned to 1.5.5")

    specifications = {
        dependency_name(spec): spec for spec in (*runtime_specs, *dev_specs)
    }
    component_results = []
    for component in sorted(components, key=lambda item: item.normalized_name):
        details = []
        declared = component.normalized_name in specifications
        evidence_ok = all(
            _safe_evidence_exists(root, value) for value in component.project_evidence
        )
        if not evidence_ok:
            details.append("project evidence is missing or outside repository")
        installed_version = "MISSING"
        usable = False
        license_ok = False
        version_ok = False
        try:
            distribution = distribution_provider(component.distribution_name)
            installed_version = distribution.version
            license_evidence = inspect_license(distribution)
            license_ok = license_is_acceptable(component, license_evidence)
            if not license_ok:
                details.append("license/status evidence is insufficient")
            version_ok = declared and version_satisfies(
                installed_version, specifications[component.normalized_name]
            )
            if not version_ok:
                details.append("installed version does not satisfy declared constraint")
            if component.import_name is not None:
                try:
                    importer(component.import_name)
                    usable = True
                except Exception as error:
                    details.append(f"import failed: {type(error).__name__}")
            else:
                usable, executable_detail = executable_version(
                    component.executable_name or ""
                )
                if not usable:
                    details.append(f"executable failed: {executable_detail}")
        except metadata.PackageNotFoundError:
            details.append("distribution is not installed")
        component_results.append(
            ComponentResult(
                component=component,
                installed_version=installed_version,
                declared=declared,
                usable=usable,
                license_ok=license_ok,
                evidence_ok=evidence_ok,
                version_ok=version_ok,
                details=tuple(details),
            )
        )

    approved_imports = {
        (component.import_name or "").split(".", 1)[0]
        for component in components
        if component.import_name
    }
    production_imports = scan_production_imports(root, access_log)
    unexplained_imports = production_imports - approved_imports
    if unexplained_imports:
        issues.append(
            "undeclared production imports: " + ",".join(sorted(unexplained_imports))
        )
    forbidden_imports = {
        name
        for name in production_imports
        if normalize_distribution_name(name) in FORBIDDEN_DIRECT
    }
    if forbidden_imports:
        issues.append("forbidden production imports: " + ",".join(sorted(forbidden_imports)))
    return GateReport(tuple(component_results), tuple(sorted(issues)))


def version_satisfies(installed: str, specification: str) -> bool:
    clauses = re.findall(r"(==|>=|<=|<|>)\s*([0-9]+(?:\.[0-9]+)*)", specification)
    value = _version_tuple(installed)
    for operator, required_text in clauses:
        required = _version_tuple(required_text)
        width = max(len(value), len(required))
        left = value + (0,) * (width - len(value))
        right = required + (0,) * (width - len(required))
        checks = {
            "==": left == right,
            ">=": left >= right,
            "<=": left <= right,
            "<": left < right,
            ">": left > right,
        }
        if not checks[operator]:
            return False
    return bool(clauses)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args(argv)
    try:
        report = run_gate(arguments.root, arguments.manifest)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"COMMUNITY COMPONENT GATE: FAIL\nCHECK: {error} | FAIL")
        return 1
    print(report.render())
    return 0 if report.passed else 1


def _required_string(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"component {key} must be a non-empty string")
    return value.strip()


def _optional_string(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"component {key} must be a non-empty string when supplied")
    return value.strip()


def _safe_evidence_exists(root: Path, relative: str) -> bool:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate.is_file()


def _require_unique(values: Sequence[str], context: str, issues: list[str]) -> None:
    if len(values) != len(set(values)):
        issues.append(f"duplicate normalized distribution name in {context}")


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"\s*([0-9]+(?:\.[0-9]+)*)", value)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


if __name__ == "__main__":
    raise SystemExit(main())
