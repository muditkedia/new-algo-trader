from __future__ import annotations

import socket
import tomllib
import urllib.request
from importlib import metadata
from pathlib import Path

import pytest

from scripts.community_component_gate import (
    DEFAULT_MANIFEST,
    EXPECTED_DEV,
    EXPECTED_RUNTIME,
    FORBIDDEN_DIRECT,
    ROOT,
    Component,
    LicenseEvidence,
    dependency_name,
    dev_dependency_specs,
    executable_version,
    license_is_acceptable,
    load_manifest,
    load_toml,
    main,
    normalize_distribution_name,
    project_dependency_specs,
    run_gate,
)


@pytest.fixture(scope="module")
def components() -> tuple[Component, ...]:
    return load_manifest(ROOT / DEFAULT_MANIFEST)


@pytest.fixture(scope="module")
def gate_report():
    return run_gate()


def test_manifest_contract_classification_and_normalized_uniqueness(
    components: tuple[Component, ...],
) -> None:
    names = [component.normalized_name for component in components]
    assert len(names) == len(set(names))
    assert normalize_distribution_name("smartapi_PYTHON") == "smartapi-python"
    runtime = {
        component.normalized_name
        for component in components
        if component.category != "DEV_TOOL"
    }
    dev = {
        component.normalized_name
        for component in components
        if component.category == "DEV_TOOL"
    }
    assert runtime == EXPECTED_RUNTIME
    assert dev == EXPECTED_DEV
    smartapi = next(
        component
        for component in components
        if component.normalized_name == "smartapi-python"
    )
    assert smartapi.category == "RUNTIME_VENDOR"
    assert smartapi.license_status == "VENDOR_LICENSE_NOT_EXPLICITLY_DECLARED"
    assert {
        component.normalized_name: component.category
        for component in components
        if component.normalized_name in EXPECTED_DEV
    } == {"pytest": "DEV_TOOL", "ruff": "DEV_TOOL"}


def test_pyproject_inventory_constraints_evidence_and_forbidden_policy(
    components: tuple[Component, ...],
) -> None:
    project = load_toml(ROOT / "pyproject.toml")
    runtime_specs = project_dependency_specs(project)
    dev_specs = dev_dependency_specs(project)
    runtime_names = {dependency_name(spec) for spec in runtime_specs}
    inventory_runtime = {
        component.normalized_name
        for component in components
        if component.category != "DEV_TOOL"
    }
    assert runtime_names == EXPECTED_RUNTIME == inventory_runtime
    assert {dependency_name(spec) for spec in dev_specs} == EXPECTED_DEV
    assert EXPECTED_DEV.isdisjoint(runtime_names)
    assert not (runtime_names & FORBIDDEN_DIRECT)
    assert [
        spec for spec in runtime_specs if dependency_name(spec) == "smartapi-python"
    ] == ["smartapi-python==1.5.5"]
    for component in components:
        assert component.project_evidence
        for evidence in component.project_evidence:
            assert (ROOT / evidence).is_file()


def test_configured_import_mapping_installation_and_ruff_executable(
    components: tuple[Component, ...], gate_report
) -> None:
    expected_imports = {
        "APScheduler": "apscheduler",
        "python-dotenv": "dotenv",
        "scikit-learn": "sklearn",
        "smartapi-python": "SmartApi",
        "TA-Lib": "talib",
        "websocket-client": "websocket",
    }
    configured = {
        component.distribution_name: component.import_name
        for component in components
        if component.distribution_name in expected_imports
    }
    assert configured == expected_imports
    assert gate_report.passed
    assert all(result.installed_version != "MISSING" for result in gate_report.component_results)
    assert all(result.usable and result.version_ok for result in gate_report.component_results)
    for component in components:
        assert metadata.version(component.distribution_name)
    ruff_ok, ruff_output = executable_version("ruff")
    assert ruff_ok
    assert ruff_output.startswith("ruff ")


def test_license_policy_requires_evidence_but_accepts_explicit_vendor_exception() -> None:
    unknown = Component(
        distribution_name="unknown-community",
        import_name="unknown_community",
        category="RUNTIME_COMMUNITY",
        role="fixture",
        project_evidence=("pyproject.toml",),
        license_status="UNKNOWN",
        license_metadata_note="fixture",
    )
    no_metadata = LicenseEvidence("NO_METADATA_FIELDS", False)
    assert not license_is_acceptable(unknown, no_metadata)

    smartapi = Component(
        distribution_name="smartapi-python",
        import_name="SmartApi",
        category="RUNTIME_VENDOR",
        role="fixture",
        project_evidence=("pyproject.toml",),
        license_status="VENDOR_LICENSE_NOT_EXPLICITLY_DECLARED",
        license_metadata_note="fixture",
    )
    assert license_is_acceptable(smartapi, no_metadata)
    rendered = run_gate().render()
    assert "VENDOR_LICENSE_NOT_EXPLICITLY_DECLARED" in rendered
    assert ".secrets" not in rendered


def test_gate_is_offline_avoids_secrets_and_data_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args, **kwargs):
        raise AssertionError("network access is prohibited")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", network_forbidden)
    access_log: list[Path] = []
    first = run_gate(access_log=access_log, importer=lambda _name: object()).render()
    second = run_gate(importer=lambda _name: object()).render()
    assert first == second
    relative_reads = [path.relative_to(ROOT) for path in access_log]
    assert all(parts.parts[:1] != (".secrets",) for parts in relative_reads)
    assert all(parts.parts[:1] != ("data",) for parts in relative_reads)
    assert ROOT / "pyproject.toml" in access_log
    assert ROOT / DEFAULT_MANIFEST in access_log


def test_cli_success_and_deliberately_missing_component_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([]) == 0
    success = capsys.readouterr().out
    assert success.rstrip().endswith("COMMUNITY COMPONENT GATE: PASS")

    manifest_text = (ROOT / DEFAULT_MANIFEST).read_text(encoding="utf-8")
    missing_manifest = tmp_path / "community_components.toml"
    missing_manifest.write_text(
        manifest_text.replace(
            'distribution_name = "APScheduler"',
            'distribution_name = "deliberately-missing-component"',
            1,
        ),
        encoding="utf-8",
    )
    assert main(["--manifest", str(missing_manifest)]) != 0
    failure = capsys.readouterr().out
    assert failure.rstrip().endswith("COMMUNITY COMPONENT GATE: FAIL")


def test_manifest_toml_parses_without_secret_or_data_configuration() -> None:
    with (ROOT / DEFAULT_MANIFEST).open("rb") as stream:
        raw = tomllib.load(stream)
    serialized = repr(raw)
    assert raw["manifest_version"] == 1
    assert ".secrets/SmartAPI.env" not in serialized
    assert all(
        not evidence.startswith("data/")
        for component in raw["component"]
        for evidence in component["project_evidence"]
    )
