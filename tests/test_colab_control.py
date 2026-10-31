"""Controller protocol mocks and generated metadata only; never live authentication evidence."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from brainagenticthon.execution import remote_job

ROOT = Path(__file__).parents[1]


def controller():
    spec = importlib.util.spec_from_file_location(
        "colab_control", ROOT / "scripts/colab_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_state_outside_git_private_and_no_symlink(tmp_path):
    module = controller()
    state = module.secure_state(tmp_path / "private-state")
    assert state.stat().st_mode & 0o777 == 0o700
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git/HEAD").write_text("ref: refs/heads/synthetic\n")
    with pytest.raises(ValueError, match="outside_Git"):
        module.secure_state(repo / "state")
    link = tmp_path / "link"
    link.symlink_to(state, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.secure_state(link)


def test_auth_failure_does_not_allocate_or_claim_connection(tmp_path, monkeypatch):
    module = controller()

    def no_credentials(*_):
        raise RuntimeError("PRIVATE-SECRET-MUST-NOT-APPEAR")

    monkeypatch.setattr(module, "configured_state", no_credentials)
    report = module.control(tmp_path / "state", operation="inventory")
    assert report["status"] == "BLOCKED"
    assert not report["authenticated_backend_request_succeeded"]
    assert not report["remote_execution_verified"] and not report["drive_accessed"]
    assert not report["training_started"] and not report["medical_data_downloaded"]
    assert "PRIVATE-SECRET" not in json.dumps(report)
    assert not list((tmp_path / "state").glob("execution-code-*"))


def test_backend_success_is_not_remote_execution(tmp_path, monkeypatch):
    module = controller()
    calls = []
    client = types.SimpleNamespace(list_assignments=lambda: calls.append("server-list") or [])
    monkeypatch.setattr(module, "configured_state", lambda _: types.SimpleNamespace(client=client))
    result = module.control(tmp_path / "state")
    assert result["status"] == "AUTHENTICATED_BACKEND_ONLY"
    assert calls == ["server-list"] and not result["remote_execution_verified"]


def test_probe_verified_only_on_marker_and_own_runtime_released(tmp_path, monkeypatch):
    module = controller()
    calls = []
    client = types.SimpleNamespace(list_assignments=lambda: [])
    monkeypatch.setattr(module, "configured_state", lambda _: types.SimpleNamespace(client=client))
    session = types.ModuleType("colab_cli.commands.session")
    session.new = lambda **kw: calls.append(("new", kw))
    session.stop = lambda **kw: calls.append(("stop", kw))
    execution = types.ModuleType("colab_cli.commands.execution")

    def execute(**kw):
        assert kw["file"].endswith(".py")
        assert "BRAIN_REMOTE_STATUS=" in Path(kw["file"]).read_text()
        print(
            "BRAIN_REMOTE_STATUS="
            + json.dumps(
                {
                    "status": "REMOTE_EXECUTION_VERIFIED",
                    "colab_import_verified": True,
                    "drive_accessed": False,
                    "training_started": False,
                }
            )
        )

    execution.exec_command = execute
    package = types.ModuleType("colab_cli.commands")
    package.session, package.execution = session, execution
    monkeypatch.setitem(sys.modules, "colab_cli", types.ModuleType("colab_cli"))
    monkeypatch.setitem(sys.modules, "colab_cli.commands", package)
    result = module.control(tmp_path / "state", operation="probe")
    assert result["remote_execution_verified"] is True
    assert result["drive_accessed"] is False and result["allocated_runtime_released"] is True
    assert calls[0][1]["gpu"] == "T4"
    assert calls[0][1]["session"] == calls[1][1]["session"]


def test_status_parser_rejects_private_report_or_arbitrary_reason():
    module = controller()
    with pytest.raises(ValueError):
        module.status_from_output(
            "BRAIN_REMOTE_STATUS="
            + json.dumps({"status": "COMPLETE_WITH_BLOCKERS", "private_manifest": ["synthetic"]})
        )
    with pytest.raises(ValueError):
        module.status_from_output(
            "BRAIN_REMOTE_STATUS="
            + json.dumps({"status": "BLOCKED", "reason": "/private/patient/path"})
        )
    assert module.status_from_output("no evidence")["status"] == "BLOCKED"


def test_bootstrap_is_code_only_valid_python_no_download_or_auth_flow():
    module = controller()
    code = module.bootstrap("/content/brain-code-" + "a" * 32 + ".zip", "b" * 64)
    compile(code, "reviewed-bootstrap", "exec")
    assert "capture_output=True" in code and "code_manifest.json" in code
    assert "requirements-inventory.txt" in code
    assert "BRAIN_REMOTE_STATUS=" in code
    assert "drive.mount" not in code and "fetch_token" not in code and "download(" not in code
    with pytest.raises(ValueError):
        module.bootstrap("/content/private-data.zip", "b" * 64)
    with pytest.raises(ValueError):
        module.bootstrap(
            "/content/brain-code-" + "a" * 32 + ".zip", "b" * 64, task_a_config="../escape.json"
        )


def test_remote_medical_job_refuses_non_colab_host(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    with pytest.raises(ValueError, match="Colab_runtime"):
        remote_job.run()


def test_remote_drive_unavailable_is_blocked_not_fabricated_access(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    monkeypatch.setattr(remote_job, "AUTHORIZED_PARENT", tmp_path / "missing")
    result = remote_job.run()
    assert result["status"] == "BLOCKED" and not result["drive_accessed"]
    assert not result["training_started"]


def test_generated_colab_inventory_stays_in_authorized_parent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    parent = tmp_path / "authorized"
    parent.mkdir()
    for name in remote_job.SOURCE_NAMES:
        (parent / name).mkdir()
        (parent / name / "source_and_license.json").write_text('{"license":"pending"}')
    before = {p: p.read_bytes() for p in parent.rglob("*.json")}
    monkeypatch.setattr(remote_job, "AUTHORIZED_PARENT", parent)
    result = remote_job.run(execute_approved=True)
    assert result["drive_accessed"] is True and not result["training_started"]
    assert result["tasks"] == {"a": "BLOCKED", "b": "BLOCKED"}
    assert all(p.read_bytes() == value for p, value in before.items())
    assert (
        len(list((parent / "_brainagenticthon_private_runs").glob("*/inventory.private.json"))) == 1
    )
    assert "license" not in json.dumps(result) and str(parent) not in json.dumps(result)


def test_private_task_configuration_cannot_escape_authorized_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "authorized"
    parent.mkdir()
    path = parent / "task.json"
    path.write_text(json.dumps({"private_drive_root": str(outside)}))
    with pytest.raises(ValueError, match="outside_authorized"):
        remote_job._task_config(parent, "task.json")


def test_unexpected_remote_failure_does_not_assert_no_data_was_accessed(monkeypatch, capsys):
    google = types.ModuleType("google")
    google.colab = types.ModuleType("google.colab")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", google.colab)
    monkeypatch.setattr(sys, "argv", ["remote_job"])

    def unexpected(**_):
        raise RuntimeError("PRIVATE-DETAILS")

    monkeypatch.setattr(remote_job, "run", unexpected)
    remote_job.main()
    text = capsys.readouterr().out
    result = json.loads(text.split("BRAIN_REMOTE_STATUS=", 1)[1])
    assert result["drive_accessed"] is None and result["training_started"] is None
    assert "PRIVATE-DETAILS" not in text
