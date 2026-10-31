"""Agent-operated official Colab CLI adapter. ADC only; no credential prompts or data downloads."""

import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_VERSION = "0.7.5"


def secure_state(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("controller_state_symlink_rejected")
    if any(
        (p / ".git").is_file() or (p / ".git" / "HEAD").is_file() for p in (path, *path.parents)
    ):
        raise ValueError("controller_state_must_be_outside_Git")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def configured_state(directory):
    """Use official package constructors, isolating state without changing HOME/authentication."""
    if importlib.metadata.version("google-colab-cli") != CLI_VERSION:
        raise ValueError("reviewed_official_CLI_version_required")
    from colab_cli import auth, common
    from colab_cli.history import HistoryLogger
    from colab_cli.state import SettingsStore

    # CLI entry-point logging/settings are hardcoded to HOME in this release.
    # Invoke the official command handlers with protected constructor paths instead.
    common.state._settings_store = SettingsStore(str(directory / "settings.json"))
    common.state._history = HistoryLogger(str(directory / "history"))
    common.state.config_path = str(directory / "sessions.json")
    common.state.auth_provider = auth.AuthProvider.ADC
    return common.state


def bootstrap(remote_zip, digest, *, execute_approved=False, task_a_config="", task_b_config=""):
    if not re.fullmatch(r"/content/brain-code-[0-9a-f]{32}\.zip", remote_zip):
        raise ValueError("code_upload_path_rejected")
    if not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("bundle_sha256_required")
    for value in (task_a_config, task_b_config):
        if value and (Path(value).is_absolute() or ".." in Path(value).parts or "\\" in value):
            raise ValueError("private_config_must_be_relative_to_authorized_dataset_parent")
    args = ["--task-a-config", task_a_config, "--task-b-config", task_b_config]
    if execute_approved:
        args.append("--execute-approved")
    # Code/settings only. Medical operations stay in a fresh remote process.
    return f"""
import contextlib, hashlib, io, json, os, stat, subprocess, sys, zipfile
from pathlib import Path
try:
    if sys.version_info[:2] != (3, 12):
        raise ValueError('remote_python_3_12_required')
    archive_path = Path({remote_zip!r})
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != {digest!r}:
        raise ValueError('reviewed_bundle_integrity_failed')
    root = Path('/content') / ('brain-reviewed-' + {digest!r}[:16])
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(entries) > 128 or len(set(names)) != len(names)
                or sum(entry.file_size for entry in entries) > 20 * 1024 * 1024):
            raise ValueError('bundle_resource_limit')
        manifest = json.loads(archive.read('code_manifest.json'))
        if (manifest.get('code_only') is not True
                or set(names) != set(manifest['files']) | {{'code_manifest.json'}}):
            raise ValueError('bundle_allowlist_invalid')
        for entry in entries:
            name = Path(entry.filename)
            if (name.is_absolute() or '..' in name.parts or '\\\\' in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.file_size > 2 * 1024 * 1024):
                raise ValueError('unsafe_bundle_member')
            target = root / name
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise ValueError('unsafe_snapshot_destination')
            content = archive.read(entry.filename)
            if (entry.filename != 'code_manifest.json'
                    and hashlib.sha256(content).hexdigest() != manifest['files'][entry.filename]):
                raise ValueError('source_integrity_failed')
            if target.exists() and target.read_bytes() != content:
                raise ValueError('existing_snapshot_changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)
    for requirements in ('requirements-task-a.txt', 'requirements-inventory.txt'):
        command = [sys.executable, '-m', 'pip', 'install', '--quiet', '-r',
                   str(root / requirements)]
        subprocess.run(command,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src')
    command = [sys.executable, '-m', 'brainagenticthon.execution.remote_job', *{args!r}]
    result = subprocess.run(command,
                            env=environment, capture_output=True, text=True)
    # Never relay arbitrary remote stdout, traceback, data, plots or reports.
    parsed = None
    for line in result.stdout.splitlines():
        if line.startswith('BRAIN_REMOTE_STATUS='):
            parsed = json.loads(line.split('=', 1)[1])
    if parsed is None:
        raise ValueError('remote_job_did_not_return_sanitized_status')
    print('BRAIN_REMOTE_STATUS=' + json.dumps(parsed, allow_nan=False))
except Exception:
    print('BRAIN_REMOTE_STATUS=' + json.dumps({{'status':'BLOCKED_OR_FAILED',
        'reason':'remote_setup_or_execution_failed', 'drive_accessed':False,
        'training_started':False, 'clinical_approval':False}}))
"""


def probe_code():
    return """
import json, platform, sys
try:
    import google.colab
    print('BRAIN_REMOTE_STATUS=' + json.dumps({'status':'REMOTE_EXECUTION_VERIFIED',
        'python':platform.python_version(), 'colab_import_verified':True,
        'drive_accessed':False, 'training_started':False, 'clinical_approval':False}))
except Exception:
    print('BRAIN_REMOTE_STATUS=' + json.dumps({'status':'BLOCKED',
        'reason':'not_a_verified_Colab_runtime', 'drive_accessed':False, 'training_started':False}))
"""


def status_from_output(text):
    markers = [
        line.split("=", 1)[1]
        for line in text.splitlines()
        if line.startswith("BRAIN_REMOTE_STATUS=")
    ]
    if not markers:
        return {"status": "BLOCKED", "reason": "no_remote_execution_evidence"}
    value = json.loads(markers[-1])
    allowed = {
        "status",
        "reason",
        "python",
        "colab_import_verified",
        "drive_accessed",
        "training_started",
        "clinical_approval",
        "tasks",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("remote_output_contains_non_status_fields")
    if value.get("clinical_approval") not in (None, False):
        raise ValueError("clinical_approval_never_asserted_by_controller")
    if value.get("status") not in {
        "BLOCKED",
        "BLOCKED_OR_FAILED",
        "REMOTE_EXECUTION_VERIFIED",
        "COMPLETE_WITH_BLOCKERS",
    }:
        raise ValueError("unexpected_remote_status")
    if value.get("reason") not in {
        None,
        "not_a_verified_Colab_runtime",
        "remote_setup_or_execution_failed",
        "remote_setup_or_gate_failed",
        "Drive_mount_or_directory_authorization_required",
    }:
        raise ValueError("unexpected_remote_reason")
    if "tasks" in value and (
        not isinstance(value["tasks"], dict)
        or set(value["tasks"]) - {"a", "b"}
        or any(
            status not in {"PASS", "BLOCKED", "BLOCKED_OR_FAILED", "READY_NOT_EXECUTED"}
            for status in value["tasks"].values()
        )
    ):
        raise ValueError("non_status_task_output_rejected")
    return value


def control(
    state_directory,
    *,
    operation="preflight",
    execute_approved=False,
    task_a_config="",
    task_b_config="",
):
    if operation not in {"preflight", "probe", "inventory"}:
        raise ValueError("unsupported_remote_operation")
    directory = secure_state(state_directory)
    previous_umask = os.umask(0o077)
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)  # No credential-bearing request/response debugging.
    report = {
        "status": "BLOCKED",
        "official_cli_version": CLI_VERSION,
        "authenticated_backend_request_succeeded": False,
        "remote_execution_verified": False,
        "drive_accessed": False,
        "training_started": False,
        "medical_data_downloaded": False,
    }
    created = None
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            state = configured_state(directory)
            # Do not trust `sessions` returning an empty list: CLI can swallow auth failures.
            state.client.list_assignments()
            report["authenticated_backend_request_succeeded"] = True
            report["status"] = "AUTHENTICATED_BACKEND_ONLY"
            if operation != "preflight":
                from colab_cli.commands import execution, session

                name = "brainagent-" + uuid.uuid4().hex[:12]
                session.new(session=name, gpu="T4", tpu=None, high_mem=False)
                created = name

                def execute_script(code, timeout):
                    script = directory / ("execution-code-" + uuid.uuid4().hex + ".py")
                    script.write_text(code)
                    capture.seek(0)
                    capture.truncate(0)
                    execution.exec_command(
                        session=name, file=str(script), output_image=None, timeout=timeout, env=None
                    )
                    return status_from_output(capture.getvalue())

                remote_status = execute_script(probe_code(), 60.0)
                if remote_status.get("status") != "REMOTE_EXECUTION_VERIFIED":
                    raise ValueError("Colab_kernel_probe_not_verified")
                report["remote_execution_verified"] = True
                report["status"] = "REMOTE_EXECUTION_VERIFIED"
                if operation == "inventory":
                    from colab_cli.contents import ContentsClient

                    spec = importlib.util.spec_from_file_location(
                        "bundle_builder", ROOT / "scripts/build_task_a_bundle.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    with tempfile.TemporaryDirectory(dir=directory) as temporary:
                        archive = Path(temporary) / "code-only.zip"
                        digest = module.build_bundle(archive)
                        remote = "/content/brain-code-" + uuid.uuid4().hex + ".zip"
                        ContentsClient(state.get_session(name)).upload(str(archive), remote)
                    remote_status = execute_script(
                        bootstrap(
                            remote,
                            digest,
                            execute_approved=execute_approved,
                            task_a_config=task_a_config,
                            task_b_config=task_b_config,
                        ),
                        3600.0,
                    )
                    report.update(
                        status=remote_status["status"],
                        drive_accessed=remote_status.get("drive_accessed", False),
                        training_started=remote_status.get("training_started", False),
                    )
                    if remote_status.get("reason"):
                        report["reason"] = remote_status["reason"]
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, GeneratorExit)):
            raise
        report.update(
            status="BLOCKED",
            error_type=type(error).__name__,
            reason="protected_ADC_or_remote_service_unavailable",
        )
    finally:
        if created:
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    from colab_cli.commands import session

                    session.stop(session=created)
                report["allocated_runtime_released"] = True
            except BaseException:
                report["allocated_runtime_released"] = False
        with (directory / ("control-status-" + uuid.uuid4().hex + ".json")).open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    logging.disable(previous_logging)
    os.umask(previous_umask)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument(
        "--operation", choices=("preflight", "probe", "inventory"), default="preflight"
    )
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--task-a-config", default="")
    parser.add_argument("--task-b-config", default="")
    args = parser.parse_args()
    result = control(
        args.state_directory,
        operation=args.operation,
        execute_approved=args.execute_approved,
        task_a_config=args.task_a_config,
        task_b_config=args.task_b_config,
    )
    print(json.dumps(result))
    return 0 if result["authenticated_backend_request_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
