"""Single safe entry: four-source inventory or explicitly gated existing A/B workflows."""

import json
from pathlib import Path

from brainagenticthon.detection.training import private_output_directory
from brainagenticthon.execution.disease_sources import inspect_sources
from brainagenticthon.execution.inventory import load_config, save_private_report
from brainagenticthon.execution.real_training import execute_real, freeze_splits
from brainagenticthon.preprocessing.oasis2_mri_quality import private_path


def run(config, output, *, action="inventory", task="a"):
    if action not in {"inventory", "validate", "freeze", "smoke", "train", "resume", "evaluate"}:
        raise ValueError("explicit_supported_action_required")
    if task not in {"a", "b"}:
        raise ValueError("explicit_task_required")
    for field in ("private_project_root", "dataset_parent"):
        candidate = Path(config[field]).absolute()
        if any(p.is_symlink() for p in (candidate, *candidate.parents)):
            raise ValueError("symlink_workflow_root_rejected")
    project = Path(config["private_project_root"]).resolve()
    parent = Path(config["dataset_parent"]).resolve()
    protected = [parent]
    task_configs = [
        private_path(project, value) for value in config.get("task_configs", {}).values() if value
    ]
    protected.extend(task_configs)
    output = private_output_directory(output, protected)
    if action == "inventory":
        return inspect_sources(config, output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "action": action,
        "task": task,
        "status": "BLOCKED",
        "clinical_approval": False,
        "training_started": False,
    }
    value = config.get("task_configs", {}).get(task)
    path = private_path(project, value) if value else None
    if path is None or not path.is_file():
        summary["blockers"] = ["approved_task_configuration_not_supplied"]
    else:
        cfg, root = load_config(path)
        report = save_private_report(
            path, output / ("readiness_" + task + ".json"), task=task, scan=False
        )
        if report["status"] != "PASS":
            summary["blockers"] = report["blockers"]
        elif action == "validate":
            summary["status"] = "PASS"
        else:
            try:
                frozen = private_path(root, cfg["freeze"])
                if "REPLACE_WITH" in cfg["freeze"]:
                    raise ValueError("configure_immutable_freeze_destination")
                if action == "freeze":
                    summary["result"] = freeze_splits(path, frozen, task=task)
                else:
                    summary["result"] = execute_real(path, frozen, task=task, action=action)
                    summary["training_started"] = action in {"smoke", "train", "resume"}
                summary["status"] = "PASS"
            except Exception as error:
                summary["blockers"] = ["explicit_action_gate_or_execution_failed"]
                summary["error_type"] = type(error).__name__
                summary["training_started"] = "not_asserted_on_failure; inspect_private_run"
    with (output / "session_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary
