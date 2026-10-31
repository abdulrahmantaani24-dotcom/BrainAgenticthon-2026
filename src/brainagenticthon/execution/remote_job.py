"""Remote-Colab-only medical operations; retain all medical artifacts on authorized Drive."""

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path

from brainagenticthon.detection.training import private_output_directory
from brainagenticthon.execution.disease_sources import (
    DEFAULT_LIMITS,
    SOURCE_NAMES,
    scan_source,
)
from brainagenticthon.execution.inventory import load_config, protected_inputs, save_private_report
from brainagenticthon.execution.real_training import execute_real, freeze_splits
from brainagenticthon.preprocessing.oasis2_mri_quality import private_path

AUTHORIZED_PARENT = Path("/content/drive/MyDrive/BrainMRI/datasets/DiseaseAcquisition")


def _assert_colab():
    if "google.colab" not in sys.modules:
        raise ValueError("medical_operations_require_Colab_runtime")


def _task_config(parent, name):
    if not name:
        return None
    path = private_path(parent, name)
    cfg, root = load_config(path)
    if not root.is_relative_to(parent):
        raise ValueError("task_project_outside_authorized_Drive_directory")
    for key in (
        "manifest",
        "reviews",
        "evidence",
        "training_evidence",
        "image_root",
        "freeze",
        "output_run",
    ):
        if cfg.get(key):
            private_path(root, cfg[key])
    # Existing gate validates authorization, source eligibility, identities and image hashes.
    return path, cfg, root


def run(*, task_configs=None, execute_approved=False):
    """No raw data or private report content is returned to the controller."""
    _assert_colab()
    parent = AUTHORIZED_PARENT
    if any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError("symlink_authorized_directory_rejected")
    if not parent.is_dir() or not os.access(parent, os.R_OK):
        return {
            "status": "BLOCKED",
            "reason": "Drive_mount_or_directory_authorization_required",
            "drive_accessed": False,
            "training_started": False,
            "clinical_approval": False,
        }
    sources = [private_path(parent, name) for name in SOURCE_NAMES]
    output = private_output_directory(
        parent / "_brainagenticthon_private_runs" / uuid.uuid4().hex, sources
    )
    output.mkdir(parents=True, exist_ok=False)
    inventory = {"schema": "Colab-only-private-inventory-v1", "sources": []}
    for name in SOURCE_NAMES:
        inventory["sources"].append(
            scan_source(
                parent,
                {
                    "directory": name,
                    "inspection_authorized": True,
                    "review_file": "task_ab_review.json",
                },
                DEFAULT_LIMITS,
            )
        )
    with (output / "inventory.private.json").open("x") as stream:
        json.dump(inventory, stream, indent=2, allow_nan=False)
    summary = {
        "status": "COMPLETE_WITH_BLOCKERS",
        "drive_accessed": True,
        "training_started": False,
        "clinical_approval": False,
        "sources": {
            row["source"]: {
                "inventory_status": row["status"],
                "task_a": row["eligibility"]["task_a"]["training_status"],
                "task_b": row["eligibility"]["task_b"]["training_status"],
            }
            for row in inventory["sources"]
        },
        "tasks": {},
    }
    for task in ("a", "b"):
        try:
            resolved = _task_config(parent, (task_configs or {}).get(task))
            if resolved is None:
                summary["tasks"][task] = {
                    "status": "BLOCKED",
                    "reason": "approved_private_task_configuration_missing",
                }
                continue
            path, cfg, root = resolved
            # All imaging/metadata roots must stay inside the single authorized parent.
            report = save_private_report(
                path, output / ("readiness_" + task + ".private.json"), task=task, scan=False
            )
            if report["status"] != "PASS":
                summary["tasks"][task] = {"status": "BLOCKED", "reason": "eligibility_gate_failed"}
                continue
            if not execute_approved or cfg.get("real_training_enabled") is not True:
                summary["tasks"][task] = {
                    "status": "READY_NOT_EXECUTED",
                    "reason": "explicit_real_execution_not_enabled",
                }
                continue
            # No approval flags or optimization settings are generated/changed here.
            frozen = private_path(root, cfg["freeze"])
            if "REPLACE_WITH" in cfg["freeze"] or "REPLACE_WITH" in cfg["output_run"]:
                raise ValueError("reviewed_run_and_freeze_destinations_required")
            if not frozen.exists():
                freeze_splits(path, frozen, task=task)
            # A separate one-batch smoke cannot overwrite the full experiment.
            smoke_cfg = copy.deepcopy(cfg)
            smoke_cfg["output_run"] = cfg["output_run"] + "-smoke-" + uuid.uuid4().hex
            smoke_path = private_output_directory(
                root / "_controller_configs" / (uuid.uuid4().hex + ".json"),
                protected_inputs(cfg, root, path),
            )
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            with smoke_path.open("x") as stream:
                json.dump(smoke_cfg, stream, allow_nan=False)
            execute_real(smoke_path, frozen, task=task, action="smoke")
            action = "resume" if cfg.get("resume") is True else "train"
            summary["training_started"] = None  # Not asserted if execution fails mid-epoch.
            result = execute_real(path, frozen, task=task, action=action)
            summary["training_started"] = True
            summary["tasks"][task] = {"status": result["status"], "action": action}
            if cfg.get("final_test_evaluation") is True:
                result = execute_real(path, frozen, task=task, action="evaluate")
                summary["tasks"][task]["final_evaluation"] = result["status"]
        except Exception:
            summary["tasks"][task] = {
                "status": "BLOCKED_OR_FAILED",
                "reason": "private_gate_or_execution_failure_see_remote_readiness",
            }
    with (output / "execution-summary.private.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    # Return only status, never counts, identities, metadata, metrics or report paths.
    return {
        "status": summary["status"],
        "drive_accessed": True,
        "training_started": summary["training_started"],
        "tasks": {k: v["status"] for k, v in summary["tasks"].items()},
        "clinical_approval": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--task-a-config", default="")
    parser.add_argument("--task-b-config", default="")
    args = parser.parse_args()
    try:
        import google.colab  # noqa: F401

        result = run(
            task_configs={"a": args.task_a_config, "b": args.task_b_config},
            execute_approved=args.execute_approved,
        )
    except Exception:
        result = {
            "status": "BLOCKED_OR_FAILED",
            "reason": "remote_setup_or_gate_failed",
            "drive_accessed": None,
            "training_started": None,
            "clinical_approval": False,
        }
    print("BRAIN_REMOTE_STATUS=" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
