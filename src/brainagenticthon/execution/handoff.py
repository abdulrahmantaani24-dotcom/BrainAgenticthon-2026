"""Optional research-only envelopes for backend/agent integration; no teammate code changes."""

import json


def research_envelope(payload, *, task, model_sha256):
    expected = {"a": "task-a-research-score-v1", "b": "task-b-brats2021-research-v1"}
    if (
        task not in expected
        or payload.get("schema") != expected[task]
        or payload.get("clinical_approval") is not False
        or len(model_sha256) != 64
        or any(c not in "0123456789abcdef" for c in model_sha256)
    ):
        raise ValueError("validated_research_payload_and_checkpoint_hash_required")
    compatible_key = "task_a_output_compatible" if task == "a" else "task_b_output_compatible"
    if payload.get(compatible_key) is not False:
        raise ValueError("unsupported_legacy_contract_must_remain_explicit")
    value = {
        "interface": "task-ab-research-envelope-v1",
        "task": task,
        "model_sha256": model_sha256,
        "clinical_approval": False,
        "TaskC_invoked": False,
        "payload": payload,
        "unsupported": ["clinical_diagnosis", "unverified_atlas_mapping", "legacy_task_export"],
    }
    json.dumps(value, allow_nan=False)
    return value
