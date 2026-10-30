"""Resumable private OASIS-2 quality audit; MRI inputs are always read-only.

Only aggregate checkpoint/report JSON is written, in a dedicated output folder.
Committed prefixes are rehashed on resume without rerunning image validation.
An in-flight pair can repeat after a hard kill. No clinical labels are assigned.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from brainagenticthon.preprocessing import data_validator, manifest_schema, oasis2_clinical_policy
from brainagenticthon.preprocessing import oasis2_mri_quality as quality

AUDIT_VERSION = "1.0.0"
STATE_NAME = "checkpoint.json"
REPORT_NAME = "aggregate_report.json"
LOCK_NAME = ".audit.lock"
STATE_LIMIT = 2 * 1024 * 1024
FORMATS = {"AnalyzeImage", "Spm99AnalyzeImage", "Spm2AnalyzeImage", "Nifti1Pair", "Nifti2Pair"}
FLAG_NAMES = {
    "analyze_world_geometry_and_units_unverified",
    "native_t1_not_verified",
    "visual_qc_not_performed",
    "orientation.match",
    "singleton_fourth_axis_in_memory_view_only",
}
CATEGORIES = {
    "dimensions",
    "spacing",
    "affine",
    "voxel_values",
    "format_or_pair",
    "resource_limits",
    "existing_validator",
    "read_error",
    "other",
}


class AuditError(RuntimeError):
    """Static error code only: no source paths, identifiers or raw exception text."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stats():
    return {
        "computational_pass_pairs": 0,
        "computational_fail_pairs": 0,
        "byte_preservation_verified_pairs": 0,
        "singleton_view_pairs": 0,
        "failure_categories": {},
        "quality_flags": {},
        "shapes": {},
        "header_spacings_units_unverified": {},
        "orientations": {},
        "formats": {},
    }


def _plan(rows, root):
    pairs = defaultdict(set)
    file_visits = defaultdict(set)
    exclusions = Counter()
    accepted_records = 0
    associations = set()
    for row in rows:
        if Path(row.image_path).is_absolute() or ".." in Path(row.image_path).parts:
            exclusions["unsafe_path_records"] += 1
            continue
        if quality.hidden_metadata(row.image_path):
            exclusions["hidden_metadata_records"] += 1
            continue
        try:
            path = quality.private_path(root, row.image_path)
            if path.suffix not in {".hdr", ".img"}:
                exclusions["unsupported_extension_records"] += 1
                continue
            header = quality.private_path(root, str(path.with_suffix(".hdr").relative_to(root)))
            payload = quality.private_path(root, str(header.with_suffix(".img").relative_to(root)))
            pairs[header].add(row.visit_key)
            file_visits[header].add(row.visit_key)
            file_visits[payload].add(row.visit_key)
            accepted_records += 1
            associations.add((header, row.visit_key))
        except (ValueError, OSError):
            exclusions["unsafe_path_records"] += 1
    ambiguous = {p for p, keys in file_visits.items() if len(keys) > 1}
    headers = [
        p
        for p in pairs
        if p not in ambiguous
        and quality.private_path(root, str(p.with_suffix(".img").relative_to(root)))
        not in ambiguous
    ]
    exclusions["pairs_linked_to_multiple_visits"] = len(pairs) - len(headers)
    return sorted(headers), {
        "image_pair_records": len(rows),
        "subjects": len({r.subject_id for r in rows}),
        "clinical_visit_keys": len({r.visit_key for r in rows}),
        "planned_unique_pairs": len(headers),
        "duplicate_pair_or_companion_records": accepted_records - len(associations),
        "exclusions": dict(sorted(exclusions.items())),
    }


def _software():
    modules = (
        sys.modules[__name__],
        quality,
        data_validator,
        oasis2_clinical_policy,
        manifest_schema,
    )
    return {
        "audit_version": AUDIT_VERSION,
        "validator_version": quality.VALIDATOR_VERSION,
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in ("numpy", "nibabel")},
        "code_sha256": _sha([quality.digest_file(Path(m.__file__)) for m in modules]),
    }


def _output_safety(output, root, manifest):
    # A dedicated sibling directory is required, never an input ancestor or child.
    for protected in (root, manifest.parent, Path(__file__).resolve().parents[3]):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise AuditError("output_overlaps_inputs_or_code")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise AuditError("unsafe_output_directory")
    if output.exists():
        if not (output / STATE_NAME).exists() and any(
            path.name != LOCK_NAME for path in output.iterdir()
        ):
            raise AuditError("unowned_output_without_checkpoint")
        for path in output.iterdir():
            if path.name not in {STATE_NAME, REPORT_NAME, LOCK_NAME} and not (
                path.name.startswith(".audit-tmp-") and path.name.endswith(".json")
            ):
                raise AuditError("output_directory_contains_unrelated_files")
            _regular_output(path)
        if (output / REPORT_NAME).exists() and not (output / STATE_NAME).exists():
            raise AuditError("report_without_checkpoint")


def _regular_output(path):
    if path.is_symlink():
        raise AuditError("output_link_rejected")
    if path.exists():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AuditError("output_link_or_nonregular_file_rejected")


def _atomic_json(path, value):
    _regular_output(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".audit-tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _regular_output(path)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _writer_lock(output, recover_stale_lock):
    path = output / LOCK_NAME
    _regular_output(path)
    if recover_stale_lock:
        # Explicit operator attestation that the previous runtime has stopped.
        path.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise AuditError("writer_active_or_stale_lock_requires_review") from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(token)
        yield
    finally:
        if path.exists() and not path.is_symlink() and path.read_text() == token:
            path.unlink()


def _read_state(output, job_id, total):
    path = output / STATE_NAME
    if not path.exists():
        return None
    _regular_output(path)
    if path.stat().st_size > STATE_LIMIT:
        raise AuditError("checkpoint_exceeds_size_limit")
    state = json.loads(path.read_bytes())
    if not isinstance(state, dict) or set(state) != {
        "job_id",
        "next_index",
        "prefix_sha256",
        "stats",
        "state_sha256",
        "started_at_utc",
        "last_committed_at_utc",
    }:
        raise AuditError("invalid_checkpoint")
    checksum = state.pop("state_sha256")
    if checksum != _sha(state):
        raise AuditError("checkpoint_checksum_mismatch")
    if state["job_id"] != job_id:
        raise AuditError("manifest_code_root_or_parameters_changed_use_new_output")
    for key in ("started_at_utc", "last_committed_at_utc"):
        value = state[key]
        if not isinstance(value, str) or len(value) > 32:
            raise AuditError("invalid_checkpoint_time")
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise AuditError("invalid_checkpoint_time")
    cursor, stats = state["next_index"], state["stats"]
    if type(cursor) is not int or not 0 <= cursor <= total or set(stats) != set(_stats()):
        raise AuditError("invalid_checkpoint_counts")
    for key, value in stats.items():
        if isinstance(_stats()[key], dict):
            if not isinstance(value, dict) or any(
                type(v) is not int or v < 0 for v in value.values()
            ):
                raise AuditError("invalid_checkpoint_counters")
        elif type(value) is not int or not 0 <= value <= cursor:
            raise AuditError("invalid_checkpoint_counts")
    if stats["computational_pass_pairs"] + stats["computational_fail_pairs"] != cursor:
        raise AuditError("invalid_checkpoint_counts")
    if not set(stats["failure_categories"]) <= CATEGORIES:
        raise AuditError("invalid_checkpoint_categories")
    if not set(stats["quality_flags"]) <= FLAG_NAMES | {"other_review_warning"}:
        raise AuditError("invalid_checkpoint_categories")
    if not set(stats["formats"]) <= FORMATS | {"unavailable_or_unsupported"}:
        raise AuditError("invalid_checkpoint_categories")
    # Geometry counter keys must contain numeric lists or known axis letters.
    for key in ("shapes", "header_spacings_units_unverified"):
        for category in stats[key]:
            if category != "unavailable" and _numeric_key(json.loads(category)) != category:
                raise AuditError("invalid_checkpoint_geometry")
    if any(
        k != "unavailable" and (len(k) != 3 or set(k) - set("LRAPSI"))
        for k in stats["orientations"]
    ):
        raise AuditError("invalid_checkpoint_geometry")
    return state


def _save_state(output, state):
    _atomic_json(output / STATE_NAME, {**state, "state_sha256": _sha(state)})


def _signature(header, root, max_pair_bytes):
    """Bounded streaming hashes; no image arrays or raw paths in the result."""
    signature = []
    for suffix, limit in ((".hdr", 1024 * 1024), (".img", max_pair_bytes), (".mat", 0)):
        path = quality.private_path(root, str(header.with_suffix(suffix).relative_to(root)))
        if not path.exists():
            signature.append([suffix, "missing"])
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            signature.append([suffix, "nonregular"])
        elif info.st_size > limit or suffix == ".mat":
            # Rejected/oversized inputs are not read beyond resource limits.
            signature.append([suffix, "not_hashed", info.st_size, info.st_mtime_ns])
        else:
            signature.append([suffix, "sha256", quality.digest_file(path)])
    return signature


def _roll(prefix, signature):
    return _sha([prefix, signature])


def _failure_category(code):
    if "dimension" in code or "shape" in code or "singleton" in code:
        return "dimensions"
    if "spacing" in code or "voxel_size" in code:
        return "spacing"
    if any(word in code for word in ("affine", "qform", "sform", "spatial_transform")):
        return "affine"
    if code in {"nonfinite_voxels", "constant_volume", "voxel_dtype_not_real_numeric"}:
        return "voxel_values"
    if "limit" in code:
        return "resource_limits"
    if any(word in code for word in ("format", "companion", "sidecar", "payload", "file_map")):
        return "format_or_pair"
    if code.startswith(("header.", "dimensions.", "voxel_spacing.", "orientation.", "validator_")):
        return "existing_validator"
    if "read" in code or "unreadable" in code:
        return "read_error"
    return "other"


def _numeric_key(values):
    if (
        not isinstance(values, (list, tuple))
        or not 1 <= len(values) <= 4
        or any(
            v is not None and (type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e9)
            for v in values
        )
    ):
        return "unavailable"
    return _json(values)


def _bump(stats, key, category):
    stats[key][category] = stats[key].get(category, 0) + 1


def _accumulate(stats, outcome):
    passed = outcome.get("computational_checks_passed") is True
    stats["computational_pass_pairs" if passed else "computational_fail_pairs"] += 1
    stats["byte_preservation_verified_pairs"] += outcome.get("original_pair_unchanged") is True
    stats["singleton_view_pairs"] += outcome.get("singleton_axis_view") is True
    for category in {_failure_category(code) for code in outcome.get("failures", [])}:
        _bump(stats, "failure_categories", category)
    for flag in set(outcome.get("quality_flags", [])):
        _bump(stats, "quality_flags", flag if flag in FLAG_NAMES else "other_review_warning")
    _bump(stats, "shapes", _numeric_key(outcome.get("shape")))
    _bump(
        stats,
        "header_spacings_units_unverified",
        _numeric_key(outcome.get("header_spacing_units_unverified")),
    )
    axes = outcome.get("array_axis_codes")
    orientation = "".join(axes) if axes and all(isinstance(a, str) for a in axes) else "unavailable"
    if len(orientation) != 3 or set(orientation) - set("LRAPSI"):
        orientation = "unavailable"
    _bump(stats, "orientations", orientation)
    format_name = outcome.get("source_pair_format")
    _bump(stats, "formats", format_name if format_name in FORMATS else "unavailable_or_unsupported")


def _report(state, inventory, software, *, dry_run, max_pairs, prefix_rechecked):
    completed = state["next_index"]
    total = inventory["planned_unique_pairs"]
    blocked = any(
        count
        for name, count in inventory["exclusions"].items()
        if name != "hidden_metadata_records"
    )
    complete = bool(total) and completed == total and not blocked
    return {
        "audit_version": AUDIT_VERSION,
        "mode": "dry_run_no_mri_reads_or_writes" if dry_run else "read_only_mri_private_reports",
        "run_started_at_utc": state["started_at_utc"],
        "last_checkpoint_at_utc": state["last_committed_at_utc"],
        "report_generated_at_utc": _now(),
        "inventory": inventory,
        "completed_pairs": completed,
        "remaining_pairs": total - completed,
        "would_process_this_invocation": min(total - completed, max_pairs or total),
        "committed_prefix_rechecked": prefix_rechecked,
        "technical_audit_complete": complete and not dry_run,
        "all_cohort_computational_checks_passed": complete
        and not dry_run
        and state["stats"]["computational_fail_pairs"] == 0
        and state["stats"]["byte_preservation_verified_pairs"] == total,
        "results": copy.deepcopy(state["stats"]),
        "software": software,
        "visual_qc_approved": False,
        "clinical_diagnoses_approved": False,
        "final_task_labels_assigned": False,
        "task_a_or_c_release_ready": False,
        "limitations": [
            "Technical checks do not approve clinical image quality or diagnoses.",
            "Analyze geometry/native-T1 provenance/visual QC remain unverified.",
            "Hash preservation is not correctness against a trusted source checksum.",
            "Use a stable source and one Colab writer; Drive durability is external.",
        ],
    }


def _write_report(output, report):
    _atomic_json(output / REPORT_NAME, report)


def _run(
    manifest,
    image_root,
    output_dir,
    *,
    dry_run,
    max_pairs,
    max_voxels,
    max_pair_bytes,
    field_map,
    records_key,
    recover_stale_lock,
    progress,
):
    if type(dry_run) is not bool or type(recover_stale_lock) is not bool:
        raise AuditError("mode_and_lock_recovery_must_be_boolean")
    if progress is not None and not callable(progress):
        raise AuditError("progress_must_be_callable_or_none")
    if max_pairs is not None and (type(max_pairs) is not int or max_pairs <= 0):
        raise AuditError("max_pairs_must_be_positive_or_none")
    if type(max_voxels) is not int or not 1 <= max_voxels <= 32_000_000:
        raise AuditError("voxel_limit_must_not_exceed_reviewed_bound")
    if type(max_pair_bytes) is not int or not 1 <= max_pair_bytes <= 256 * 1024 * 1024:
        raise AuditError("byte_limit_must_not_exceed_reviewed_bound")
    manifest, root = Path(manifest).resolve(), Path(image_root).resolve()
    original_output = Path(output_dir).absolute()
    if original_output.is_symlink():
        raise AuditError("unsafe_output_directory")
    output = original_output.resolve()
    if not root.is_dir() or not manifest.is_file():
        raise AuditError("required_input_missing")
    _output_safety(output, root, manifest)
    rows, digest = quality.private_call(
        oasis2_clinical_policy.load_clinical_images,
        manifest,
        field_map=field_map,
        records_key=records_key,
    )
    if not rows:
        raise AuditError("empty_manifest")
    headers, inventory = _plan(rows, root)
    software = _software()
    job_id = _sha(
        {
            "manifest": digest,
            "root": str(root),
            "software": software,
            "pairs": [str(p.relative_to(root)) for p in headers],
            "max_voxels": max_voxels,
            "max_pair_bytes": max_pair_bytes,
            "field_map": field_map,
            "records_key": records_key,
            "inventory": inventory,
        }
    )
    initial = {
        "job_id": job_id,
        "next_index": 0,
        "prefix_sha256": _sha([job_id, "start"]),
        "stats": _stats(),
        "started_at_utc": _now(),
        "last_committed_at_utc": _now(),
    }
    if dry_run:
        state = _read_state(output, job_id, len(headers)) or initial
        return _report(
            state, inventory, software, dry_run=True, max_pairs=max_pairs, prefix_rechecked=False
        )
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output, recover_stale_lock):
        _output_safety(output, root, manifest)
        state = _read_state(output, job_id, len(headers)) or initial
        prefix = initial["prefix_sha256"]
        # Rehash only; completed image validation/voxel checks are never rerun.
        for header in headers[: state["next_index"]]:
            prefix = _roll(prefix, _signature(header, root, max_pair_bytes))
        if prefix != state["prefix_sha256"]:
            raise AuditError("completed_source_bytes_changed_use_new_output")
        if quality.digest_file(manifest) != digest:
            raise AuditError("manifest_changed_during_audit")
        _save_state(output, state)
        for temporary in output.glob(".audit-tmp-*.json"):
            _regular_output(temporary)
            temporary.unlink()
        report = _report(
            state, inventory, software, dry_run=False, max_pairs=max_pairs, prefix_rechecked=True
        )
        _write_report(output, report)  # Regenerate a stale/missing report from committed state.
        stop = min(len(headers), state["next_index"] + (max_pairs or len(headers)))
        for index in range(state["next_index"], stop):
            header = headers[index]
            before = _signature(header, root, max_pair_bytes)
            try:
                outcome = quality.private_call(
                    quality.validate_analyze_pair,
                    header,
                    root,
                    max_voxels=max_voxels,
                    max_pair_bytes=max_pair_bytes,
                )
            except RuntimeError:
                outcome = {
                    "computational_checks_passed": False,
                    "failures": ["unreadable_pair"],
                    "quality_flags": [],
                }
            if outcome.get("original_pair_unchanged") is False:
                raise AuditError("source_changed_during_pair_check")
            if quality.digest_file(manifest) != digest:
                raise AuditError("manifest_changed_during_audit")
            signature = _signature(header, root, max_pair_bytes)
            if signature != before:
                raise AuditError("source_changed_during_pair_check")
            updated = copy.deepcopy(state)
            updated["next_index"] = index + 1
            updated["last_committed_at_utc"] = _now()
            updated["prefix_sha256"] = _roll(state["prefix_sha256"], signature)
            _accumulate(updated["stats"], outcome)
            _save_state(output, updated)  # One atomic authoritative commit per pair.
            state = updated
            report = _report(
                state,
                inventory,
                software,
                dry_run=False,
                max_pairs=max_pairs,
                prefix_rechecked=True,
            )
            _write_report(output, report)
            if progress is not None:
                progress(copy.deepcopy(report))  # Aggregates only; no rows, paths or IDs.
            del outcome
        return report


def audit_cohort(
    manifest,
    image_root,
    output_dir,
    *,
    dry_run=True,
    max_pairs=None,
    max_voxels=32_000_000,
    max_pair_bytes=256 * 1024 * 1024,
    field_map=None,
    records_key="records",
    recover_stale_lock=False,
    progress=None,
):
    """Audit one pair at a time; reports/cursor are private, labels stay untouched.

    max_pairs caps NEW checks per invocation, not the cohort plan. Reusing the
    same output directory resumes its committed prefix. Changed source bytes,
    manifest, software or validation limits require a NEW output directory.
    """
    try:
        return _run(
            manifest,
            image_root,
            output_dir,
            dry_run=dry_run,
            max_pairs=max_pairs,
            max_voxels=max_voxels,
            max_pair_bytes=max_pair_bytes,
            field_map=field_map,
            records_key=records_key,
            recover_stale_lock=recover_stale_lock,
            progress=progress,
        )
    except AuditError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        raise AuditError("invalid_input_checkpoint_or_io_failure") from None


class _PrivateParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default may echo an invalid argument containing a private path.
        self.exit(2, "Invalid audit arguments; inspect the command privately.\n")


def main(argv=None):
    parser = _PrivateParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run", action="store_true", help="Write private progress; default is dry-run"
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--field-map", type=Path)
    parser.add_argument("--records-key", default="records")
    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="Use ONLY after confirming the previous writer/runtime has stopped",
    )
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.field_map.read_bytes()) if args.field_map else None
        report = audit_cohort(
            args.manifest,
            args.image_root,
            args.output_dir,
            dry_run=not args.run,
            max_pairs=args.max_pairs,
            field_map=mapping,
            records_key=args.records_key,
            recover_stale_lock=args.recover_stale_lock,
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if not args.run or report["all_cohort_computational_checks_passed"] else 1
    except KeyboardInterrupt:
        print(_json({"status": "interrupted", "resume_from_last_committed_checkpoint": True}))
        return 130
    except (AuditError, OSError, ValueError, TypeError):
        print(
            _json(
                {
                    "status": "audit_stopped",
                    "details": "Inspect inputs privately; no labels changed.",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
