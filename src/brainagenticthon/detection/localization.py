"""Verified annotation targets for a FUTURE localizer, never classifier output.

No conversion from classification scores to TaskAOutput is provided. Annotation
boxes are ground truth, not model predictions, and must not enter evaluation as
predicted detections. Coordinates use the unchanged inclusive native-voxel schema.
"""

from dataclasses import dataclass

from brainagenticthon.preprocessing.task_output_schemas import BoundingBox3D


@dataclass(frozen=True)
class SpatialAnnotation:
    case_id: str
    dataset_source: str
    image_sha256: str
    native_shape: tuple[int, int, int]
    boxes: tuple[tuple[int, int, int, int, int, int], ...]
    provenance: str
    evidence_reference: str
    reviewer_reference: str
    review_status: str = "pending"
    coordinate_space: str = "native_voxel_inclusive"


def localization_target(record, annotation, *, image_sha256, native_shape):
    """Require real, reviewed spatial annotations; no image-wide fallback boxes."""
    if not isinstance(annotation, SpatialAnnotation):
        raise ValueError("genuine_spatial_annotations_required")
    if (annotation.review_status != "approved"
            or annotation.provenance not in {"challenge_annotation", "expert_annotation"}
            or not annotation.evidence_reference.strip()
            or not annotation.reviewer_reference.strip()
            or annotation.coordinate_space != "native_voxel_inclusive"
            or annotation.case_id != record.case_id
            or annotation.dataset_source != record.dataset_source
            or annotation.image_sha256 != image_sha256
            or tuple(annotation.native_shape) != tuple(native_shape)
            or len(image_sha256) != 64
            or any(c not in "0123456789abcdef" for c in image_sha256)):
        raise ValueError("spatial_annotation_not_verified_or_mismatched")
    if (len(annotation.native_shape) != 3
            or any(type(d) is not int or d <= 0 for d in annotation.native_shape)):
        raise ValueError("invalid_annotation_grid")
    if type(record.task_a_label) is not int or record.task_a_label not in (0, 1):
        raise ValueError("explicit_binary_label_required")
    if bool(annotation.boxes) != bool(record.task_a_label):
        raise ValueError("annotation_label_conflict_or_missing_boxes")
    for box in annotation.boxes:
        if len(box) != 6 or any(type(value) is not int for value in box):
            raise ValueError("invalid_annotation_box")
        BoundingBox3D(*box)  # Reuse bounds ordering; confidence is not a model score.
        if any(box[2 * axis + 1] >= size
               for axis, size in enumerate(annotation.native_shape)):
            raise ValueError("annotation_box_outside_native_grid")
    return annotation.boxes
