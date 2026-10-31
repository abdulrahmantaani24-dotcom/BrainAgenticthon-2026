"""Versioned research input semantics; independent of Task B/C preprocessing."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InputProfile:
    """One externally verified native T1, full field of view, no lesion cropping.

    Header-based RAS permutation/flip is in memory only and is not independent
    anatomical verification. Fixed-grid interpolation changes voxel resolution;
    this classifier does not produce spatial predictions or physical measurements.
    """

    name: str = "task-a-native-t1-research-v1"
    target_shape: tuple[int, int, int] = (96, 112, 96)
    max_voxels: int = 32_000_000
    max_file_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if self.name != "task-a-native-t1-research-v1":
            raise ValueError("unsupported_input_profile")
        if (len(self.target_shape) != 3
                or any(type(d) is not int or not 16 <= d <= 192 for d in self.target_shape)):
            raise ValueError("invalid_target_shape")
        if (type(self.max_voxels) is not int or not 1 <= self.max_voxels <= 32_000_000
                or type(self.max_file_bytes) is not int
                or not 1 <= self.max_file_bytes <= 256 * 1024 * 1024):
            raise ValueError("invalid_resource_limits")

    def to_dict(self):
        result = asdict(self)
        result["target_shape"] = list(self.target_shape)
        return result
