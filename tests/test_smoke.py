"""Smoke test — verify the package is importable and version is set."""


def test_package_import():
    """Importing brainagenticthon should succeed and expose __version__."""
    import brainagenticthon

    assert hasattr(brainagenticthon, "__version__")
    assert isinstance(brainagenticthon.__version__, str)
    assert brainagenticthon.__version__ == "0.1.0"


def test_subpackage_imports():
    """All declared sub-packages should be importable."""
    import brainagenticthon.preprocessing
    import brainagenticthon.detection
    import brainagenticthon.spatial
    import brainagenticthon.classification
    import brainagenticthon.agents
    import brainagenticthon.verification
    import brainagenticthon.reporting

    # If we got here without ImportError, the structure is sound.
    assert True
