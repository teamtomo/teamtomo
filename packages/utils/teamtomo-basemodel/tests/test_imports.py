"""Basic import tests for the teamtomo_basemodel package."""


def test_import_basemodel():
    """Test that the custom BaseModel class can be imported."""
    from teamtomo_basemodel import BaseModelTeamTomo

    # Ensure import not removed by auto-formatters
    _ = BaseModelTeamTomo


def test_import_typehints():
    """Test that type hints can be imported."""
    from teamtomo_basemodel import ExcludedDataFrame, ExcludedNumpyArray, ExcludedTensor

    # Ensure import not removed by auto-formatters
    _ = ExcludedTensor
    _ = ExcludedNumpyArray
    _ = ExcludedDataFrame
