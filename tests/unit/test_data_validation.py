import pandas as pd
import pytest

from aoty_pred.data.validation import validate_raw_schema


def test_validate_raw_schema_missing_columns():
    df = pd.DataFrame({"Artist": ["a"]})
    with pytest.raises(ValueError):
        validate_raw_schema(df)
