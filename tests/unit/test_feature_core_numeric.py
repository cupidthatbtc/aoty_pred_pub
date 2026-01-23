import pandas as pd

from aoty_pred.features.base import FeatureContext, FeatureOutput
from aoty_pred.features.core import CoreNumericBlock


def test_core_numeric_block_returns_output():
    df = pd.DataFrame({"Artist": ["a"], "Year": [2000]})
    ctx = FeatureContext(config={}, random_state=0)
    block = CoreNumericBlock({})
    out = block.fit_transform(df, ctx)
    assert isinstance(out, FeatureOutput)
    assert out.data.index.equals(df.index)
