import pandas as pd

from aoty_pred.features.base import FeatureContext, FeatureOutput
from aoty_pred.features.descriptor_pca import DescriptorPCABlock


def test_descriptor_pca_block_returns_output():
    df = pd.DataFrame({"Artist": ["a"], "Year": [2000]})
    ctx = FeatureContext(config={}, random_state=0)
    block = DescriptorPCABlock({})
    out = block.fit_transform(df, ctx)
    assert isinstance(out, FeatureOutput)
    assert out.data.index.equals(df.index)
