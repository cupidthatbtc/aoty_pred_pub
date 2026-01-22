from aoty_pred.features.registry import build_default_registry, parse_feature_specs


def test_registry_builds_blocks():
    config = {"features": {"blocks": [{"name": "core_numeric", "params": {}}]}}
    registry = build_default_registry()
    specs = parse_feature_specs(config)
    blocks = registry.build_all(specs)
    assert blocks[0].name == "core_numeric"
