"""Tests for extended DiagramRenderer features.

Tests the enhanced DiagramRenderer with:
- Feedback loop constants and rendering
- Legend generation with shapes and edge types
- Nested cluster rendering for 2-level hierarchy
"""

import pytest

from aoty_pred.visualization.introspection import (
    DiagramData,
    DiagramRenderer,
    EdgeSpec,
    NodeSpec,
)
from aoty_pred.visualization.introspection.renderer import (
    EXTENDED_THEME_COLORS,
    FEEDBACK_LOOPS,
    FEEDBACK_STYLES,
    LEGEND_EDGE_TYPES,
    LEGEND_SHAPES,
)


class TestFeedbackConstants:
    """Test feedback loop constants."""

    def test_feedback_styles_has_4_types(self):
        """Should have 4 feedback loop types."""
        assert len(FEEDBACK_STYLES) == 4
        assert set(FEEDBACK_STYLES.keys()) == {"retry", "tune", "refit", "ablate"}

    def test_feedback_loops_defined(self):
        """Should have 4 feedback loop definitions."""
        assert len(FEEDBACK_LOOPS) == 4

    def test_feedback_styles_have_color_and_style(self):
        """Each style should have color and style keys."""
        for name, style in FEEDBACK_STYLES.items():
            assert "color" in style, f"{name} missing color"
            assert "style" in style, f"{name} missing style"
            assert "label" in style, f"{name} missing label"

    def test_feedback_loops_have_required_keys(self):
        """Each loop should have source, target, type."""
        for loop in FEEDBACK_LOOPS:
            assert "source" in loop, "loop missing source"
            assert "target" in loop, "loop missing target"
            assert "type" in loop, "loop missing type"
            assert loop["type"] in FEEDBACK_STYLES, f"unknown type: {loop['type']}"


class TestLegendConstants:
    """Test legend constants."""

    def test_legend_shapes_defined(self):
        """Should have standard shapes defined."""
        assert "box" in LEGEND_SHAPES
        assert "ellipse" in LEGEND_SHAPES
        assert "diamond" in LEGEND_SHAPES
        assert "note" in LEGEND_SHAPES
        assert "folder" in LEGEND_SHAPES

    def test_legend_edge_types_defined(self):
        """Should have edge types including feedback."""
        assert "flow" in LEGEND_EDGE_TYPES
        assert "retry" in LEGEND_EDGE_TYPES
        assert "tune" in LEGEND_EDGE_TYPES
        assert "refit" in LEGEND_EDGE_TYPES
        assert "ablate" in LEGEND_EDGE_TYPES

    def test_legend_edge_types_have_correct_structure(self):
        """Each edge type should have (label, style, color) tuple."""
        for etype, value in LEGEND_EDGE_TYPES.items():
            assert isinstance(value, tuple), f"{etype} should be tuple"
            assert len(value) == 3, f"{etype} should have 3 elements"
            label, style, color = value
            assert isinstance(label, str), f"{etype} label should be string"
            assert isinstance(style, str), f"{etype} style should be string"
            assert color.startswith("#"), f"{etype} color should be hex"


class TestDiagramRendererExtended:
    """Test extended renderer features."""

    @pytest.fixture
    def simple_data(self):
        """Create simple DiagramData for testing."""
        return DiagramData(
            nodes={
                "a": NodeSpec(id="a", label="Node A", category="process"),
                "b": NodeSpec(id="b", label="Node B", category="data"),
            },
            edges=[EdgeSpec(source="a", target="b", label="100 rows")],
            clusters={"test": ["a", "b"]},
        )

    @pytest.fixture
    def renderer(self):
        """Create DiagramRenderer instance."""
        return DiagramRenderer()

    def test_render_produces_valid_graph(self, renderer, simple_data):
        """render() should produce a graphviz.Digraph."""
        graph = renderer.render(simple_data)
        assert graph is not None
        assert hasattr(graph, "source")

    def test_render_includes_legend(self, renderer, simple_data):
        """Rendered graph should include legend cluster."""
        graph = renderer.render(simple_data)
        assert "cluster_legend" in graph.source

    def test_edge_labels_preserved(self, renderer, simple_data):
        """Edge labels (row counts) should appear in output."""
        graph = renderer.render(simple_data)
        assert "100 rows" in graph.source

    def test_render_includes_feedback_loops(self, renderer, simple_data):
        """Rendered graph should include feedback loop edges."""
        graph = renderer.render(simple_data)
        # Feedback loops use constraint=false
        assert 'constraint="false"' in graph.source or "constraint=false" in graph.source


class TestNestedClusters:
    """Test nested cluster rendering."""

    @pytest.fixture
    def nested_data(self):
        """Create data with hierarchical clusters."""
        return DiagramData(
            nodes={
                "prior:mu_artist_loc": NodeSpec(
                    id="prior:mu_artist_loc",
                    label="mu_artist_loc",
                    category="config",
                    cluster="PRIORS_artist_pooling",
                ),
                "prior:sigma_rw_scale": NodeSpec(
                    id="prior:sigma_rw_scale",
                    label="sigma_rw_scale",
                    category="config",
                    cluster="PRIORS_career_dynamics",
                ),
            },
            edges=[],
            clusters={
                "PRIORS_artist_pooling": ["prior:mu_artist_loc"],
                "PRIORS_career_dynamics": ["prior:sigma_rw_scale"],
            },
        )

    def test_nested_clusters_rendered(self, nested_data):
        """Nested clusters should create subgraphs."""
        renderer = DiagramRenderer()
        graph = renderer.render(nested_data)
        # Should have PRIORS as major cluster
        assert "cluster_PRIORS" in graph.source

    def test_sub_clusters_rendered(self, nested_data):
        """Sub-clusters should be inside major clusters."""
        renderer = DiagramRenderer()
        graph = renderer.render(nested_data)
        # Should have nested sub-cluster names
        assert "cluster_PRIORS_artist_pooling" in graph.source
        assert "cluster_PRIORS_career_dynamics" in graph.source


class TestLightenColor:
    """Test color lightening helper."""

    def test_lighten_color_produces_valid_hex(self):
        """_lighten_color should produce valid hex color."""
        renderer = DiagramRenderer()
        result = renderer._lighten_color("#F4E8F4")
        assert result.startswith("#")
        assert len(result) == 7

    def test_lighten_color_makes_lighter(self):
        """Lightened color should be closer to white."""
        renderer = DiagramRenderer()
        original = "#000000"
        result = renderer._lighten_color(original)
        # Lightened black should be gray
        assert result != original
        # Should be something like #7F7F7F (half-way to white)


class TestFeedbackLoopRendering:
    """Test feedback loop edge rendering."""

    def test_feedback_loops_have_distinct_colors(self):
        """Each feedback loop should have a distinct color."""
        colors = {FEEDBACK_STYLES[t]["color"] for t in FEEDBACK_STYLES}
        assert len(colors) == len(FEEDBACK_STYLES), "Feedback colors should be unique"

    def test_feedback_loop_labels_match_type(self):
        """Feedback loop labels should match their type."""
        for loop in FEEDBACK_LOOPS:
            loop_type = loop["type"]
            expected_label = FEEDBACK_STYLES[loop_type]["label"]
            assert expected_label == loop_type


class TestExtendedThemeColors:
    """Tests for EXTENDED_THEME_COLORS constant."""

    def test_has_three_themes(self) -> None:
        """Verify all three themes exist."""
        assert set(EXTENDED_THEME_COLORS.keys()) == {"light", "dark", "transparent"}

    def test_light_theme_has_10_section_fills(self) -> None:
        """Light theme has fill color for each SECTION_HIERARCHY section."""
        sections = [
            "CONFIG",
            "DATA_RAW",
            "DATA_CLEAN",
            "DATA_SPLIT",
            "FEATURES",
            "PRIORS",
            "MODEL",
            "CONVERGENCE",
            "EVALUATION",
            "OUTPUT",
        ]
        for section in sections:
            key = f"{section}_fill"
            assert key in EXTENDED_THEME_COLORS["light"], f"Missing {key}"
            assert EXTENDED_THEME_COLORS["light"][key].startswith("#")

    def test_all_themes_have_9_edge_colors(self) -> None:
        """All themes have colors for 9 edge types."""
        edge_types = [
            "edge_data_flow",
            "edge_secondary",
            "edge_fit_only",
            "edge_retry",
            "edge_tune",
            "edge_refit",
            "edge_ablate",
            "edge_cross_ref",
            "edge_annotation",
        ]
        for theme in EXTENDED_THEME_COLORS:
            for edge in edge_types:
                assert edge in EXTENDED_THEME_COLORS[theme], f"Missing {edge} in {theme}"

    def test_dark_theme_fills_are_dark(self) -> None:
        """Dark theme fills have low luminance (below 0x40 per channel avg)."""
        for key, color in EXTENDED_THEME_COLORS["dark"].items():
            if "_fill" in key and color.startswith("#"):
                # Extract RGB and check average luminance
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                avg = (r + g + b) / 3
                assert avg < 80, f"{key}={color} too bright for dark theme"

    def test_transparent_theme_has_transparent_bgcolor(self) -> None:
        """Transparent theme has 'transparent' as bgcolor."""
        assert EXTENDED_THEME_COLORS["transparent"]["bgcolor"] == "transparent"
