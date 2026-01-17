"""Tests for DataFlowDiagram builder class.

These tests verify the diagram generation infrastructure including:
- STAGE_COLORS mapping
- DiagramNode dataclass
- DataFlowDiagram initialization with themes
- Node and arrow addition
- Legend generation
- Multi-format export
"""

from pathlib import Path

# Use non-interactive backend before importing pyplot
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from aoty_pred.visualization.diagrams import (
    STAGE_COLORS,
    DataFlowDiagram,
    DetailLevel,
    DiagramNode,
    DiagramTheme,
)


class TestStageColors:
    """Tests for STAGE_COLORS constant."""

    def test_has_seven_entries(self):
        """STAGE_COLORS should have 7 pipeline stage mappings."""
        assert len(STAGE_COLORS) == 7

    def test_all_values_are_hex_colors(self):
        """All color values should be valid hex color strings."""
        for stage, color in STAGE_COLORS.items():
            assert isinstance(color, str), f"{stage} color is not a string"
            assert color.startswith("#"), f"{stage} color doesn't start with #"
            assert len(color) == 7, f"{stage} color is not valid hex (6 digits)"
            # Verify all characters after # are valid hex
            hex_chars = set("0123456789ABCDEFabcdef")
            for char in color[1:]:
                assert char in hex_chars, f"{stage} has invalid hex char: {char}"

    def test_contains_expected_keys(self):
        """STAGE_COLORS should contain all expected pipeline stage keys."""
        expected_keys = {
            "data_input",
            "sanitization",
            "splitting",
            "features",
            "model",
            "validation",
            "output",
        }
        assert set(STAGE_COLORS.keys()) == expected_keys


class TestDiagramNode:
    """Tests for DiagramNode dataclass."""

    def test_create_node_with_all_fields(self):
        """DiagramNode should accept all specified fields."""
        node = DiagramNode(
            id="test_node",
            x=0.1,
            y=0.5,
            width=0.2,
            height=0.1,
            label="Test Node",
            stage_type="data_input",
            sublabel="optional text",
        )
        assert node.id == "test_node"
        assert node.x == 0.1
        assert node.y == 0.5
        assert node.width == 0.2
        assert node.height == 0.1
        assert node.label == "Test Node"
        assert node.stage_type == "data_input"
        assert node.sublabel == "optional text"

    def test_sublabel_defaults_to_none(self):
        """sublabel should default to None when not provided."""
        node = DiagramNode(
            id="simple",
            x=0.0,
            y=0.0,
            width=0.1,
            height=0.1,
            label="Simple",
            stage_type="model",
        )
        assert node.sublabel is None

    def test_id_and_label_accessible(self):
        """Node id and label should be accessible as attributes."""
        node = DiagramNode(
            id="my_id",
            x=0.2,
            y=0.3,
            width=0.15,
            height=0.08,
            label="My Label",
            stage_type="features",
        )
        assert node.id == "my_id"
        assert node.label == "My Label"
        assert node.stage_type == "features"


class TestDataFlowDiagramInit:
    """Tests for DataFlowDiagram initialization."""

    def test_light_theme_colors(self):
        """Light theme should return correct color tuple."""
        diagram = DataFlowDiagram("high", "light")
        try:
            assert diagram.bg_color == "white"
            assert diagram.text_color == "#333333"
            assert diagram.border_color == "#555555"
        finally:
            diagram.close()

    def test_dark_theme_colors(self):
        """Dark theme should return correct color tuple."""
        diagram = DataFlowDiagram("high", "dark")
        try:
            assert diagram.bg_color == "#1E1E1E"
            assert diagram.text_color == "#E0E0E0"
            assert diagram.border_color == "#888888"
        finally:
            diagram.close()

    def test_transparent_theme_colors(self):
        """Transparent theme should return correct color tuple."""
        diagram = DataFlowDiagram("high", "transparent")
        try:
            assert diagram.bg_color == "none"
            assert diagram.text_color == "#333333"
            assert diagram.border_color == "#555555"
        finally:
            diagram.close()

    def test_figure_is_created(self):
        """Figure should be created on initialization."""
        diagram = DataFlowDiagram("intermediate", "light")
        try:
            assert diagram.fig is not None
            assert diagram.ax is not None
        finally:
            diagram.close()

    def test_custom_figure_size(self):
        """Custom figure dimensions should be applied."""
        diagram = DataFlowDiagram(
            "detailed", "light", fig_width=15.0, fig_height=10.0
        )
        try:
            fig_width, fig_height = diagram.fig.get_size_inches()
            assert fig_width == pytest.approx(15.0)
            assert fig_height == pytest.approx(10.0)
        finally:
            diagram.close()

    def test_title_is_set(self):
        """Title should be set when provided."""
        diagram = DataFlowDiagram("high", "light", title="Test Title")
        try:
            # Title is set via ax.set_title
            title = diagram.ax.get_title()
            assert title == "Test Title"
        finally:
            diagram.close()


class TestAddNode:
    """Tests for add_node method."""

    @pytest.fixture
    def diagram(self):
        """Create a diagram for testing."""
        d = DataFlowDiagram("high", "light")
        yield d
        d.close()

    def test_add_node_appends_to_list(self, diagram):
        """add_node should append node to nodes list."""
        node = DiagramNode(
            id="test",
            x=0.1,
            y=0.5,
            width=0.15,
            height=0.08,
            label="Test",
            stage_type="data_input",
        )
        diagram.add_node(node)
        assert len(diagram.nodes) == 1
        assert diagram.nodes[0] == node

    def test_add_node_creates_patch(self, diagram):
        """add_node should add a patch to the axes."""
        initial_patches = len(diagram.ax.patches)
        node = DiagramNode(
            id="test",
            x=0.1,
            y=0.5,
            width=0.15,
            height=0.08,
            label="Test",
            stage_type="model",
        )
        diagram.add_node(node)
        # Should have added one patch (the box)
        assert len(diagram.ax.patches) == initial_patches + 1


class TestAddArrow:
    """Tests for add_arrow method."""

    @pytest.fixture
    def diagram(self):
        """Create a diagram for testing."""
        d = DataFlowDiagram("high", "light")
        yield d
        d.close()

    def test_add_straight_arrow(self, diagram):
        """add_arrow should create an arrow patch."""
        initial_patches = len(diagram.ax.patches)
        diagram.add_arrow((0.1, 0.5), (0.3, 0.5))
        assert len(diagram.ax.patches) == initial_patches + 1

    def test_add_curved_arrow(self, diagram):
        """add_arrow with curved=True should work."""
        initial_patches = len(diagram.ax.patches)
        diagram.add_arrow((0.1, 0.5), (0.3, 0.7), curved=True)
        assert len(diagram.ax.patches) == initial_patches + 1

    def test_add_dashed_arrow(self, diagram):
        """add_arrow with dashed=True should work."""
        initial_patches = len(diagram.ax.patches)
        diagram.add_arrow((0.1, 0.5), (0.3, 0.5), dashed=True)
        assert len(diagram.ax.patches) == initial_patches + 1

    def test_add_arrow_with_label(self, diagram):
        """add_arrow with label should add text."""
        initial_texts = len(diagram.ax.texts)
        diagram.add_arrow((0.1, 0.5), (0.3, 0.5), label="Flow")
        # Should have added text for label
        assert len(diagram.ax.texts) > initial_texts


class TestAddLegend:
    """Tests for add_legend method."""

    @pytest.fixture
    def diagram(self):
        """Create a diagram for testing."""
        d = DataFlowDiagram("high", "light")
        yield d
        d.close()

    def test_legend_is_added(self, diagram):
        """add_legend should create a legend on axes."""
        assert diagram.ax.get_legend() is None
        diagram.add_legend()
        assert diagram.ax.get_legend() is not None

    def test_legend_has_correct_entry_count(self, diagram):
        """Legend should have entries for key stage types."""
        diagram.add_legend()
        legend = diagram.ax.get_legend()
        # Legend shows 5 key stages (data_input, features, model, validation, output)
        # to avoid clutter in technical manual style
        assert len(legend.legend_handles) == 5


class TestSave:
    """Tests for save method."""

    def test_save_creates_svg_file(self, tmp_path):
        """save() should create SVG file when requested."""
        diagram = DataFlowDiagram("high", "light")

        # Add a simple node to have content
        node = DiagramNode(
            id="test",
            x=0.4,
            y=0.4,
            width=0.2,
            height=0.1,
            label="Test",
            stage_type="data_input",
        )
        diagram.add_node(node)

        output_base = tmp_path / "test_diagram"
        paths = diagram.save(output_base, formats=("svg",))

        assert len(paths) == 1
        assert paths[0].suffix == ".svg"
        assert paths[0].exists()
        assert paths[0].stat().st_size > 0

    def test_save_creates_multiple_formats(self, tmp_path):
        """save() should create files for all requested formats."""
        diagram = DataFlowDiagram("high", "dark")

        node = DiagramNode(
            id="test",
            x=0.4,
            y=0.4,
            width=0.2,
            height=0.1,
            label="Test",
            stage_type="model",
        )
        diagram.add_node(node)

        output_base = tmp_path / "multi_format"
        paths = diagram.save(output_base, formats=("svg", "png", "pdf"))

        assert len(paths) == 3
        extensions = {p.suffix for p in paths}
        assert extensions == {".svg", ".png", ".pdf"}
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 0


class TestThemeTransparency:
    """Tests for transparent theme handling."""

    def test_transparent_theme_alpha(self):
        """Transparent theme should set figure alpha to 0."""
        diagram = DataFlowDiagram("high", "transparent")
        try:
            # Check figure patch alpha
            alpha = diagram.fig.patch.get_alpha()
            # Alpha should be 0 or close to 0
            assert alpha is None or alpha == 0 or alpha < 0.01
        finally:
            diagram.close()

    def test_transparent_theme_axes_alpha(self):
        """Transparent theme should set axes patch alpha to 0."""
        diagram = DataFlowDiagram("high", "transparent")
        try:
            alpha = diagram.ax.patch.get_alpha()
            assert alpha is None or alpha == 0 or alpha < 0.01
        finally:
            diagram.close()


class TestClose:
    """Tests for close method."""

    def test_close_does_not_error_on_open_figure(self):
        """close() should not raise on an open figure."""
        diagram = DataFlowDiagram("high", "light")
        # Should not raise
        diagram.close()

    def test_close_does_not_error_on_already_closed(self):
        """close() should not raise if called twice."""
        diagram = DataFlowDiagram("high", "light")
        diagram.close()
        # Second close should not raise
        diagram.close()


class TestTypeAliases:
    """Tests for type aliases."""

    def test_detail_level_type(self):
        """DetailLevel should accept valid values."""
        # These should all work without error
        for level in ["high", "intermediate", "detailed"]:
            diagram = DataFlowDiagram(level, "light")  # type: ignore[arg-type]
            diagram.close()

    def test_diagram_theme_type(self):
        """DiagramTheme should accept valid values."""
        for theme in ["light", "dark", "transparent"]:
            diagram = DataFlowDiagram("high", theme)  # type: ignore[arg-type]
            diagram.close()
