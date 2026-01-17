"""Tests for Graphviz DOT diagram generation.

These tests verify the diagram generation infrastructure including:
- THEME_COLORS configuration
- create_aoty_pipeline_diagram() function
- generate_all_diagrams() function
- Theme-specific styling
- Multi-format export
"""

from pathlib import Path

import pytest

from aoty_pred.visualization.diagrams import (
    THEME_COLORS,
    DiagramTheme,
    create_aoty_pipeline_diagram,
    generate_all_diagrams,
)


class TestThemeColors:
    """Tests for THEME_COLORS constant."""

    def test_has_three_themes(self):
        """THEME_COLORS should have light, dark, and transparent themes."""
        assert len(THEME_COLORS) == 3
        assert "light" in THEME_COLORS
        assert "dark" in THEME_COLORS
        assert "transparent" in THEME_COLORS

    def test_light_theme_has_cream_bgcolor(self):
        """Light theme should have cream background."""
        assert THEME_COLORS["light"]["bgcolor"] == "#FFFEF0"

    def test_dark_theme_has_dark_bgcolor(self):
        """Dark theme should have dark background."""
        assert THEME_COLORS["dark"]["bgcolor"] == "#1E1E1E"

    def test_transparent_theme_has_transparent_bgcolor(self):
        """Transparent theme should have transparent background."""
        assert THEME_COLORS["transparent"]["bgcolor"] == "transparent"

    def test_all_themes_have_required_keys(self):
        """All themes should have required color keys."""
        required_keys = {
            "bgcolor",
            "fontcolor",
            "color",
            "fillcolor",
            "input_fill",
            "preprocess_fill",
            "split_fill",
            "feature_fill",
            "model_fill",
            "eval_fill",
            "output_fill",
            "data_fill",
            "storage_fill",
            "decision_fill",
            "result_fill",
            "note_fill",
            "merge_fill",
            "train_fill",
            "val_fill",
            "test_fill",
        }
        for theme in THEME_COLORS:
            theme_keys = set(THEME_COLORS[theme].keys())
            assert required_keys.issubset(theme_keys), f"{theme} missing keys: {required_keys - theme_keys}"


class TestCreateAotyPipelineDiagram:
    """Tests for create_aoty_pipeline_diagram function."""

    def test_returns_digraph(self):
        """Function should return a graphviz.Digraph object."""
        import graphviz

        diagram = create_aoty_pipeline_diagram("light")
        assert isinstance(diagram, graphviz.Digraph)

    def test_light_theme_creates_diagram(self):
        """Light theme should create a valid diagram."""
        diagram = create_aoty_pipeline_diagram("light")
        # Check DOT source contains expected content
        source = diagram.source
        assert "digraph" in source
        assert "bgcolor" in source
        assert "#FFFEF0" in source

    def test_dark_theme_creates_diagram(self):
        """Dark theme should create a valid diagram."""
        diagram = create_aoty_pipeline_diagram("dark")
        source = diagram.source
        assert "digraph" in source
        assert "#1E1E1E" in source

    def test_transparent_theme_creates_diagram(self):
        """Transparent theme should create a valid diagram."""
        diagram = create_aoty_pipeline_diagram("transparent")
        source = diagram.source
        assert "digraph" in source
        # Transparent theme should not have bgcolor attribute
        # (or it's set to transparent which graphviz ignores)

    def test_diagram_has_title(self):
        """Diagram should have a title label."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "AOTY PREDICTION PIPELINE" in source

    def test_diagram_uses_ortho_splines(self):
        """Diagram should use orthogonal splines."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "splines=ortho" in source

    def test_diagram_uses_courier_font(self):
        """Diagram should use Courier New font."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "Courier New" in source

    def test_diagram_has_section_clusters(self):
        """Diagram should have numbered section clusters."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "1.0 INPUT" in source
        assert "2.0 PREPROCESSING" in source
        assert "3.0 SPLIT INFRASTRUCTURE" in source
        assert "4.0 FEATURE ENGINEERING" in source
        assert "5.0 BAYESIAN MODEL" in source
        assert "6.0 EVALUATION" in source
        assert "7.0 OUTPUT" in source

    def test_diagram_has_feature_blocks(self):
        """Diagram should include all 6 feature blocks."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "TemporalBlock" in source
        assert "AlbumTypeBlock" in source
        assert "ArtistHistoryBlock" in source
        assert "ArtistReputationBlock" in source
        assert "GenreBlock" in source
        assert "CollaborationBlock" in source

    def test_diagram_has_legend(self):
        """Diagram should have a legend cluster."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "LEGEND" in source
        assert "legend_data" in source
        assert "legend_storage" in source

    def test_diagram_has_various_shapes(self):
        """Diagram should use various node shapes."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "shape=folder" in source
        assert "shape=cylinder" in source
        assert "shape=diamond" in source
        assert "shape=parallelogram" in source
        assert "shape=doubleoctagon" in source
        assert "shape=note" in source


class TestGenerateAllDiagrams:
    """Tests for generate_all_diagrams function."""

    def test_creates_three_diagram_sets(self, tmp_path):
        """Function should create 3 diagram sets (one per theme)."""
        results = generate_all_diagrams(tmp_path)
        assert len(results) == 3
        assert "aoty_pipeline_light" in results
        assert "aoty_pipeline_dark" in results
        assert "aoty_pipeline_transparent" in results

    def test_creates_four_files_per_set(self, tmp_path):
        """Each diagram set should have 4 files (svg, png, pdf, dot)."""
        results = generate_all_diagrams(tmp_path)
        for name, paths in results.items():
            assert len(paths) == 4, f"{name} should have 4 files, got {len(paths)}"

    def test_creates_svg_files(self, tmp_path):
        """Function should create SVG files."""
        results = generate_all_diagrams(tmp_path)
        for name, paths in results.items():
            svg_paths = [p for p in paths if p.suffix == ".svg"]
            assert len(svg_paths) == 1, f"{name} missing SVG"
            assert svg_paths[0].exists()
            assert svg_paths[0].stat().st_size > 0

    def test_creates_png_files(self, tmp_path):
        """Function should create PNG files."""
        results = generate_all_diagrams(tmp_path)
        for name, paths in results.items():
            png_paths = [p for p in paths if p.suffix == ".png"]
            assert len(png_paths) == 1, f"{name} missing PNG"
            assert png_paths[0].exists()
            assert png_paths[0].stat().st_size > 0

    def test_creates_pdf_files(self, tmp_path):
        """Function should create PDF files."""
        results = generate_all_diagrams(tmp_path)
        for name, paths in results.items():
            pdf_paths = [p for p in paths if p.suffix == ".pdf"]
            assert len(pdf_paths) == 1, f"{name} missing PDF"
            assert pdf_paths[0].exists()
            assert pdf_paths[0].stat().st_size > 0

    def test_creates_dot_files(self, tmp_path):
        """Function should create DOT source files."""
        results = generate_all_diagrams(tmp_path)
        for name, paths in results.items():
            dot_paths = [p for p in paths if p.suffix == ".dot"]
            assert len(dot_paths) == 1, f"{name} missing DOT"
            assert dot_paths[0].exists()
            # DOT file should be readable text
            content = dot_paths[0].read_text(encoding="utf-8")
            assert "digraph" in content

    def test_creates_output_directory(self, tmp_path):
        """Function should create output directory if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "output" / "dir"
        assert not nested_dir.exists()
        generate_all_diagrams(nested_dir)
        assert nested_dir.exists()

    def test_dot_files_contain_theme_specific_colors(self, tmp_path):
        """DOT files should contain theme-specific background colors."""
        results = generate_all_diagrams(tmp_path)

        # Check light theme has cream bgcolor
        light_dot = next(p for p in results["aoty_pipeline_light"] if p.suffix == ".dot")
        light_content = light_dot.read_text(encoding="utf-8")
        assert "#FFFEF0" in light_content

        # Check dark theme has dark bgcolor
        dark_dot = next(p for p in results["aoty_pipeline_dark"] if p.suffix == ".dot")
        dark_content = dark_dot.read_text(encoding="utf-8")
        assert "#1E1E1E" in dark_content


class TestDiagramThemeType:
    """Tests for DiagramTheme type alias."""

    def test_valid_themes_are_accepted(self):
        """DiagramTheme should accept light, dark, transparent."""
        for theme in ["light", "dark", "transparent"]:
            # Should not raise
            diagram = create_aoty_pipeline_diagram(theme)  # type: ignore[arg-type]
            assert diagram is not None


class TestDiagramContent:
    """Tests for diagram content accuracy."""

    def test_input_section_has_csv_info(self):
        """Input section should mention CSV file details."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "all_albums_full.csv" in source
        assert "130,023" in source

    def test_preprocessing_section_has_steps(self):
        """Preprocessing section should have schema, cleaning, filter steps."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "Schema Validation" in source
        assert "Cleaning Rules" in source
        assert "Min Ratings Filter" in source

    def test_split_section_has_strategies(self):
        """Split section should mention both split strategies."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "Within-Artist Split" in source
        assert "Artist-Disjoint Split" in source

    def test_model_section_has_components(self):
        """Model section should have all model components."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "PriorConfig" in source
        assert "Hierarchical" in source
        assert "Non-centered" in source
        assert "Time-varying" in source
        assert "AR(1)" in source
        assert "MCMC Sampling" in source

    def test_eval_section_has_checks(self):
        """Evaluation section should have convergence and validation checks."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "Convergence" in source
        assert "R-hat" in source
        assert "ESS" in source
        assert "LOO-CV" in source
        assert "Calibration" in source

    def test_output_section_has_artifacts(self):
        """Output section should mention predictions and artifacts."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        assert "Predictions" in source
        assert "Model Artifacts" in source
        assert "Publication" in source


class TestEdgeConnections:
    """Tests for diagram edge connections."""

    def test_has_feedback_loops(self):
        """Diagram should have feedback loop edges."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        # Feedback loops use dashed style and constraint=false
        assert "style=dashed" in source
        assert "constraint=false" in source

    def test_has_train_flow_edges(self):
        """Diagram should have train set to feature block edges."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        # Green color for train flow
        assert "#388E3C" in source

    def test_has_test_evaluation_edge(self):
        """Diagram should have test set to predictions edge."""
        diagram = create_aoty_pipeline_diagram("light")
        source = diagram.source
        # Purple color for test evaluation
        assert "#7B1FA2" in source
