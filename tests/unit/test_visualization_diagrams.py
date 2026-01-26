"""Tests for Graphviz DOT diagram generation.

These tests verify the diagram generation infrastructure including:
- THEME_COLORS configuration
- Three detail levels: high, intermediate, detailed
- create_high_level_diagram() function
- create_aoty_pipeline_diagram() function (intermediate)
- create_detailed_diagram() function
- generate_all_diagrams() function with level support
- DetailLevel type alias and LEVEL_FUNCTIONS mapping
- Theme-specific styling
- Multi-format export
"""

from aoty_pred.visualization.diagrams import (
    LEVEL_FUNCTIONS,
    THEME_COLORS,
    create_aoty_pipeline_diagram,
    create_detailed_diagram,
    create_high_level_diagram,
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
            assert required_keys.issubset(
                theme_keys
            ), f"{theme} missing keys: {required_keys - theme_keys}"


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
    """Tests for generate_all_diagrams function (intermediate level)."""

    def test_default_generates_nine_diagram_sets(self, tmp_path):
        """Default should create 9 diagram sets (3 levels x 3 themes)."""
        results = generate_all_diagrams(tmp_path)
        assert len(results) == 9
        # Check naming pattern includes level
        assert "pipeline_high_light" in results
        assert "pipeline_intermediate_light" in results
        assert "pipeline_detailed_light" in results
        assert "pipeline_high_dark" in results
        assert "pipeline_intermediate_dark" in results
        assert "pipeline_detailed_dark" in results

    def test_single_level_generates_three_themes(self, tmp_path):
        """Single level should generate 3 theme variants."""
        results = generate_all_diagrams(tmp_path, levels=["high"])
        assert len(results) == 3
        assert all("high" in name for name in results.keys())

    def test_intermediate_level_generates_three_themes(self, tmp_path):
        """Intermediate level should generate 3 theme variants."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])
        assert len(results) == 3
        assert all("intermediate" in name for name in results.keys())

    def test_creates_four_files_per_set(self, tmp_path):
        """Each diagram set should have 4 files (svg, png, pdf, dot)."""
        results = generate_all_diagrams(tmp_path, levels=["high"])
        for name, paths in results.items():
            assert len(paths) == 4, f"{name} should have 4 files, got {len(paths)}"

    def test_creates_svg_files(self, tmp_path):
        """Function should create SVG files."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])
        for name, paths in results.items():
            svg_paths = [p for p in paths if p.suffix == ".svg"]
            assert len(svg_paths) == 1, f"{name} missing SVG"
            assert svg_paths[0].exists()
            assert svg_paths[0].stat().st_size > 0

    def test_creates_png_files(self, tmp_path):
        """Function should create PNG files."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])
        for name, paths in results.items():
            png_paths = [p for p in paths if p.suffix == ".png"]
            assert len(png_paths) == 1, f"{name} missing PNG"
            assert png_paths[0].exists()
            assert png_paths[0].stat().st_size > 0

    def test_creates_pdf_files(self, tmp_path):
        """Function should create PDF files."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])
        for name, paths in results.items():
            pdf_paths = [p for p in paths if p.suffix == ".pdf"]
            assert len(pdf_paths) == 1, f"{name} missing PDF"
            assert pdf_paths[0].exists()
            assert pdf_paths[0].stat().st_size > 0

    def test_creates_dot_files(self, tmp_path):
        """Function should create DOT source files."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])
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
        generate_all_diagrams(nested_dir, levels=["high"])
        assert nested_dir.exists()

    def test_dot_files_contain_theme_specific_colors(self, tmp_path):
        """DOT files should contain theme-specific background colors."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])

        # Check light theme has cream bgcolor
        light_dot = next(p for p in results["pipeline_intermediate_light"] if p.suffix == ".dot")
        light_content = light_dot.read_text(encoding="utf-8")
        assert "#FFFEF0" in light_content

        # Check dark theme has dark bgcolor
        dark_dot = next(p for p in results["pipeline_intermediate_dark"] if p.suffix == ".dot")
        dark_content = dark_dot.read_text(encoding="utf-8")
        assert "#1E1E1E" in dark_content

    def test_filenames_include_level(self, tmp_path):
        """Output filenames should include detail level."""
        results = generate_all_diagrams(tmp_path, levels=["detailed"])
        names = list(results.keys())
        assert all("detailed" in name for name in names)

    def test_each_set_has_four_formats(self, tmp_path):
        """Each diagram set should have svg, png, pdf, dot files."""
        results = generate_all_diagrams(tmp_path, levels=["high"])
        for _name, paths in results.items():
            extensions = {p.suffix for p in paths}
            assert extensions == {".svg", ".png", ".pdf", ".dot"}


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


class TestCreateHighLevelDiagram:
    """Tests for create_high_level_diagram function."""

    def test_returns_digraph(self):
        """Function should return a graphviz.Digraph object."""
        import graphviz

        diagram = create_high_level_diagram("light")
        assert isinstance(diagram, graphviz.Digraph)

    def test_light_theme_creates_diagram(self):
        """Light theme should create a valid diagram with cream background."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        assert "digraph" in source
        assert "#FFFEF0" in source

    def test_dark_theme_creates_diagram(self):
        """Dark theme should create a valid diagram."""
        diagram = create_high_level_diagram("dark")
        source = diagram.source
        assert "digraph" in source
        assert "#1E1E1E" in source

    def test_has_seven_section_clusters(self):
        """High-level diagram should have 7 section clusters."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        assert "1.0 INPUT" in source
        assert "2.0 PREPROCESSING" in source
        assert "3.0 SPLITTING" in source
        assert "4.0 FEATURES" in source
        assert "5.0 MODEL" in source
        assert "6.0 EVALUATION" in source
        assert "7.0 OUTPUT" in source

    def test_has_simplified_nodes(self):
        """High-level should have simplified single nodes per section."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        # Should have summary nodes
        assert "Data Cleaning" in source
        assert "Train/Val/Test Split" in source
        assert "Feature Engineering" in source
        assert "Bayesian Model" in source
        assert "Validation & Diagnostics" in source
        assert "Predictions" in source

    def test_does_not_have_individual_feature_blocks(self):
        """High-level should not show individual feature blocks."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        assert "TemporalBlock" not in source
        assert "GenreBlock" not in source
        assert "ArtistHistoryBlock" not in source

    def test_has_feedback_loop(self):
        """High-level diagram should have EVAL -> MODEL feedback loop."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        assert "tune" in source  # Feedback loop label
        assert "constraint=false" in source  # Feedback edge styling

    def test_has_legend(self):
        """High-level diagram should have a legend."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        assert "LEGEND" in source

    def test_uses_ortho_splines(self):
        """High-level diagram should use orthogonal splines."""
        diagram = create_high_level_diagram("light")
        source = diagram.source
        assert "splines=ortho" in source

    def test_all_themes_work(self):
        """All three themes should produce valid diagrams."""
        for theme in ["light", "dark", "transparent"]:
            diagram = create_high_level_diagram(theme)
            assert "digraph" in diagram.source


class TestCreateDetailedDiagram:
    """Tests for create_detailed_diagram function."""

    def test_returns_digraph(self):
        """Function should return a graphviz.Digraph object."""
        import graphviz

        diagram = create_detailed_diagram("light")
        assert isinstance(diagram, graphviz.Digraph)

    def test_light_theme_creates_diagram(self):
        """Light theme should create a valid diagram."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "digraph" in source
        assert "#FFFEF0" in source

    def test_has_all_six_feature_blocks(self):
        """Detailed diagram should show all 6 feature blocks."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "TemporalBlock" in source
        assert "AlbumTypeBlock" in source
        assert "ArtistHistoryBlock" in source
        assert "ArtistReputationBlock" in source
        assert "GenreBlock" in source
        assert "CollaborationBlock" in source

    def test_has_expanded_preprocessing(self):
        """Detailed diagram should have expanded preprocessing steps."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "Null Handling" in source
        assert "Date Parsing" in source
        assert "Deduplication" in source

    def test_has_expanded_evaluation(self):
        """Detailed diagram should have expanded evaluation section."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "R-hat" in source
        assert "ESS" in source
        assert "LOO" in source
        assert "Pareto-k" in source

    def test_has_model_structure_details(self):
        """Detailed diagram should have complete model structure."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "Non-centered" in source
        assert "Time-varying" in source
        assert "AR(1)" in source
        assert "Likelihood" in source

    def test_has_multiple_feedback_loops(self):
        """Detailed diagram should have multiple feedback loop edges."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "constraint=false" in source  # Feedback loops use constraint=false
        assert "style=dashed" in source
        # Should have convergence retry, sensitivity tune, and pareto-k refit
        assert "retry" in source
        assert "tune" in source
        assert "refit" in source

    def test_uses_smaller_fonts(self):
        """Detailed diagram should use smaller fonts for density."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        # Check for smaller font size (7pt)
        assert 'fontsize="7"' in source or "fontsize=7" in source

    def test_has_hash_verification(self):
        """Detailed diagram should have hash verification node."""
        diagram = create_detailed_diagram("light")
        source = diagram.source
        assert "Hash Verification" in source
        assert "SHA-256" in source

    def test_all_themes_work(self):
        """All three themes should produce valid diagrams."""
        for theme in ["light", "dark", "transparent"]:
            diagram = create_detailed_diagram(theme)
            assert "digraph" in diagram.source


class TestDetailLevelType:
    """Tests for DetailLevel type alias and LEVEL_FUNCTIONS mapping."""

    def test_level_functions_has_three_entries(self):
        """LEVEL_FUNCTIONS should have exactly 3 entries."""
        assert len(LEVEL_FUNCTIONS) == 3

    def test_level_functions_has_all_levels(self):
        """LEVEL_FUNCTIONS should map all three levels."""
        assert "high" in LEVEL_FUNCTIONS
        assert "intermediate" in LEVEL_FUNCTIONS
        assert "detailed" in LEVEL_FUNCTIONS

    def test_level_functions_high_maps_to_high_level(self):
        """LEVEL_FUNCTIONS['high'] should map to create_high_level_diagram."""
        assert LEVEL_FUNCTIONS["high"] == create_high_level_diagram

    def test_level_functions_intermediate_maps_to_pipeline(self):
        """LEVEL_FUNCTIONS['intermediate'] should map to create_aoty_pipeline_diagram."""
        assert LEVEL_FUNCTIONS["intermediate"] == create_aoty_pipeline_diagram

    def test_level_functions_detailed_maps_to_detailed(self):
        """LEVEL_FUNCTIONS['detailed'] should map to create_detailed_diagram."""
        assert LEVEL_FUNCTIONS["detailed"] == create_detailed_diagram

    def test_all_level_functions_are_callable(self):
        """All level functions should be callable."""
        for level, func in LEVEL_FUNCTIONS.items():
            assert callable(func), f"{level} function not callable"

    def test_all_level_functions_accept_theme(self):
        """All level functions should accept a theme parameter."""
        for _level, func in LEVEL_FUNCTIONS.items():
            # Should not raise
            diagram = func("light")
            assert diagram is not None

    def test_all_level_functions_return_digraph(self):
        """All level functions should return graphviz.Digraph."""
        import graphviz

        for _level, func in LEVEL_FUNCTIONS.items():
            diagram = func("light")
            assert isinstance(diagram, graphviz.Digraph)


class TestGenerateAllDiagramsWithLevels:
    """Tests for generate_all_diagrams with level parameter."""

    def test_default_generates_all_levels(self, tmp_path):
        """Default should generate all 3 levels x 3 themes = 9 sets."""
        results = generate_all_diagrams(tmp_path)
        assert len(results) == 9

    def test_single_level_generates_three_themes(self, tmp_path):
        """Single level should generate 3 theme variants."""
        results = generate_all_diagrams(tmp_path, levels=["high"])
        assert len(results) == 3
        assert all("high" in name for name in results.keys())

    def test_two_levels_generates_six_sets(self, tmp_path):
        """Two levels should generate 6 sets (2 levels x 3 themes)."""
        results = generate_all_diagrams(tmp_path, levels=["high", "detailed"])
        assert len(results) == 6
        # Should have high and detailed, not intermediate
        names = list(results.keys())
        assert any("high" in n for n in names)
        assert any("detailed" in n for n in names)
        assert not any("intermediate" in n for n in names)

    def test_filenames_include_level(self, tmp_path):
        """Output filenames should include detail level."""
        results = generate_all_diagrams(tmp_path, levels=["detailed"])
        names = list(results.keys())
        assert all("detailed" in name for name in names)

    def test_each_set_has_four_files(self, tmp_path):
        """Each diagram set should have 4 files (svg, png, pdf, dot)."""
        results = generate_all_diagrams(tmp_path, levels=["intermediate"])
        for _name, paths in results.items():
            assert len(paths) == 4
            extensions = {p.suffix for p in paths}
            assert extensions == {".svg", ".png", ".pdf", ".dot"}

    def test_high_level_files_exist(self, tmp_path):
        """High-level diagram files should exist after generation."""
        results = generate_all_diagrams(tmp_path, levels=["high"])
        for _name, paths in results.items():
            for path in paths:
                assert path.exists(), f"{path} should exist"

    def test_detailed_level_files_exist(self, tmp_path):
        """Detailed diagram files should exist after generation."""
        results = generate_all_diagrams(tmp_path, levels=["detailed"])
        for _name, paths in results.items():
            for path in paths:
                assert path.exists(), f"{path} should exist"
