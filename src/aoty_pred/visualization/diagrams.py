"""Data flow diagram generation for pipeline visualization using Graphviz DOT.

This module generates publication-quality SVG/PNG/PDF diagrams using Graphviz DOT
format with technical manual styling: clustered subgraphs, orthogonal connectors,
monospace fonts, and section numbering.

Three theme variants are supported:
- light: Cream background (#FFFEF0) with dark text
- dark: Dark background (#1E1E1E) with light text
- transparent: No background for embedding

Style Reference: Technical manual aesthetic with:
- rankdir=TB (top-to-bottom)
- splines=ortho (orthogonal right-angle connectors)
- fontname="Courier New" (monospace)
- Clustered subgraphs with section labels (1.0 INPUT, 2.0 PREPROCESSING)
- Node shapes: box, folder, cylinder, diamond, parallelogram, doubleoctagon, note
- Label separators: horizontal lines

Usage:
    >>> from aoty_pred.visualization.diagrams import generate_all_diagrams
    >>> results = generate_all_diagrams(Path("docs/figures"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import graphviz

__all__ = [
    "DiagramTheme",
    "create_aoty_pipeline_diagram",
    "generate_all_diagrams",
]

# Type alias for theme
DiagramTheme = Literal["light", "dark", "transparent"]

# Theme color configurations
THEME_COLORS: dict[DiagramTheme, dict[str, str]] = {
    "light": {
        "bgcolor": "#FFFEF0",
        "fontcolor": "#333333",
        "color": "#333333",
        "fillcolor": "#FFFEF0",
        # Cluster fills (muted pastels)
        "input_fill": "#E8E8D0",
        "preprocess_fill": "#F5F5DC",
        "split_fill": "#FFF3E0",
        "feature_fill": "#E8F5E9",
        "model_fill": "#FFF9C4",
        "eval_fill": "#EDE7F6",
        "output_fill": "#E8E8D0",
        # Node fills
        "data_fill": "#FFF8DC",
        "storage_fill": "#C8E6C9",
        "decision_fill": "#FFE0B2",
        "result_fill": "#FFEB3B",
        "note_fill": "#FFFACD",
        "merge_fill": "#FFD54F",
        "train_fill": "#81C784",
        "val_fill": "#64B5F6",
        "test_fill": "#CE93D8",
    },
    "dark": {
        "bgcolor": "#1E1E1E",
        "fontcolor": "#E0E0E0",
        "color": "#888888",
        "fillcolor": "#2D2D2D",
        # Cluster fills (darker versions)
        "input_fill": "#3D3D2D",
        "preprocess_fill": "#3A3A30",
        "split_fill": "#3D3520",
        "feature_fill": "#2D3D2D",
        "model_fill": "#3D3D20",
        "eval_fill": "#352D3D",
        "output_fill": "#3D3D2D",
        # Node fills (darker versions)
        "data_fill": "#4D4830",
        "storage_fill": "#2D4D2D",
        "decision_fill": "#4D3D20",
        "result_fill": "#4D4D20",
        "note_fill": "#4D4D30",
        "merge_fill": "#4D4020",
        "train_fill": "#2D4D2D",
        "val_fill": "#2D3D4D",
        "test_fill": "#3D2D4D",
    },
    "transparent": {
        "bgcolor": "transparent",
        "fontcolor": "#333333",
        "color": "#333333",
        "fillcolor": "#FFFEF0",
        # Same as light for cluster/node fills
        "input_fill": "#E8E8D0",
        "preprocess_fill": "#F5F5DC",
        "split_fill": "#FFF3E0",
        "feature_fill": "#E8F5E9",
        "model_fill": "#FFF9C4",
        "eval_fill": "#EDE7F6",
        "output_fill": "#E8E8D0",
        "data_fill": "#FFF8DC",
        "storage_fill": "#C8E6C9",
        "decision_fill": "#FFE0B2",
        "result_fill": "#FFEB3B",
        "note_fill": "#FFFACD",
        "merge_fill": "#FFD54F",
        "train_fill": "#81C784",
        "val_fill": "#64B5F6",
        "test_fill": "#CE93D8",
    },
}

# Separator line for labels
SEP = "\u2500" * 21  # Unicode box drawing horizontal line


def _create_graph(theme: DiagramTheme) -> graphviz.Digraph:
    """Create base Digraph with theme-specific global settings."""
    colors = THEME_COLORS[theme]

    graph = graphviz.Digraph(
        name="AOTYPipeline",
        format="svg",
        engine="dot",
    )

    # Global graph settings
    graph.attr(
        rankdir="TB",
        fontname="Courier New",
        fontsize="10",
        label="AOTY PREDICTION PIPELINE\nTechnical Reference",
        labelloc="t",
        labeljust="c",
        pad="0.75",
        nodesep="0.6",
        ranksep="0.8",
        splines="ortho",
        compound="true",
        forcelabels="true",
        overlap="false",
    )

    # Set bgcolor only for non-transparent themes
    if theme != "transparent":
        graph.attr(bgcolor=colors["bgcolor"])

    # Default node style
    graph.attr(
        "node",
        fontname="Courier New",
        fontsize="8",
        shape="box",
        style="filled",
        fillcolor=colors["fillcolor"],
        color=colors["color"],
        fontcolor=colors["fontcolor"],
        penwidth="1",
        margin="0.15,0.1",
    )

    # Default edge style
    graph.attr(
        "edge",
        fontname="Courier New",
        fontsize="7",
        color=colors["color"],
        fontcolor=colors["fontcolor"],
        penwidth="1",
    )

    return graph


def create_aoty_pipeline_diagram(theme: DiagramTheme = "light") -> graphviz.Digraph:
    """Create AOTY prediction pipeline diagram in DOT format.

    Generates a technical manual style diagram showing the complete AOTY
    prediction pipeline with:
    - 1.0 INPUT (raw CSV)
    - 2.0 PREPROCESSING (cleaning, filtering)
    - 3.0 SPLIT INFRASTRUCTURE
    - 4.0 FEATURE ENGINEERING (6 feature blocks)
    - 5.0 BAYESIAN MODEL (priors, MCMC)
    - 6.0 EVALUATION (diagnostics, LOO-CV)
    - 7.0 OUTPUT

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme: "light" (cream bg), "dark" (dark bg), "transparent".

    Returns
    -------
    graphviz.Digraph
        Configured diagram ready for rendering.
    """
    colors = THEME_COLORS[theme]
    graph = _create_graph(theme)

    # =========================================================================
    # SECTION 1.0: INPUT
    # =========================================================================
    with graph.subgraph(name="cluster_input") as c:
        c.attr(
            label="1.0 INPUT",
            style="filled",
            fillcolor=colors["input_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )
        c.node(
            "input_file",
            f"1.1 all_albums_full.csv\n{SEP}\n130,023 rows x 18 cols",
            shape="folder",
            fillcolor=colors["data_fill"],
        )

    # =========================================================================
    # SECTION 2.0: PREPROCESSING
    # =========================================================================
    with graph.subgraph(name="cluster_preprocess") as c:
        c.attr(
            label="2.0 PREPROCESSING",
            style="filled",
            fillcolor=colors["preprocess_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node(
            "schema_val",
            f"2.1 Schema Validation\n{SEP}\nPandera checks\ntype enforcement",
        )

        c.node(
            "cleaning",
            f"2.2 Cleaning Rules\n{SEP}\nnull handling\ndate parsing\ndeduplication",
        )

        c.node(
            "min_ratings",
            f"2.3 Min Ratings Filter\n{SEP}\nthreshold=10\nquality gate",
        )

        c.node(
            "audit_trail",
            f"2.4 Audit Trail\n{SEP}\nJSONL exclusion log\nreproducibility",
            shape="note",
            fillcolor=colors["note_fill"],
        )

        c.node(
            "cleaned_data",
            f"2.5 CLEANED DATA\n{SEP}\n~62,000 rows\n(52% retained)",
            shape="cylinder",
            fillcolor=colors["storage_fill"],
        )

        c.edge("schema_val", "cleaning")
        c.edge("cleaning", "min_ratings")
        c.edge("min_ratings", "audit_trail")
        c.edge("min_ratings", "cleaned_data")

    # =========================================================================
    # SECTION 3.0: SPLIT INFRASTRUCTURE
    # =========================================================================
    with graph.subgraph(name="cluster_split") as c:
        c.attr(
            label="3.0 SPLIT INFRASTRUCTURE",
            style="filled",
            fillcolor=colors["split_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node(
            "within_artist",
            f"3.1 Within-Artist Split\n{SEP}\ntemporal ordering\nper-artist",
        )

        c.node(
            "artist_disjoint",
            f"3.2 Artist-Disjoint Split\n{SEP}\nholdout artists\ngeneralization",
        )

        c.node(
            "split_manifest",
            f"3.3 Split Manifest\n{SEP}\nhash verification\nversion tracking",
            shape="note",
            fillcolor=colors["note_fill"],
        )

        # Data partitions
        c.node(
            "train_set",
            f"TRAIN\n{SEP}\n~41,000 rows (64%)",
            shape="cylinder",
            fillcolor=colors["train_fill"],
        )

        c.node(
            "val_set",
            f"VAL\n{SEP}\n~7,500 rows (12%)",
            shape="cylinder",
            fillcolor=colors["val_fill"],
        )

        c.node(
            "test_set",
            f"TEST\n{SEP}\n~7,500 rows (12%)",
            shape="cylinder",
            fillcolor=colors["test_fill"],
        )

    # =========================================================================
    # SECTION 4.0: FEATURE ENGINEERING
    # =========================================================================
    with graph.subgraph(name="cluster_features") as c:
        c.attr(
            label="4.0 FEATURE ENGINEERING (fit on TRAIN only)",
            style="filled",
            fillcolor=colors["feature_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node(
            "temporal_block",
            f"4.1 TemporalBlock\n{SEP}\nalbum_sequence\ncareer_years\nrelease_gap",
        )

        c.node(
            "album_type_block",
            f"4.2 AlbumTypeBlock\n{SEP}\none-hot encoding\nFittedVocabulary",
        )

        c.node(
            "artist_history_block",
            f"4.3 ArtistHistoryBlock\n{SEP}\nleave-one-out\nprior stats",
        )

        c.node(
            "artist_rep_block",
            f"4.4 ArtistReputationBlock\n{SEP}\nprior mean\nsmoothing",
        )

        c.node(
            "genre_block",
            f"4.5 GenreBlock\n{SEP}\nmulti-hot + PCA\nmin_count=20",
        )

        c.node(
            "collab_block",
            f"4.6 CollaborationBlock\n{SEP}\nordinal encoding\nsolo -> ensemble",
        )

        c.node(
            "feature_pipeline",
            f"4.7 FeaturePipeline\n{SEP}\nfit(train)\ntransform(all)",
            shape="parallelogram",
            fillcolor=colors["merge_fill"],
        )

    # =========================================================================
    # SECTION 5.0: BAYESIAN MODEL
    # =========================================================================
    with graph.subgraph(name="cluster_model") as c:
        c.attr(
            label="5.0 BAYESIAN MODEL",
            style="filled",
            fillcolor=colors["model_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node(
            "prior_config",
            f"5.1 PriorConfig\n{SEP}\n9 hyperparameters\ndefault/diffuse/informative",
        )

        c.node(
            "hierarchical",
            f"5.2 Hierarchical Structure\n{SEP}\nartist random effects\npartial pooling",
        )

        c.node(
            "non_centered",
            f"5.3 Non-centered Param\n{SEP}\nLocScaleReparam\nbetter sampling",
        )

        c.node(
            "time_varying",
            f"5.4 Time-varying Effects\n{SEP}\nrandom walk\ntrend capture",
        )

        c.node(
            "ar1",
            f"5.5 AR(1) Structure\n{SEP}\nstationary constraint\n|rho| < 1",
        )

        c.node(
            "mcmc_sampling",
            f"5.6 MCMC Sampling\n{SEP}\n4 chains x 1000 draws\nNUTS / JAX GPU",
            shape="doubleoctagon",
            fillcolor=colors["result_fill"],
        )

    # =========================================================================
    # SECTION 6.0: EVALUATION
    # =========================================================================
    with graph.subgraph(name="cluster_eval") as c:
        c.attr(
            label="6.0 EVALUATION",
            style="filled",
            fillcolor=colors["eval_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node(
            "convergence",
            f"6.1 Convergence Checks\n{SEP}\nR-hat < 1.01\nESS > 400\ndivergences = 0",
            shape="diamond",
            fillcolor=colors["decision_fill"],
        )

        c.node(
            "loo_cv",
            f"6.2 LOO-CV\n{SEP}\nPSIS importance\nPareto-k < 0.7",
        )

        c.node(
            "calibration",
            f"6.3 Calibration\n{SEP}\ncoverage analysis\nreliability diagrams",
        )

        c.node(
            "prior_predictive",
            f"6.4 Prior Predictive\n{SEP}\nsanity check\nprior influence",
        )

        c.node(
            "sensitivity",
            f"6.5 Sensitivity Analysis\n{SEP}\nprior configs\nfeature ablation",
        )

    # =========================================================================
    # SECTION 7.0: OUTPUT
    # =========================================================================
    with graph.subgraph(name="cluster_output") as c:
        c.attr(
            label="7.0 OUTPUT",
            style="filled",
            fillcolor=colors["output_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node(
            "predictions",
            f"7.1 Predictions\n{SEP}\nposterior mean\n95% credible intervals",
            shape="cylinder",
            fillcolor=colors["storage_fill"],
        )

        c.node(
            "model_artifacts",
            f"7.2 Model Artifacts\n{SEP}\nInferenceData (netCDF)\nmanifest JSON",
            shape="folder",
            fillcolor=colors["data_fill"],
        )

        c.node(
            "publication",
            f"7.3 Publication\n{SEP}\nfigures, tables\nmodel card",
            shape="note",
            fillcolor=colors["note_fill"],
        )

    # =========================================================================
    # APPENDIX: LEGEND
    # =========================================================================
    with graph.subgraph(name="cluster_legend") as c:
        c.attr(
            label="LEGEND",
            style="filled",
            fillcolor=colors["fillcolor"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
        )

        c.node("legend_data", "Data/File", shape="folder", fillcolor=colors["data_fill"])
        c.node("legend_storage", "Storage", shape="cylinder", fillcolor=colors["storage_fill"])
        c.node("legend_decision", "Decision", shape="diamond", fillcolor=colors["decision_fill"])
        c.node("legend_merge", "Transform", shape="parallelogram", fillcolor=colors["merge_fill"])
        c.node("legend_result", "Key Result", shape="doubleoctagon", fillcolor=colors["result_fill"])
        c.node("legend_note", "Note/Ref", shape="note", fillcolor=colors["note_fill"])

        c.edge("legend_data", "legend_storage", style="invis")
        c.edge("legend_storage", "legend_decision", style="invis")
        c.edge("legend_decision", "legend_merge", style="invis")
        c.edge("legend_merge", "legend_result", style="invis")
        c.edge("legend_result", "legend_note", style="invis")

    # =========================================================================
    # MAIN FLOW CONNECTIONS
    # =========================================================================

    # Input -> Preprocessing
    graph.edge("input_file", "schema_val", penwidth="1.5")

    # Preprocessing -> Splits
    graph.edge("cleaned_data", "within_artist", penwidth="1.5")
    graph.edge("cleaned_data", "artist_disjoint", penwidth="1.5")

    # Splits -> Manifests and Data Sets
    graph.edge("within_artist", "split_manifest", style="dashed")
    graph.edge("artist_disjoint", "split_manifest", style="dashed")
    graph.edge("within_artist", "train_set", penwidth="1.5")
    graph.edge("within_artist", "val_set", penwidth="1.5")
    graph.edge("within_artist", "test_set", penwidth="1.5")

    # Train -> Feature Blocks (green flow)
    graph.edge("train_set", "temporal_block", color="#388E3C", penwidth="1.5")
    graph.edge("train_set", "album_type_block", color="#388E3C", penwidth="1.5")
    graph.edge("train_set", "artist_history_block", color="#388E3C", penwidth="1.5")
    graph.edge("train_set", "artist_rep_block", color="#388E3C", penwidth="1.5")
    graph.edge("train_set", "genre_block", color="#388E3C", penwidth="1.5")
    graph.edge("train_set", "collab_block", color="#388E3C", penwidth="1.5")

    # Feature Blocks -> Pipeline
    graph.edge("temporal_block", "feature_pipeline")
    graph.edge("album_type_block", "feature_pipeline")
    graph.edge("artist_history_block", "feature_pipeline")
    graph.edge("artist_rep_block", "feature_pipeline")
    graph.edge("genre_block", "feature_pipeline")
    graph.edge("collab_block", "feature_pipeline")

    # Feature Pipeline -> Model
    graph.edge("feature_pipeline", "prior_config", penwidth="1.5")

    # Model flow
    graph.edge("prior_config", "hierarchical")
    graph.edge("hierarchical", "non_centered")
    graph.edge("non_centered", "time_varying")
    graph.edge("time_varying", "ar1")
    graph.edge("ar1", "mcmc_sampling")

    # MCMC -> Evaluation
    graph.edge("mcmc_sampling", "convergence", penwidth="1.5")
    graph.edge("convergence", "loo_cv")
    graph.edge("loo_cv", "calibration")
    graph.edge("calibration", "prior_predictive")
    graph.edge("prior_predictive", "sensitivity")

    # Evaluation -> Output
    graph.edge("convergence", "predictions", penwidth="1.5")
    graph.edge("mcmc_sampling", "model_artifacts", style="dashed")
    graph.edge("sensitivity", "publication")

    # Feedback loops (dashed)
    graph.edge(
        "convergence", "mcmc_sampling",
        style="dashed", color="#D32F2F",
        xlabel="retry", constraint="false",
    )
    graph.edge(
        "sensitivity", "prior_config",
        style="dashed", color="#7B1FA2",
        xlabel="tune", constraint="false",
    )

    # Test set evaluation (purple flow)
    graph.edge("test_set", "predictions", color="#7B1FA2", style="dashed", xlabel="eval")

    return graph


def generate_all_diagrams(output_dir: Path) -> dict[str, list[Path]]:
    """Generate all AOTY pipeline diagram variants.

    Creates 3 diagram variants (one per theme) in multiple formats.
    Each diagram is saved in SVG, PNG, and PDF formats.

    Parameters
    ----------
    output_dir : Path
        Directory for output files (created if needed).

    Returns
    -------
    dict[str, list[Path]]
        Dict mapping diagram name to list of created file paths.

    Example
    -------
    >>> results = generate_all_diagrams(Path("docs/figures"))
    >>> for name, paths in results.items():
    ...     print(f"{name}: {len(paths)} files")
    aoty_pipeline_light: 3 files
    aoty_pipeline_dark: 3 files
    aoty_pipeline_transparent: 3 files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Path]] = {}
    themes: list[DiagramTheme] = ["light", "dark", "transparent"]
    formats = ["svg", "png", "pdf"]

    for theme in themes:
        name = f"aoty_pipeline_{theme}"
        diagram = create_aoty_pipeline_diagram(theme)

        created_paths: list[Path] = []
        for fmt in formats:
            # graphviz render() writes to directory with auto filename
            # We use render to generate the dot file then convert
            base_path = output_dir / name
            diagram.format = fmt
            output_path = diagram.render(
                filename=str(base_path),
                directory=None,
                cleanup=True,  # Remove intermediate dot file
            )
            created_paths.append(Path(output_path))

        # Also save the .dot source file for reference
        dot_path = output_dir / f"{name}.dot"
        dot_path.write_text(diagram.source, encoding="utf-8")
        created_paths.append(dot_path)

        results[name] = created_paths

    return results
