"""Introspector for CLI configuration parameters.

This module provides ConfigIntrospector, which extracts CLI argument metadata
from the Typer `run` command and produces diagram nodes showing parameter names,
types, default values, and help text.

The introspector groups parameters into semantic clusters (MCMC, convergence,
data filtering, etc.) for organized diagram rendering.

Example:
    >>> from aoty_pred.visualization.introspection import ConfigIntrospector
    >>> ci = ConfigIntrospector()
    >>> result = ci.introspect()
    >>> print(f"Found {len(result.nodes)} config parameters")
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_args, get_origin

from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["ConfigIntrospector", "CONFIG_GROUPS"]


# Semantic grouping of CLI parameters.
# Maps parameter name to group for diagram clustering.
CONFIG_GROUPS: dict[str, str] = {
    # MCMC Configuration (4 params)
    "num_chains": "mcmc",
    "num_samples": "mcmc",
    "num_warmup": "mcmc",
    "target_accept": "mcmc",
    # Convergence Thresholds (3 params)
    "rhat_threshold": "convergence",
    "ess_threshold": "convergence",
    "allow_divergences": "convergence",
    # Data Filtering (3 params)
    "min_ratings": "data_filter",
    "min_albums": "data_filter",
    "max_albums": "data_filter",
    # Feature Ablation (3 params)
    "enable_genre": "ablation",
    "enable_artist": "ablation",
    "enable_temporal": "ablation",
    # Heteroscedastic Noise (4 params)
    "n_exponent": "noise",
    "learn_n_exponent": "noise",
    "n_exponent_alpha": "noise",
    "n_exponent_beta": "noise",
    # Execution Control (7 params)
    "seed": "execution",
    "skip_existing": "execution",
    "stages": "execution",
    "dry_run": "execution",
    "strict": "execution",
    "verbose": "execution",
    "resume": "execution",
    # Preflight (5 params)
    "preflight": "preflight",
    "preflight_only": "preflight",
    "force_run": "preflight",
    "preflight_full": "preflight",
    "recalibrate": "preflight",
}


def _get_type_name(annotation: Any) -> str:
    """Extract a readable type name from a type annotation.

    Handles Annotated types, Optional, and basic types. Also handles
    string annotations which are common with forward references and
    runtime signature inspection.

    Args:
        annotation: Type annotation from parameter signature.

    Returns:
        Human-readable type name string.
    """
    if annotation is inspect.Parameter.empty:
        return "Any"

    # Handle string annotations (common with runtime signature inspection)
    if isinstance(annotation, str):
        ann_str = annotation
        # Parse Annotated[BaseType, ...] -> BaseType
        if ann_str.startswith("Annotated["):
            # Extract the base type from Annotated[BaseType, ...]
            # Find the first comma or bracket that ends the type
            inner = ann_str[10:]  # Remove "Annotated["
            # Find where the base type ends
            bracket_depth = 0
            for i, char in enumerate(inner):
                if char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "," and bracket_depth == 0:
                    base_type = inner[:i].strip()
                    return base_type
            # No comma found, shouldn't happen with valid Annotated
            return inner.rstrip("]").strip()
        # Parse Optional[Type] -> Type?
        if ann_str.startswith("Optional["):
            inner = ann_str[9:-1]  # Remove "Optional[" and "]"
            return f"{inner}?"
        return ann_str

    # Handle Annotated types (e.g., Annotated[int, typer.Option(...)])
    origin = get_origin(annotation)
    if origin is not None:
        # Check if it's typing.Annotated
        type_name = getattr(origin, "__name__", str(origin))
        if "Annotated" in str(origin):
            # Get the base type from Annotated[BaseType, ...]
            args = get_args(annotation)
            if args:
                base_type = args[0]
                # Handle Optional[str] -> str?
                base_origin = get_origin(base_type)
                if base_origin is not None:
                    # E.g., Optional[str] shows as Union[str, None]
                    base_args = get_args(base_type)
                    if type(None) in base_args:
                        # It's Optional
                        non_none = [a for a in base_args if a is not type(None)]
                        if non_none:
                            return f"{non_none[0].__name__}?"
                return getattr(base_type, "__name__", str(base_type))
        # Handle Optional[str] directly
        if type_name in ("Union", "UnionType"):
            args = get_args(annotation)
            if type(None) in args:
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    return f"{non_none[0].__name__}?"
        return type_name

    # Basic types
    return getattr(annotation, "__name__", str(annotation))


def _format_default(default: Any) -> str:
    """Format a default value for display.

    Args:
        default: Default value from parameter.

    Returns:
        Formatted string representation.
    """
    if default is None:
        return "None"
    if isinstance(default, bool):
        return str(default)  # Python True/False
    if isinstance(default, str):
        return f'"{default}"'
    return str(default)


class ConfigIntrospector:
    """Introspector for CLI configuration parameters.

    Extracts parameter metadata from the CLI `run` command and produces
    diagram nodes showing each parameter with its type, default value,
    and help text. Parameters are grouped into semantic clusters.

    Attributes:
        func: The function to introspect (defaults to cli.run).

    Example:
        >>> ci = ConfigIntrospector()
        >>> result = ci.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata['type_name']}")
    """

    def __init__(self, func: Callable[..., Any] | None = None) -> None:
        """Initialize ConfigIntrospector.

        Args:
            func: Optional function to introspect. Defaults to the CLI
                `run` command.
        """
        self._func = func

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "config"

    def _get_func(self) -> Callable[..., Any]:
        """Get the function to introspect.

        Returns:
            The target function (default: aoty_pred.cli.run).
        """
        if self._func is not None:
            return self._func
        # Import here to avoid circular dependencies
        from aoty_pred.cli import run

        return run

    def _extract_param_info(self, name: str, param: inspect.Parameter) -> dict[str, Any]:
        """Extract metadata from a single parameter.

        Args:
            name: Parameter name.
            param: Parameter object from signature.

        Returns:
            Dict with keys: param_name, default, help, type_name, group.
        """
        default_obj = param.default

        # Extract default value
        default_val: Any = None
        if hasattr(default_obj, "default"):
            # Typer OptionInfo wraps the actual default
            default_val = default_obj.default
        elif default_obj is not inspect.Parameter.empty:
            default_val = default_obj

        # Extract help text
        help_text: str | None = None
        if hasattr(default_obj, "help"):
            help_text = default_obj.help

        # Extract type name
        type_name = _get_type_name(param.annotation)

        # Determine group
        group = CONFIG_GROUPS.get(name, "other")

        return {
            "param_name": name,
            "default": default_val,
            "help": help_text,
            "type_name": type_name,
            "group": group,
        }

    def introspect(self) -> IntrospectionResult:
        """Introspect CLI parameters and return diagram elements.

        Creates one node per CLI parameter with the parameter name, type,
        and default value in the label. Parameters are grouped into semantic
        clusters for organized diagram rendering.

        Returns:
            IntrospectionResult containing config parameter nodes.
        """
        func = self._get_func()
        sig = inspect.signature(func)

        nodes: list[NodeSpec] = []
        clusters: dict[str, list[str]] = {}
        groups_seen: set[str] = set()

        # Process parameters in sorted order for determinism
        for name in sorted(sig.parameters.keys()):
            # Skip typer internal context parameter
            if name == "ctx":
                continue

            param = sig.parameters[name]
            info = self._extract_param_info(name, param)

            # Format label with type and default
            default_str = _format_default(info["default"])
            label = f"{name}\n{info['type_name']}\ndefault: {default_str}"

            node_id = f"config:{name}"
            group = info["group"]
            cluster_name = f"config_{group}"

            node = NodeSpec(
                id=node_id,
                label=label,
                category="config",
                cluster=cluster_name,
                metadata=info,
            )
            nodes.append(node)

            # Track cluster membership
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(node_id)
            groups_seen.add(group)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=[],  # Config parameters don't have edges
            clusters=clusters,
            metadata={
                "param_count": len(nodes),
                "groups": sorted(groups_seen),
            },
        )
