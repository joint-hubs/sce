"""
@module: sce.meta
@depends: 
@exports: component, tracks_dependency
@paper_ref: N/A
@data_flow: decorator metadata -> architecture docs
"""

from functools import wraps
from typing import Any, Callable, List, Optional


def component(
    name: str,
    responsibility: str,
    depends_on: Optional[List[str]] = None,
):
    """
    Decorator to mark classes as architectural components for auto-documentation.
    
    Args:
        name: Component name (e.g., "StatisticalContextEngine")
        responsibility: Brief description of component's role
        depends_on: List of component names this depends on
    
    Example:
        @component(
            name="StatisticalContextEngine",
            responsibility="Computes hierarchical statistical features",
            depends_on=["ContextConfig", "StatsAggregator"]
        )
        class StatisticalContextEngine:
            pass
    """
    def decorator(cls: type) -> type:
        cls.__component_metadata__ = {
            "name": name,
            "responsibility": responsibility,
            "depends_on": depends_on or [],
        }
        return cls
    return decorator


def tracks_dependency(dependency_type: str) -> Callable:
    """
    Decorator to track external dependencies (data, models, etc.) for lineage.
    
    Args:
        dependency_type: Type of dependency ("data", "model", "config")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # TODO: Implement dependency tracking logic
            return func(*args, **kwargs)
        wrapper.__tracks_dependency__ = dependency_type
        return wrapper
    return decorator
