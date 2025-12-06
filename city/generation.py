"""
Generation functions for city components (centers, corridors, parks).

This module provides pure functions that generate city components as data structures
without modifying the grid. Follows a functional programming approach.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set


def place_points(
    num_points: int,
    grid_rows: int,
    grid_cols: int,
    placement_strategy: str = "uniform",
    min_separation: int = 5,
    starting_point: Optional[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """
    Common utility for placing points (centers, parks, etc.) on a grid.

    Args:
        num_points: Number of points to place
        grid_rows: Grid height
        grid_cols: Grid width
        placement_strategy: "uniform" (evenly spaced), "clustered" (grouped), or "random"
        min_separation: Minimum distance between points (used for random placement)
        starting_point: Optional (row, col) tuple for the first point.
                       If None, defaults to center of grid

    Returns:
        List of (row, col) tuples representing point positions
    """
    if num_points == 0:
        return []

    points = []

    # Determine starting point
    if starting_point is None:
        base_row = grid_rows // 2
        base_col = grid_cols // 2
    else:
        base_row, base_col = starting_point

    if placement_strategy == "uniform":
        # First point at starting location
        points.append((base_row, base_col))

        # Additional points in a circle around the starting point
        if num_points > 1:
            radius = min(grid_rows, grid_cols) // 3
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

            for angle in angles[1:]:
                row = int(base_row + radius * np.sin(angle))
                col = int(base_col + radius * np.cos(angle))

                # Ensure within bounds
                row = np.clip(row, 0, grid_rows - 1)
                col = np.clip(col, 0, grid_cols - 1)

                points.append((row, col))

    elif placement_strategy == "clustered":
        # Place points clustered around the starting point
        for i in range(num_points):
            # Add some randomness but keep clustered
            offset_row = np.random.randint(-grid_rows//6, grid_rows//6)
            offset_col = np.random.randint(-grid_cols//6, grid_cols//6)

            row = np.clip(base_row + offset_row, 0, grid_rows - 1)
            col = np.clip(base_col + offset_col, 0, grid_cols - 1)

            points.append((row, col))

    else:  # random
        # Random placement with minimum separation
        attempts = 0
        max_attempts = 1000

        while len(points) < num_points and attempts < max_attempts:
            row = np.random.randint(0, grid_rows)
            col = np.random.randint(0, grid_cols)

            # Check minimum separation from existing points
            valid = True
            for existing_point in points:
                distance = np.sqrt((row - existing_point[0])**2 + (col - existing_point[1])**2)
                if distance < min_separation:
                    valid = False
                    break

            if valid:
                points.append((row, col))

            attempts += 1

    return points


@dataclass
class CityCenter:
    """
    Represents a single city center with density multipliers.

    City centers provide density multipliers that decay exponentially from the center.
    Multipliers are relative to base densities set in CityConfig.

    Attributes:
        position: (row, col) position on grid
        strength: Overall strength of this center (applied to all density types)
        housing_peak_multiplier: Peak density multiplier for housing at center
        office_peak_multiplier: Peak density multiplier for offices at center (1.0 = neutral)
        shop_peak_multiplier: Peak density multiplier for shops at center (1.0 = neutral)
        decay_rate: Exponential decay rate from center
    """
    position: Tuple[int, int]
    strength: float
    housing_peak_multiplier: float
    office_peak_multiplier: float = 1.0
    shop_peak_multiplier: float = 1.0
    decay_rate: float = 0.20


@dataclass
class TransportationCorridor:
    """
    Represents a transportation corridor with density multipliers.

    Attributes:
        corridor_type: Type of corridor (from CorridorType enum)
        blocks: Set of (row, col) positions affected by this corridor
        housing_multiplier: Density multiplier for housing (1.0 = neutral)
        office_multiplier: Density multiplier for offices (1.0 = neutral)
        shop_multiplier: Density multiplier for shops (1.0 = neutral)
        width_blocks: Width of corridor in blocks
    """
    corridor_type: str  # CorridorType value
    blocks: Set[Tuple[int, int]]
    housing_multiplier: float = 1.0
    office_multiplier: float = 1.0
    shop_multiplier: float = 1.0
    width_blocks: int = 2


@dataclass
class Park:
    """
    Represents a park in the city.

    Attributes:
        center: (row, col) center position of the park
        size: Size in blocks
        blocks: Set of (row, col) positions occupied by the park
        shape: Shape of the park ("square" or "circle")
    """
    center: Tuple[int, int]
    size: int
    blocks: Set[Tuple[int, int]]
    shape: str = "square"


def generate_city_centers(
    centers_config,
    city_config,
    grid_rows: int,
    grid_cols: int
) -> List[CityCenter]:
    """
    Generate city centers with density multipliers.

    Converts absolute peak densities from CityCentersConfig to multipliers
    relative to base densities in CityConfig.

    Args:
        centers_config: CityCentersConfig with absolute peak densities
        city_config: CityConfig with base densities
        grid_rows: Grid height
        grid_cols: Grid width

    Returns:
        List of CityCenter objects with calculated multipliers
    """
    # Place center positions
    positions = place_points(
        num_points=centers_config.num_centers,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        placement_strategy=centers_config.center_distribution,
        min_separation=centers_config.min_center_separation_blocks,
        starting_point=centers_config.starting_point
    )

    centers = []
    for i, position in enumerate(positions):
        # Apply strength decay: each subsequent center is weaker
        strength = centers_config.center_strength_decay ** i

        # Calculate housing multiplier from absolute density
        housing_peak_density = centers_config.primary_density_km2 * strength
        housing_peak_multiplier = housing_peak_density / city_config.base_housing_density_km2

        # Calculate office multiplier if configured
        office_peak_multiplier = 1.0
        if centers_config.office_density_km2 is not None:
            office_peak_density = centers_config.office_density_km2 * strength
            office_peak_multiplier = office_peak_density / city_config.base_office_density_km2

        # Calculate shop multiplier if configured
        shop_peak_multiplier = 1.0
        if centers_config.shop_density_km2 is not None:
            shop_peak_density = centers_config.shop_density_km2 * strength
            shop_peak_multiplier = shop_peak_density / city_config.base_shop_density_km2

        # Determine decay rate (can be overridden per type)
        decay_rate = centers_config.density_decay_rate

        centers.append(CityCenter(
            position=position,
            strength=strength,
            housing_peak_multiplier=housing_peak_multiplier,
            office_peak_multiplier=office_peak_multiplier,
            shop_peak_multiplier=shop_peak_multiplier,
            decay_rate=decay_rate
        ))

    return centers


def generate_parks(
    park_configs: List,
    grid_rows: int,
    grid_cols: int
) -> List[Park]:
    """
    Generate parks on the grid.

    Args:
        park_configs: List of ParkConfig objects
        grid_rows: Grid height
        grid_cols: Grid width

    Returns:
        List of Park objects
    """
    parks = []

    for park_config in park_configs:
        if park_config.num_parks == 0:
            continue

        # Place park centers
        park_positions = place_points(
            num_points=park_config.num_parks,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            placement_strategy=park_config.placement_strategy,
            min_separation=park_config.min_separation_blocks
        )

        # For each park position, determine size and blocks
        for park_center in park_positions:
            size = np.random.randint(
                park_config.min_size_blocks,
                park_config.max_size_blocks + 1
            )

            park_blocks = _get_park_blocks(
                park_center, size, park_config.shape, grid_rows, grid_cols
            )

            parks.append(Park(
                center=park_center,
                size=size,
                blocks=park_blocks,
                shape=park_config.shape
            ))

    return parks


def _get_park_blocks(
    center: Tuple[int, int],
    size: int,
    shape: str,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Get set of block positions for a park given its center and size."""
    blocks = set()
    center_y, center_x = center

    if shape == "circle":
        # Circular park
        radius = np.sqrt(size / np.pi)
        for dy in range(-int(radius) - 1, int(radius) + 2):
            for dx in range(-int(radius) - 1, int(radius) + 2):
                if dx*dx + dy*dy <= radius*radius:
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= x < grid_cols and 0 <= y < grid_rows:
                        blocks.add((y, x))
                        if len(blocks) >= size:
                            return blocks
    else:
        # Square park
        side = int(np.sqrt(size))
        for dy in range(-side // 2, side // 2 + 1):
            for dx in range(-side // 2, side // 2 + 1):
                y = center_y + dy
                x = center_x + dx
                if 0 <= x < grid_cols and 0 <= y < grid_rows:
                    blocks.add((y, x))
                    if len(blocks) >= size:
                        return blocks

    return blocks
