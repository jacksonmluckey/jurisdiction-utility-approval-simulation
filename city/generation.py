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


def generate_transportation_corridors(
    centers: List[CityCenter],
    transport_configs: List,
    grid_rows: int,
    grid_cols: int
) -> List[TransportationCorridor]:
    """
    Generate transportation corridors from configuration.

    Args:
        centers: List of CityCenter objects
        transport_configs: List of TransportationConfig objects
        grid_rows: Grid height
        grid_cols: Grid width

    Returns:
        List of TransportationCorridor objects
    """
    corridors = []

    # Import here to avoid circular dependency
    from .transportation_corridor import CorridorType

    # Convert centers to dict format for compatibility with corridor generation logic
    center_positions = [(c.position[0], c.position[1]) for c in centers]

    for config in transport_configs:
        # Generate blocks for this corridor configuration
        corridor_blocks = _generate_corridor_blocks(
            config, center_positions, grid_rows, grid_cols
        )

        # Create TransportationCorridor object
        # Get multipliers from config
        # housing_multiplier uses density_multiplier (backwards compatible)
        housing_mult = getattr(config, 'housing_multiplier', config.density_multiplier)
        # office and shop default to 1.0 (neutral) if not explicitly set
        office_mult = config.office_multiplier if config.office_multiplier is not None else 1.0
        shop_mult = config.shop_multiplier if config.shop_multiplier is not None else 1.0

        corridor = TransportationCorridor(
            corridor_type=config.corridor_type.value if hasattr(config.corridor_type, 'value') else str(config.corridor_type),
            blocks=corridor_blocks,
            housing_multiplier=housing_mult,
            office_multiplier=office_mult,
            shop_multiplier=shop_mult,
            width_blocks=config.corridor_width_blocks
        )
        corridors.append(corridor)

    return corridors


def _generate_corridor_blocks(
    config,
    center_positions: List[Tuple[int, int]],
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Generate corridor blocks for a single configuration."""
    from .transportation_corridor import CorridorType

    blocks = set()

    if config.corridor_type == CorridorType.RADIAL:
        blocks = _generate_radial_blocks(
            center_positions, config, grid_rows, grid_cols
        )
    elif config.corridor_type == CorridorType.INTER_CENTER:
        blocks = _generate_inter_center_blocks(
            center_positions, config, grid_rows, grid_cols
        )
    elif config.corridor_type == CorridorType.RING:
        blocks = _generate_ring_blocks(
            center_positions, config, grid_rows, grid_cols
        )
    elif config.corridor_type == CorridorType.GRID:
        blocks = _generate_grid_blocks(
            config, grid_rows, grid_cols
        )

    if config.include_ring_roads and center_positions:
        ring_blocks = _add_ring_at_radius(
            center_positions[0],
            config.ring_road_radius_blocks,
            config.corridor_width_blocks,
            grid_rows,
            grid_cols
        )
        blocks.update(ring_blocks)

    return blocks


def _generate_radial_blocks(
    center_positions: List[Tuple[int, int]],
    config,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Generate radial corridor blocks from primary center."""
    if not center_positions:
        return set()

    blocks = set()
    primary_center = center_positions[0]
    angles = np.linspace(0, 2*np.pi, config.radial_corridors_count, endpoint=False)
    max_radius = max(grid_rows, grid_cols)

    for angle in angles:
        for r in range(max_radius):
            row = int(primary_center[0] + r * np.sin(angle))
            col = int(primary_center[1] + r * np.cos(angle))

            if 0 <= row < grid_rows and 0 <= col < grid_cols:
                blocks.update(_add_block_with_width(
                    row, col, config.corridor_width_blocks, grid_rows, grid_cols
                ))

    return blocks


def _generate_inter_center_blocks(
    center_positions: List[Tuple[int, int]],
    config,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Generate corridors connecting centers."""
    if len(center_positions) < 2:
        return set()

    blocks = set()

    if config.connect_all_centers:
        # Connect all pairs of centers
        for i in range(len(center_positions)):
            for j in range(i + 1, len(center_positions)):
                blocks.update(_connect_two_points(
                    center_positions[i],
                    center_positions[j],
                    config.corridor_width_blocks,
                    config.max_corridor_distance,
                    grid_rows,
                    grid_cols
                ))
    else:
        # Connect each center to nearest neighbor
        for i, center in enumerate(center_positions):
            if i == 0:
                continue
            # Find nearest center
            distances = [
                np.sqrt((center[0] - other[0])**2 + (center[1] - other[1])**2)
                for j, other in enumerate(center_positions) if j != i
            ]
            nearest_idx = np.argmin(distances)
            if nearest_idx >= i:
                nearest_idx += 1
            blocks.update(_connect_two_points(
                center,
                center_positions[nearest_idx],
                config.corridor_width_blocks,
                config.max_corridor_distance,
                grid_rows,
                grid_cols
            ))

    return blocks


def _generate_ring_blocks(
    center_positions: List[Tuple[int, int]],
    config,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Generate concentric ring corridors."""
    if not center_positions:
        return set()

    blocks = set()
    primary_center = center_positions[0]

    # Multiple rings at different radii
    radii = [config.ring_road_radius_blocks * (i + 1)
            for i in range(min(grid_rows, grid_cols) //
                          (2 * config.ring_road_radius_blocks))]

    for radius in radii:
        blocks.update(_add_ring_at_radius(
            primary_center, radius, config.corridor_width_blocks, grid_rows, grid_cols
        ))

    return blocks


def _generate_grid_blocks(
    config,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Generate orthogonal grid corridors."""
    blocks = set()

    # Vertical corridors
    vertical_spacing = config.grid_spacing_blocks if config.grid_spacing_blocks else max(5, grid_cols // 6)
    for col in range(0, grid_cols, vertical_spacing):
        for row in range(grid_rows):
            blocks.update(_add_block_with_width(
                row, col, config.corridor_width_blocks, grid_rows, grid_cols
            ))

    # Horizontal corridors
    horizontal_spacing = config.grid_spacing_blocks if config.grid_spacing_blocks else max(5, grid_rows // 6)
    for row in range(0, grid_rows, horizontal_spacing):
        for col in range(grid_cols):
            blocks.update(_add_block_with_width(
                row, col, config.corridor_width_blocks, grid_rows, grid_cols
            ))

    return blocks


def _add_ring_at_radius(
    center: Tuple[int, int],
    radius: int,
    width: int,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Add a circular corridor at given radius from center."""
    blocks = set()
    center_row, center_col = center

    # Sample points around circle
    num_points = int(2 * np.pi * radius * 2)
    angles = np.linspace(0, 2*np.pi, num_points)

    for angle in angles:
        row = int(center_row + radius * np.sin(angle))
        col = int(center_col + radius * np.cos(angle))

        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            blocks.update(_add_block_with_width(
                row, col, width, grid_rows, grid_cols
            ))

    return blocks


def _connect_two_points(
    pos1: Tuple[int, int],
    pos2: Tuple[int, int],
    width: int,
    max_distance: Optional[int],
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Connect two points using Bresenham's line algorithm."""
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    if max_distance and distance > max_distance:
        return set()

    blocks = set()

    # Bresenham's line algorithm
    x0, y0 = pos1[1], pos1[0]  # col, row
    x1, y1 = pos2[1], pos2[0]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        blocks.update(_add_block_with_width(
            y0, x0, width, grid_rows, grid_cols
        ))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return blocks


def _add_block_with_width(
    row: int,
    col: int,
    width: int,
    grid_rows: int,
    grid_cols: int
) -> Set[Tuple[int, int]]:
    """Add a block and its neighbors based on corridor width."""
    blocks = set()
    half_width = width // 2

    for dr in range(-half_width, half_width + 1):
        for dc in range(-half_width, half_width + 1):
            new_row = row + dr
            new_col = col + dc

            if 0 <= new_row < grid_rows and 0 <= new_col < grid_cols:
                blocks.add((new_row, new_col))

    return blocks


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
