import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class AmenityCounts:
    """
    Represents counts of nearby amenities for a block.

    Attributes:
        units: Weighted count of nearby housing units
        shops: Weighted count of nearby shops
        offices: Weighted count of nearby offices
        parks: Weighted count of nearby parks
        raw_units: Unweighted count of nearby housing units
        raw_shops: Unweighted count of nearby shops
        raw_offices: Unweighted count of nearby offices
        raw_parks: Unweighted count of nearby parks
    """
    units: float
    shops: float
    offices: float
    parks: float
    raw_units: int
    raw_shops: int
    raw_offices: int
    raw_parks: int


def calculate_distance(block1, block2) -> float:
    """
    Calculate Euclidean distance between two blocks.

    Args:
        block1: First Block object
        block2: Second Block object

    Returns:
        Euclidean distance in block units
    """
    return np.sqrt((block1.x - block2.x)**2 + (block1.y - block2.y)**2)


def exponential_decay(distance: float, decay_rate: float = 0.5) -> float:
    """
    Calculate exponential decay weight based on distance.

    Args:
        distance: Distance in block units
        decay_rate: Rate of decay (higher = faster decay)

    Returns:
        Weight between 0 and 1
    """
    return np.exp(-decay_rate * distance)


def inverse_distance_decay(distance: float, power: float = 1.0) -> float:
    """
    Calculate inverse distance decay weight.

    Args:
        distance: Distance in block units
        power: Power for inverse distance (1.0 = linear, 2.0 = squared)

    Returns:
        Weight (approaches infinity as distance approaches 0)
    """
    if distance == 0:
        return 1.0
    return 1.0 / (distance ** power)


def gaussian_decay(distance: float, sigma: float = 2.0) -> float:
    """
    Calculate Gaussian decay weight based on distance.

    Args:
        distance: Distance in block units
        sigma: Standard deviation of the Gaussian

    Returns:
        Weight between 0 and 1
    """
    return np.exp(-(distance**2) / (2 * sigma**2))


def count_nearby_amenities(
    block,
    grid,
    search_radius: float = 10.0,
    decay_function: Optional[Callable[[float], float]] = None,
    decay_params: Optional[Dict] = None,
    include_self: bool = False
) -> AmenityCounts:
    """
    Count amenities within a search radius of a block with distance-based weighting.

    Uses a bounding box optimization to avoid unnecessary distance calculations.

    Args:
        block: The Block object to search around
        grid: The Grid object containing all blocks
        search_radius: Maximum distance to search (in block units)
        decay_function: Function to calculate distance weights. Options:
            - None: No decay, uniform weighting (default)
            - exponential_decay: Exponential decay
            - inverse_distance_decay: Inverse distance weighting
            - gaussian_decay: Gaussian decay
            - Custom function taking distance as input
        decay_params: Dictionary of parameters to pass to decay_function
        include_self: Whether to include the block itself in counts

    Returns:
        AmenityCounts object with weighted and raw counts

    Example:
        >>> from city import Grid
        >>> from market import count_nearby_amenities, exponential_decay
        >>> grid = Grid(width=20, height=20)
        >>> block = grid.blocks[0]
        >>> # Count with exponential decay
        >>> amenities = count_nearby_amenities(
        ...     block, grid,
        ...     search_radius=5.0,
        ...     decay_function=exponential_decay,
        ...     decay_params={'decay_rate': 0.3}
        ... )
        >>> print(f"Weighted shops nearby: {amenities.shops:.2f}")
        >>> print(f"Raw shops nearby: {amenities.raw_shops}")
        >>>
        >>> # Count with Gaussian decay
        >>> amenities = count_nearby_amenities(
        ...     block, grid,
        ...     search_radius=8.0,
        ...     decay_function=gaussian_decay,
        ...     decay_params={'sigma': 2.5}
        ... )
        >>>
        >>> # Count without decay (uniform weighting)
        >>> amenities = count_nearby_amenities(block, grid, search_radius=5.0)
    """
    if decay_params is None:
        decay_params = {}

    # Initialize counters
    weighted_units = 0.0
    weighted_shops = 0.0
    weighted_offices = 0.0
    weighted_parks = 0.0

    raw_units = 0
    raw_shops = 0
    raw_offices = 0
    raw_parks = 0

    # Define bounding box for search
    # Only consider blocks where both x and y are within search_radius
    x_min = max(0, int(np.floor(block.x - search_radius)))
    x_max = min(grid.width - 1, int(np.ceil(block.x + search_radius)))
    y_min = max(0, int(np.floor(block.y - search_radius)))
    y_max = min(grid.height - 1, int(np.ceil(block.y + search_radius)))

    # Iterate only through blocks in the bounding box
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            other_block = grid.get_block(x, y)

            if other_block is None:
                continue

            # Skip self if requested
            if not include_self and other_block.block_id == block.block_id:
                continue

            # Calculate distance
            distance = calculate_distance(block, other_block)

            # Skip blocks outside search radius (circular constraint within bounding box)
            if distance > search_radius:
                continue

            # Calculate weight
            if decay_function is None:
                # No decay - uniform weighting
                weight = 1.0
            else:
                # Apply decay function
                weight = decay_function(distance, **decay_params)

            # Count housing units
            if other_block.units > 0:
                weighted_units += other_block.units * weight
                raw_units += other_block.units

            # Count shops
            if other_block.shops > 0:
                weighted_shops += other_block.shops * weight
                raw_shops += other_block.shops

            # Count offices
            if other_block.offices > 0:
                weighted_offices += other_block.offices * weight
                raw_offices += other_block.offices

            # Count parks
            if other_block.is_park:
                weighted_parks += 1.0 * weight
                raw_parks += 1

    return AmenityCounts(
        units=weighted_units,
        shops=weighted_shops,
        offices=weighted_offices,
        parks=weighted_parks,
        raw_units=raw_units,
        raw_shops=raw_shops,
        raw_offices=raw_offices,
        raw_parks=raw_parks
    )
