import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, Optional, Callable, Literal
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


def visualize_search_area(
    block,
    grid,
    search_radius: float = 10.0,
    decay_function: Optional[Callable[[float], float]] = None,
    decay_params: Optional[Dict] = None,
    use_weighting: bool = False,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None
):
    """
    Visualize the blocks included in a search from a particular block.

    Args:
        block: The Block object to search around
        grid: The Grid object containing all blocks
        search_radius: Maximum distance to search (in block units)
        decay_function: If None, weighting is not used. Accepts a function to calculate distance weights. 
        decay_params: Dictionary of parameters to pass to decay_function
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Example:
        >>> from city import Grid
        >>> from market import visualize_search_area, exponential_decay
        >>> grid = Grid(width=20, height=20)
        >>> block = grid.blocks[210]
        >>> # Show blocks included without weighting
        >>> visualize_search_area(block, grid, search_radius=5.0, use_weighting=False)
        >>> # Show blocks with weighting visualization
        >>> visualize_search_area(
        ...     block, grid, search_radius=5.0,
        ...     decay_function=exponential_decay,
        ...     decay_params={'decay_rate': 0.5},
        ...     use_weighting=True
        ... )
    """
    if decay_params is None:
        decay_params = {}

    if decay_function is None:
        use_weighting = False
    else:
        use_weighting = True

    # Create grid to store weights or inclusion
    weight_grid = np.zeros((grid.height, grid.width))

    # Define bounding box for search
    x_min = max(0, int(np.floor(block.x - search_radius)))
    x_max = min(grid.width - 1, int(np.ceil(block.x + search_radius)))
    y_min = max(0, int(np.floor(block.y - search_radius)))
    y_max = min(grid.height - 1, int(np.ceil(block.y + search_radius)))

    # Calculate weights or inclusion for each block
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            other_block = grid.get_block(x, y)
            if other_block is None:
                continue

            distance = calculate_distance(block, other_block)

            if distance <= search_radius:
                if use_weighting:
                    if decay_function is None:
                        weight = 1.0
                    else:
                        weight = decay_function(distance, **decay_params)
                    weight_grid[y, x] = weight
                else:
                    weight_grid[y, x] = 1.0

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    if use_weighting:
        im = ax.imshow(weight_grid, cmap='YlOrRd', origin='lower', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Weight', rotation=270, labelpad=20, fontsize=12)
        title = f'Search Area with Weighting (radius={search_radius:.1f})'
    else:
        im = ax.imshow(weight_grid, cmap='Greys', origin='lower', interpolation='nearest', vmin=0, vmax=1)
        title = f'Search Area Coverage (radius={search_radius:.1f})'

    # Mark the center block
    ax.plot(block.x, block.y, 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1.5,
            label='Search Center')

    # Draw search radius circle
    circle = Circle((block.x, block.y), search_radius, fill=False, edgecolor='blue',
                    linewidth=2, linestyle='--', label=f'Radius={search_radius:.1f}')
    ax.add_patch(circle)

    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_xlabel('X Coordinate (blocks)', fontsize=12)
    ax.set_ylabel('Y Coordinate (blocks)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_amenity_counts_single_block(
    block,
    grid,
    amenity_type: Literal['units', 'shops', 'offices', 'parks'] = 'units',
    search_radius: float = 10.0,
    decay_function: Optional[Callable[[float], float]] = None,
    decay_params: Optional[Dict] = None,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None
):
    """
    Visualize amenities found in the search area around a particular block.

    Shows the search area with blocks colored by their amenity counts.

    Args:
        block: The Block object to search around
        grid: The Grid object containing all blocks
        amenity_type: Which amenity to visualize ('units', 'shops', 'offices', 'parks')
        search_radius: Maximum distance to search (in block units)
        decay_function: If None, weighting is not used. Accepts a function to calculate distance weights. 
        decay_params: Dictionary of parameters to pass to decay_function
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Example:
        >>> from city import Grid
        >>> from market import visualize_amenity_counts_single_block, gaussian_decay
        >>> grid = Grid(width=20, height=20)
        >>> block = grid.blocks[210]
        >>> # Show raw unit counts in search area
        >>> visualize_amenity_counts_single_block(
        ...     block, grid, amenity_type='units',
        ...     search_radius=5.0, use_weighting=False
        ... )
        >>> # Show weighted shop counts in search area
        >>> visualize_amenity_counts_single_block(
        ...     block, grid, amenity_type='shops',
        ...     search_radius=5.0,
        ...     decay_function=gaussian_decay,
        ...     decay_params={'sigma': 2.0},
        ...     use_weighting=True
        ... )
    """
    if decay_params is None:
        decay_params = {}

    if decay_function is None:
        use_weighting = False
    else:
        use_weighting = True

    # Create grid to store amenity counts
    amenity_grid = np.zeros((grid.height, grid.width))

    # Define bounding box for search
    x_min = max(0, int(np.floor(block.x - search_radius)))
    x_max = min(grid.width - 1, int(np.ceil(block.x + search_radius)))
    y_min = max(0, int(np.floor(block.y - search_radius)))
    y_max = min(grid.height - 1, int(np.ceil(block.y + search_radius)))

    # Iterate through blocks in the search area
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            other_block = grid.get_block(x, y)
            if other_block is None:
                continue

            distance = calculate_distance(block, other_block)

            if distance <= search_radius:
                # Get the amenity count for this block
                if amenity_type == 'units':
                    count = other_block.units
                elif amenity_type == 'shops':
                    count = other_block.shops
                elif amenity_type == 'offices':
                    count = other_block.offices
                elif amenity_type == 'parks':
                    count = 1 if other_block.is_park else 0

                # Apply weighting if requested
                if use_weighting and count > 0:
                    if decay_function is None:
                        weight = 1.0
                    else:
                        weight = decay_function(distance, **decay_params)
                    amenity_grid[y, x] = count * weight
                else:
                    amenity_grid[y, x] = count

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Choose colormap based on amenity type
    cmaps = {
        'units': 'Blues',
        'shops': 'Reds',
        'offices': 'Oranges',
        'parks': 'Greens'
    }
    cmap = cmaps.get(amenity_type, 'viridis')

    im = ax.imshow(amenity_grid, cmap=cmap, origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    count_type = 'Weighted' if use_weighting else 'Raw'
    cbar.set_label(f'{count_type} {amenity_type.capitalize()} Count',
                   rotation=270, labelpad=20, fontsize=12)

    # Mark the center block
    ax.plot(block.x, block.y, 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1.5,
            label='Search Center')

    # Draw search radius circle
    circle = Circle((block.x, block.y), search_radius, fill=False, edgecolor='blue',
                    linewidth=2, linestyle='--', label=f'Radius={search_radius:.1f}')
    ax.add_patch(circle)

    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(-0.5, grid.height - 0.5)
    ax.set_xlabel('X Coordinate (blocks)', fontsize=12)
    ax.set_ylabel('Y Coordinate (blocks)', fontsize=12)
    ax.set_title(f'{count_type} {amenity_type.capitalize()} in Search Area\n'
                 f'Center: ({block.x}, {block.y}), Radius: {search_radius:.1f} blocks',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5, color='white')
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_amenity_counts_all_blocks(
    grid,
    amenity_type: Literal['units', 'shops', 'offices', 'parks'] = 'units',
    search_radius: float = 10.0,
    decay_function: Optional[Callable[[float], float]] = None,
    decay_params: Optional[Dict] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
):
    """
    Visualize amenity counts for all blocks in the city.

    Args:
        grid: The Grid object containing all blocks
        amenity_type: Which amenity to visualize ('units', 'shops', 'offices', 'parks')
        search_radius: Maximum distance to search (in block units)
        decay_function: If None, weighting is not used. Accepts a function to calculate distance weights.
        decay_params: Dictionary of parameters to pass to decay_function
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Example:
        >>> from city import Grid
        >>> from market import visualize_amenity_counts_all_blocks, exponential_decay
        >>> grid = Grid(width=30, height=30)
        >>> # Show raw unit counts for all blocks
        >>> visualize_amenity_counts_all_blocks(
        ...     grid, amenity_type='units',
        ...     search_radius=5.0, use_weighting=False
        ... )
        >>> # Show weighted shop counts for all blocks
        >>> visualize_amenity_counts_all_blocks(
        ...     grid, amenity_type='shops',
        ...     search_radius=5.0,
        ...     decay_function=exponential_decay,
        ...     decay_params={'decay_rate': 0.5},
        ...     use_weighting=True
        ... )
    """
    if decay_params is None:
        decay_params = {}

    if decay_function is None:
        use_weighting = False
    else:
        use_weighting = True

    # Create grid to store counts
    count_grid = np.zeros((grid.height, grid.width))

    # Calculate amenity counts for each block
    for block in grid.blocks:
        amenities = count_nearby_amenities(
            block, grid,
            search_radius=search_radius,
            decay_function=decay_function,
            decay_params=decay_params,
            include_self=False
        )

        # Select the appropriate count based on amenity type and weighting
        if use_weighting:
            if amenity_type == 'units':
                count = amenities.units
            elif amenity_type == 'shops':
                count = amenities.shops
            elif amenity_type == 'offices':
                count = amenities.offices
            elif amenity_type == 'parks':
                count = amenities.parks
        else:
            if amenity_type == 'units':
                count = amenities.raw_units
            elif amenity_type == 'shops':
                count = amenities.raw_shops
            elif amenity_type == 'offices':
                count = amenities.raw_offices
            elif amenity_type == 'parks':
                count = amenities.raw_parks

        count_grid[block.y, block.x] = count

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Choose colormap based on amenity type
    cmaps = {
        'units': 'Blues',
        'shops': 'Reds',
        'offices': 'Oranges',
        'parks': 'Greens'
    }
    cmap = cmaps.get(amenity_type, 'viridis')

    im = ax.imshow(count_grid, cmap=cmap, origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    count_type = 'Weighted' if use_weighting else 'Raw'
    cbar.set_label(f'{count_type} {amenity_type.capitalize()} Count',
                   rotation=270, labelpad=25, fontsize=12, fontweight='bold')

    ax.set_xlabel('X Coordinate (blocks)', fontsize=12)
    ax.set_ylabel('Y Coordinate (blocks)', fontsize=12)
    ax.set_title(f'{count_type} {amenity_type.capitalize()} Counts Across City\n'
                 f'Search Radius: {search_radius:.1f} blocks',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=0.5, color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
