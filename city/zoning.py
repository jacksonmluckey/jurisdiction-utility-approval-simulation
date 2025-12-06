"""
Zoning system for city blocks.

Defines allowed uses and density levels for each block.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Set, Optional
import numpy as np


class Use(Enum):
    """Allowed uses for a block."""
    RESIDENTIAL = "Residential"
    OFFICE = "Office"
    COMMERCIAL = "Commercial"


class Density(Enum):
    """Density levels for zoning. Lower densities can fit in higher density zones."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    def allows(self, other: 'Density') -> bool:
        """Check if this density level allows another density level."""
        return self.value >= other.value


@dataclass
class Zoning:
    """
    Zoning information for a block.

    Attributes:
        allowed_uses: Set of allowed uses for this block
        max_density: Maximum allowed density level
    """
    allowed_uses: Set[Use] = field(default_factory=set)
    max_density: Density = Density.MEDIUM

    def allows_use(self, use: Use) -> bool:
        """Check if a use is allowed in this zone."""
        return use in self.allowed_uses

    def allows_density(self, density: Density) -> bool:
        """Check if a density level is allowed in this zone."""
        return self.max_density.allows(density)


@dataclass
class ZoningConfig:
    """
    Configuration for automatic zoning generation.

    Attributes:
        enabled: Whether to generate zoning (default: True)
        low_density_threshold: Units per block below this are zoned low density (default: 30)
        medium_density_threshold: Units per block below this are zoned medium density (default: 60)
        residential_weight: Weight for residential zoning in mixed areas (default: 1.0)
        office_weight: Weight for office zoning in high-density areas (default: 0.5)
        commercial_weight: Weight for commercial zoning (default: 0.3)
        center_radius_blocks: Radius around centers to zone for all uses (default: 3)
        auto_upzone_enabled: Whether to apply automatic upzoning based on neighbors (default: False)
        auto_upzone_density_threshold: Number of adjacent blocks with higher density needed
            to upzone this block's density (default: 3)
        auto_upzone_use_threshold: Number of adjacent blocks with an additional use needed
            to add that use to this block (default: 3)
        auto_upzone_include_diagonals: Whether to include diagonal/corner neighbors when
            counting adjacent blocks (default: True). If True, blocks have 8 neighbors;
            if False, only 4 (cardinal directions).
        auto_upzone_iterations: Number of upzoning passes to apply. Higher values allow
            upzoning to spread further from high-density areas (default: 1)
    """
    enabled: bool = True
    low_density_threshold: int = 30
    medium_density_threshold: int = 60
    residential_weight: float = 1.0
    office_weight: float = 0.5
    commercial_weight: float = 0.3
    center_radius_blocks: int = 3
    auto_upzone_enabled: bool = False
    auto_upzone_density_threshold: int = 3
    auto_upzone_use_threshold: int = 3
    auto_upzone_include_diagonals: bool = True
    auto_upzone_iterations: int = 1


def _get_neighbors(grid, block, include_diagonals: bool = True) -> List:
    """
    Get neighboring blocks for a given block.

    Args:
        grid: Grid object containing blocks
        block: The block to get neighbors for
        include_diagonals: If True, include diagonal neighbors (8 total);
                          if False, only cardinal directions (4 total)

    Returns:
        List of neighboring Block objects
    """
    neighbors = []

    # Cardinal directions (N, S, E, W)
    offsets = [
        (-1, 0),  # North
        (1, 0),   # South
        (0, 1),   # East
        (0, -1),  # West
    ]

    # Add diagonal directions if requested
    if include_diagonals:
        offsets.extend([
            (-1, -1),  # Northwest
            (-1, 1),   # Northeast
            (1, -1),   # Southwest
            (1, 1),    # Southeast
        ])

    for dy, dx in offsets:
        neighbor = grid.get_block(block.x + dx, block.y + dy)
        if neighbor is not None:
            neighbors.append(neighbor)

    return neighbors


def _apply_auto_upzone_density(grid, config: ZoningConfig) -> int:
    """
    Apply automatic density upzoning based on neighboring blocks.

    Args:
        grid: Grid object containing blocks
        config: ZoningConfig with upzoning parameters

    Returns:
        Number of blocks that were upzoned
    """
    upzoned_count = 0
    blocks_to_upzone = []  # Store changes to apply after analysis

    for block in grid.blocks:
        # Skip if no zoning or is a park
        if not hasattr(block, 'zoning') or block.zoning is None:
            continue
        if hasattr(block, 'is_park') and block.is_park:
            continue

        current_density = block.zoning.max_density

        # Can't upzone HIGH density
        if current_density == Density.HIGH:
            continue

        # Get neighbors
        neighbors = _get_neighbors(grid, block, config.auto_upzone_include_diagonals)

        # Count neighbors with higher density
        higher_density_count = 0
        target_density = None

        for neighbor in neighbors:
            if not hasattr(neighbor, 'zoning') or neighbor.zoning is None:
                continue

            neighbor_density = neighbor.zoning.max_density

            # Check if neighbor has higher density
            if neighbor_density.value > current_density.value:
                higher_density_count += 1
                # Track the target density (most common higher density)
                if target_density is None or neighbor_density.value > target_density.value:
                    target_density = neighbor_density

        # Apply upzoning if threshold met
        if higher_density_count >= config.auto_upzone_density_threshold and target_density:
            # Upzone by one level, not necessarily to the highest neighbor
            if current_density == Density.LOW:
                new_density = Density.MEDIUM
            elif current_density == Density.MEDIUM:
                new_density = Density.HIGH
            else:
                new_density = current_density

            if new_density != current_density:
                blocks_to_upzone.append((block, new_density))

    # Apply all upzoning changes
    for block, new_density in blocks_to_upzone:
        block.zoning.max_density = new_density
        upzoned_count += 1

    return upzoned_count


def _apply_auto_upzone_uses(grid, config: ZoningConfig) -> int:
    """
    Apply automatic use upzoning based on neighboring blocks.

    Args:
        grid: Grid object containing blocks
        config: ZoningConfig with upzoning parameters

    Returns:
        Number of blocks that had uses added
    """
    upzoned_count = 0
    blocks_to_upzone = []  # Store changes to apply after analysis

    for block in grid.blocks:
        # Skip if no zoning or is a park
        if not hasattr(block, 'zoning') or block.zoning is None:
            continue
        if hasattr(block, 'is_park') and block.is_park:
            continue

        current_uses = block.zoning.allowed_uses

        # Get neighbors
        neighbors = _get_neighbors(grid, block, config.auto_upzone_include_diagonals)

        # Count neighbors with each use
        use_counts = {
            Use.RESIDENTIAL: 0,
            Use.OFFICE: 0,
            Use.COMMERCIAL: 0
        }

        for neighbor in neighbors:
            if not hasattr(neighbor, 'zoning') or neighbor.zoning is None:
                continue

            for use in neighbor.zoning.allowed_uses:
                use_counts[use] += 1

        # Check which uses to add
        uses_to_add = set()
        for use, count in use_counts.items():
            # Only add if not already allowed and threshold met
            if use not in current_uses and count >= config.auto_upzone_use_threshold:
                uses_to_add.add(use)

        if uses_to_add:
            blocks_to_upzone.append((block, uses_to_add))

    # Apply all upzoning changes
    for block, new_uses in blocks_to_upzone:
        block.zoning.allowed_uses.update(new_uses)
        upzoned_count += 1

    return upzoned_count


def generate_zoning(grid, centers: List[dict], config: ZoningConfig) -> None:
    """
    Generate zoning for all blocks in the grid.

    Args:
        grid: Grid object containing blocks
        centers: List of activity centers with position information
        config: ZoningConfig with parameters for zoning generation
    """
    if not config.enabled:
        return

    # First, zone centers for all uses at high density
    for center in centers:
        center_y, center_x = center['position']
        for block in grid.blocks:
            distance = np.sqrt((block.x - center_x)**2 + (block.y - center_y)**2)
            if distance <= config.center_radius_blocks:
                block.zoning = Zoning(
                    allowed_uses={Use.RESIDENTIAL, Use.OFFICE, Use.COMMERCIAL},
                    max_density=Density.HIGH
                )

    # Then, zone remaining blocks based on density and characteristics
    for block in grid.blocks:
        # Skip if already zoned (e.g., near centers)
        if hasattr(block, 'zoning') and block.zoning is not None:
            continue

        # Skip parks
        if hasattr(block, 'is_park') and block.is_park:
            block.zoning = Zoning(allowed_uses=set(), max_density=Density.LOW)
            continue

        # Determine density level based on units
        if block.units < config.low_density_threshold:
            density = Density.LOW
        elif block.units < config.medium_density_threshold:
            density = Density.MEDIUM
        else:
            density = Density.HIGH

        # Determine allowed uses based on density and characteristics
        allowed_uses = set()

        # Residential is allowed almost everywhere
        if config.residential_weight > 0:
            allowed_uses.add(Use.RESIDENTIAL)

        # Office and commercial in medium/high density areas
        if density in [Density.MEDIUM, Density.HIGH]:
            # Use random weights to create variety
            if np.random.random() < config.commercial_weight:
                allowed_uses.add(Use.COMMERCIAL)
            if density == Density.HIGH and np.random.random() < config.office_weight:
                allowed_uses.add(Use.OFFICE)

        # Ensure at least residential is allowed if not a park
        if not allowed_uses:
            allowed_uses.add(Use.RESIDENTIAL)

        block.zoning = Zoning(allowed_uses=allowed_uses, max_density=density)

    # Apply automatic upzoning if enabled
    if config.auto_upzone_enabled:
        for iteration in range(config.auto_upzone_iterations):
            # Apply density upzoning
            density_upzoned = _apply_auto_upzone_density(grid, config)
            # Apply use upzoning
            use_upzoned = _apply_auto_upzone_uses(grid, config)

            # Stop if no changes in this iteration
            if density_upzoned == 0 and use_upzoned == 0:
                break


def get_zoning_summary(grid) -> dict:
    """
    Get summary statistics about zoning in the grid.

    Args:
        grid: Grid object containing blocks

    Returns:
        Dictionary with zoning statistics
    """
    total_blocks = len(grid.blocks)

    # Count by density
    density_counts = {
        Density.LOW: 0,
        Density.MEDIUM: 0,
        Density.HIGH: 0
    }

    # Count by use
    use_counts = {
        Use.RESIDENTIAL: 0,
        Use.OFFICE: 0,
        Use.COMMERCIAL: 0
    }

    # Count mixed-use blocks
    mixed_use_count = 0

    for block in grid.blocks:
        if not hasattr(block, 'zoning') or block.zoning is None:
            continue

        # Count density
        density_counts[block.zoning.max_density] += 1

        # Count uses
        for use in block.zoning.allowed_uses:
            use_counts[use] += 1

        # Count mixed-use (more than one use)
        if len(block.zoning.allowed_uses) > 1:
            mixed_use_count += 1

    return {
        'total_blocks': total_blocks,
        'density_counts': density_counts,
        'use_counts': use_counts,
        'mixed_use_count': mixed_use_count,
        'density_percentages': {
            d: (count / total_blocks * 100) for d, count in density_counts.items()
        },
        'use_percentages': {
            u: (count / total_blocks * 100) for u, count in use_counts.items()
        }
    }
