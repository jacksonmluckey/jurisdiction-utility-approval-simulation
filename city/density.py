"""
Density calculation functions for city generation.

This module provides pure functions for calculating density maps from city centers,
corridors, and parks. Uses a multiplier-based system where all density sources
contribute multipliers relative to base densities.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from .generation import CityCenter, TransportationCorridor, Park


@dataclass
class DensityMap:
    """
    Represents final density values for all density types across the grid.

    Attributes:
        housing_densities: Final housing density in units/km² (height, width)
        office_densities: Final office density in units/km² (height, width)
        shop_densities: Final shop density in units/km² (height, width)
        individual_multiplier_maps: Optional dict storing individual multiplier maps for debugging
        grid_rows: Grid height
        grid_cols: Grid width
    """
    housing_densities: np.ndarray
    office_densities: np.ndarray
    shop_densities: np.ndarray
    individual_multiplier_maps: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    grid_rows: int = 0
    grid_cols: int = 0

    def get_density(self, x: int, y: int, density_type: str = "housing") -> float:
        """Get density at a specific location for a specific type."""
        if density_type == "housing":
            return self.housing_densities[y, x]
        elif density_type == "office":
            return self.office_densities[y, x]
        elif density_type == "shop":
            return self.shop_densities[y, x]
        else:
            raise ValueError(f"Unknown density_type: {density_type}")

    def apply_to_grid(self, grid, config):
        """
        Apply density map to grid blocks.

        Converts densities (units/km²) to actual unit counts and population.
        Applies noise and persons_per_unit calculation.
        """
        for block in grid.blocks:
            # Get densities for this block
            housing_density = self.housing_densities[block.y, block.x]
            office_density = self.office_densities[block.y, block.x]
            shop_density = self.shop_densities[block.y, block.x]

            # Convert densities to base units
            base_units = int(housing_density * config.block_area_km2)

            # Apply noise to units if configured
            noise_value = None
            units = base_units
            if config.units_noise is not None and base_units > 0:
                if callable(config.units_noise):
                    # Function that takes base_units and returns noise value
                    noise_value = config.units_noise(base_units)
                else:
                    # Float scaling factor: std_dev = units_noise * base_units
                    import numpy as np
                    std_dev = config.units_noise * base_units
                    if std_dev > 0:  # Only add noise if std_dev is positive
                        noise_value = np.random.normal(0, std_dev)
                    else:
                        noise_value = 0
                units = max(0, int(base_units + noise_value))

            block.units = units
            block.offices = int(office_density * config.block_area_km2)
            block.shops = int(shop_density * config.block_area_km2)

            # Calculate population
            if callable(config.persons_per_unit):
                block.population = int(block.units * config.persons_per_unit(block.units, noise_value))
            else:
                block.population = int(block.units * config.persons_per_unit)


def calculate_center_multiplier_map(
    center: CityCenter,
    grid_rows: int,
    grid_cols: int,
    density_type: str
) -> np.ndarray:
    """
    Calculate density multiplier map from ONE city center for ONE density type.

    Uses exponential decay: multiplier(d) = max(peak_multiplier * exp(-decay_rate * d), 1.0)

    Multipliers are clipped at 1.0 minimum, so centers only provide positive boosts.
    Being far from a center never reduces density below the base level.

    Args:
        center: CityCenter object
        grid_rows: Grid height
        grid_cols: Grid width
        density_type: "housing", "office", or "shop"

    Returns:
        2D array of multiplier values (all values >= 1.0)
    """
    # Get appropriate peak multiplier based on density type
    if density_type == "housing":
        peak_multiplier = center.housing_peak_multiplier
    elif density_type == "office":
        peak_multiplier = center.office_peak_multiplier
    elif density_type == "shop":
        peak_multiplier = center.shop_peak_multiplier
    else:
        raise ValueError(f"Unknown density_type: {density_type}")

    # Create distance grid
    center_y, center_x = center.position
    y_coords, x_coords = np.ogrid[0:grid_rows, 0:grid_cols]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

    # Apply exponential decay and clip at 1.0 minimum
    # Centers only provide positive boosts - being far away doesn't reduce density below base
    multipliers = peak_multiplier * np.exp(-center.decay_rate * distances)

    return np.maximum(multipliers, 1.0)


def calculate_corridor_multiplier_map(
    corridor: TransportationCorridor,
    grid_rows: int,
    grid_cols: int,
    density_type: str
) -> np.ndarray:
    """
    Calculate density multiplier map from ONE corridor for ONE density type.

    Corridor blocks get the multiplier value, non-corridor blocks get 1.0 (neutral).

    Args:
        corridor: TransportationCorridor object
        grid_rows: Grid height
        grid_cols: Grid width
        density_type: "housing", "office", or "shop"

    Returns:
        2D array of multiplier values (corridor blocks = multiplier, others = 1.0)
    """
    # Get appropriate multiplier based on density type
    if density_type == "housing":
        multiplier = corridor.housing_multiplier
    elif density_type == "office":
        multiplier = corridor.office_multiplier
    elif density_type == "shop":
        multiplier = corridor.shop_multiplier
    else:
        raise ValueError(f"Unknown density_type: {density_type}")

    # Initialize all blocks to 1.0 (neutral)
    multiplier_map = np.ones((grid_rows, grid_cols))

    # Set corridor blocks to the multiplier value
    for y, x in corridor.blocks:
        multiplier_map[y, x] = multiplier

    return multiplier_map


def combine_multiplier_maps(
    multiplier_maps: List[np.ndarray],
    method: str
) -> np.ndarray:
    """
    Combine multiple multiplier maps into a single multiplier map.

    Args:
        multiplier_maps: List of 2D multiplier arrays
        method: "additive", "multiplicative", or "max"

    Returns:
        Single combined 2D multiplier array
    """
    if not multiplier_maps:
        raise ValueError("Cannot combine empty list of multiplier maps")

    if method == "additive":
        # Additive: sum(M - 1) + 1
        # This makes 1.0 neutral: [1.1, 1.2, 1.0] → 0.1 + 0.2 + 0.0 + 1 = 1.3
        deviations = [m - 1.0 for m in multiplier_maps]
        result = np.sum(deviations, axis=0) + 1.0
        # Ensure non-negative multipliers
        return np.maximum(result, 0.0)

    elif method == "multiplicative":
        # Multiplicative: M1 * M2 * M3
        # This makes 1.0 neutral: [1.1, 1.2, 1.0] → 1.32
        result = multiplier_maps[0].copy()
        for m in multiplier_maps[1:]:
            result *= m
        return result

    elif method == "max":
        # Max: max(M1, M2, M3)
        # This makes 1.0 a valid value: [1.1, 1.2, 1.0] → 1.2
        return np.maximum.reduce(multiplier_maps)

    else:
        raise ValueError(f"Unknown combination method: {method}")


def create_density_map(
    centers: List[CityCenter],
    corridors: List[TransportationCorridor],
    parks: List[Park],
    config,
    combination_method: str
) -> DensityMap:
    """
    Create final density map for all density types.

    Process:
    1. For each density type (housing, office, shop):
       - Create multiplier maps from all centers
       - Create multiplier maps from all corridors
       - Combine all multiplier maps using the specified method
       - Multiply base density by combined multiplier
    2. Zero out park blocks
    3. Return DensityMap

    Args:
        centers: List of CityCenter objects
        corridors: List of TransportationCorridor objects
        parks: List of Park objects
        config: CityConfig with base densities and grid dimensions
        combination_method: "additive", "multiplicative", or "max"

    Returns:
        DensityMap object with final densities for all types
    """
    grid_rows = config.height
    grid_cols = config.width

    # Dictionary to store individual maps for debugging (optional)
    individual_maps = {
        "housing": {},
        "office": {},
        "shop": {}
    }

    # Process each density type
    density_arrays = {}
    for density_type in ["housing", "office", "shop"]:
        # Get base density
        if density_type == "housing":
            base_density = config.base_housing_density_km2
        elif density_type == "office":
            base_density = config.base_office_density_km2
        elif density_type == "shop":
            base_density = config.base_shop_density_km2

        # Collect all multiplier maps for this density type
        multiplier_maps = []

        # Add center multiplier maps
        for i, center in enumerate(centers):
            center_map = calculate_center_multiplier_map(
                center, grid_rows, grid_cols, density_type
            )
            multiplier_maps.append(center_map)
            individual_maps[density_type][f"center_{i}"] = center_map

        # Add corridor multiplier maps
        for i, corridor in enumerate(corridors):
            corridor_map = calculate_corridor_multiplier_map(
                corridor, grid_rows, grid_cols, density_type
            )
            multiplier_maps.append(corridor_map)
            individual_maps[density_type][f"corridor_{i}"] = corridor_map

        # Combine all multiplier maps
        if multiplier_maps:
            combined_multiplier = combine_multiplier_maps(multiplier_maps, combination_method)
        else:
            # No centers or corridors - use neutral multiplier
            combined_multiplier = np.ones((grid_rows, grid_cols))

        # Calculate final density
        final_density = base_density * combined_multiplier

        density_arrays[density_type] = final_density

    # Zero out park blocks for all density types
    for park in parks:
        for y, x in park.blocks:
            density_arrays["housing"][y, x] = 0.0
            density_arrays["office"][y, x] = 0.0
            density_arrays["shop"][y, x] = 0.0

    return DensityMap(
        housing_densities=density_arrays["housing"],
        office_densities=density_arrays["office"],
        shop_densities=density_arrays["shop"],
        individual_multiplier_maps=individual_maps,
        grid_rows=grid_rows,
        grid_cols=grid_cols
    )
