"""
City generation function.

This module provides the generate_city() function which is the main entry point
for creating fully-generated cities.
"""
import numpy as np
from typing import Optional, List, Union
from .city import City, CityConfig, ParkConfig
from .city_centers import CityCentersConfig
from .transportation_corridor import TransportationConfig
from .zoning import ZoningConfig, generate_zoning
from .generation import (
    generate_city_centers,
    generate_transportation_corridors,
    generate_parks as gen_parks
)
from .density import create_density_map


def generate_city(
    config: Optional[CityConfig] = None,
    centers_config: Optional[CityCentersConfig] = None,
    transport_configs: Optional[List[TransportationConfig]] = None,
    park_configs: Optional[Union[ParkConfig, List[ParkConfig]]] = None,
    zoning_config: Optional[ZoningConfig] = None
) -> City:
    """
    Generate a complete city with all configured features.

    This is the main entry point for city generation. It creates a City instance
    and populates it with centers, corridors, parks, density, and zoning.

    Args:
        config: City-wide configuration (dimensions, block size, max density, etc.)
        centers_config: Configuration for city centers density patterns
        transport_configs: List of transportation corridor configurations
        park_configs: Configuration(s) for park generation (single ParkConfig or list)
        zoning_config: Configuration for zoning generation

    Returns:
        Fully generated City object

    Example:
        >>> from city import generate_city, CityConfig, CityCentersConfig
        >>> config = CityConfig(width=50, height=50)
        >>> centers_config = CityCentersConfig(num_centers=3)
        >>> city = generate_city(config=config, centers_config=centers_config)
        >>> city.summary()
        >>> city.visualize()
    """
    # Create city instance
    city = City(config, centers_config, transport_configs, park_configs, zoning_config)

    # Generate city components as objects (don't modify grid yet)
    if city.centers_config:
        city.centers = generate_city_centers(
            city.centers_config,
            city.config,
            city.config.height,
            city.config.width
        )
    else:
        # No centers - will use uniform density
        city.centers = []

    # Generate transportation corridors (if configured)
    if city.transport_configs and city.centers:
        city.corridors = generate_transportation_corridors(
            city.centers,
            city.transport_configs,
            city.config.height,
            city.config.width
        )
    else:
        city.corridors = []

    # Generate parks (if configured)
    if city.park_configs:
        city.parks = gen_parks(
            city.park_configs,
            city.config.height,
            city.config.width
        )
    else:
        city.parks = []

    # Create density map from all components
    if city.centers:
        city.density_map = create_density_map(
            city.centers,
            city.corridors,
            city.parks,
            city.config,
            city.config.density_combination_method
        )
        # Apply density map to grid
        city.density_map.apply_to_grid(city.grid, city.config)
    else:
        # No centers - use uniform density
        _generate_uniform_density(city)

    # Mark park blocks on grid (BEFORE applying constraints so they can be skipped)
    for park in city.parks:
        for y, x in park.blocks:
            block = city.grid.get_block(x, y)
            if block:
                block.is_park = True

    # Apply city-wide density constraints (will skip park blocks)
    _apply_density_constraints(city)

    # Generate zoning (after parks, using centers and density info)
    if city.zoning_config.enabled:
        generate_zoning(city.grid, city.centers, city.zoning_config)

    city._generated = True
    return city


def _generate_uniform_density(city: City):
    """Generate uniform density across the city (fallback if no centers config)"""
    default_density = 2471.0  # units per km²

    for block in city.grid.blocks:
        base_units = int(default_density * city.config.block_area_km2)

        # Apply noise to units if configured
        noise_value = None
        units = base_units
        if city.config.units_noise is not None:
            if callable(city.config.units_noise):
                # Function that takes base_units and returns noise value
                noise_value = city.config.units_noise(base_units)
            else:
                # Float scaling factor: std_dev = units_noise * base_units
                std_dev = city.config.units_noise * base_units
                noise_value = np.random.normal(0, std_dev)
            units = max(0, int(base_units + noise_value))

        block.units = units

        # Calculate population based on whether persons_per_unit is callable
        if callable(city.config.persons_per_unit):
            block.population = units * city.config.persons_per_unit(units, noise_value)
        else:
            block.population = units * city.config.persons_per_unit


def _apply_density_constraints(city: City):
    """Apply city-wide density constraints to all blocks"""
    max_units_per_block = int(city.config.max_density_units_per_km2 *
                               city.config.block_area_km2)
    min_units_per_block = int(city.config.min_density_units_per_km2 *
                               city.config.block_area_km2)

    for block in city.grid.blocks:
        # Skip park blocks
        if hasattr(block, 'is_park') and block.is_park:
            continue

        # Apply maximum density constraint
        if block.units > max_units_per_block:
            block.units = max_units_per_block
            # Recalculate population based on new units
            if callable(city.config.persons_per_unit):
                block.population = block.units * city.config.persons_per_unit(block.units, None)
            else:
                block.population = block.units * city.config.persons_per_unit

        # Apply minimum density constraint
        if block.units < min_units_per_block:
            block.units = min_units_per_block
            # Recalculate population based on new units
            if callable(city.config.persons_per_unit):
                block.population = block.units * city.config.persons_per_unit(block.units, None)
            else:
                block.population = block.units * city.config.persons_per_unit
