"""
City class - unified interface for creating and managing simulated cities
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from .grid import Grid
from .polycentric_city import PolycentricConfig
from .transportation_corridor import TransportationConfig, TransportationNetwork


@dataclass
class CityConfig:
    """Configuration parameters for city-wide properties"""
    # Grid dimensions
    width: int = 50
    height: int = 50

    # Block physical properties
    block_size_meters: float = 100.0  # Side length of each block in meters
    block_area_acres: float = 2.47  # Area of each block (100m x 100m ≈ 2.47 acres)

    # Density constraints
    max_density_units_per_acre: float = 50.0  # Maximum housing units per acre
    min_density_units_per_acre: float = 0.5   # Minimum housing units per acre

    # Population parameters
    persons_per_unit: float = 2.5  # Average household size

    # Random seed for reproducibility
    random_seed: Optional[int] = None


class City:
    """
    Main city class that integrates grid, polycentric density, and transportation.

    This class serves as the primary interface for creating and managing simulated cities.
    It handles:
    - Grid creation and management
    - Polycentric density patterns
    - Transportation network generation
    - City-wide parameters (block size, max density, etc.)

    Example:
        >>> config = CityConfig(width=100, height=100, max_density_units_per_acre=60)
        >>> poly_config = PolycentricConfig(num_centers=5, primary_density=25.0)
        >>> transport = TransportationConfig(corridor_type=CorridorType.INTER_CENTER)
        >>> city = City(config, poly_config, transport)
        >>> grid = city.generate()
        >>> city.visualize()
    """

    def __init__(self,
                 config: Optional[CityConfig] = None,
                 polycentric_config: Optional[PolycentricConfig] = None,
                 transport_config: Optional[TransportationConfig] = None):
        """
        Initialize a city with configuration parameters.

        Args:
            config: City-wide configuration (dimensions, block size, max density, etc.)
            polycentric_config: Configuration for polycentric density patterns
            transport_config: Configuration for transportation corridors
        """
        self.config = config or CityConfig()
        self.polycentric_config = polycentric_config
        self.transport_config = transport_config

        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Initialize grid
        self.grid = Grid(width=self.config.width, height=self.config.height)

        # Components (populated during generation)
        self.centers = []
        self.transport_network = None

        # Track if city has been generated
        self._generated = False

    def generate(self) -> Grid:
        """
        Generate the city with all configured features.

        Returns:
            Grid object with populated blocks
        """
        # Generate polycentric density pattern
        if self.polycentric_config:
            self._generate_polycentric_density()
        else:
            self._generate_uniform_density()

        # Add transportation corridors
        if self.transport_config and self.centers:
            self._generate_transportation_network()

        # Apply city-wide constraints
        self._apply_density_constraints()

        self._generated = True
        return self.grid

    def _generate_polycentric_density(self):
        """Generate density using polycentric model"""
        # Update polycentric config with city parameters
        if self.polycentric_config.block_area_acres != self.config.block_area_acres:
            self.polycentric_config.block_area_acres = self.config.block_area_acres
        if self.polycentric_config.persons_per_unit != self.config.persons_per_unit:
            self.polycentric_config.persons_per_unit = self.config.persons_per_unit

        # Place centers
        self._place_centers()

        # Calculate densities based on distance from centers
        self._calculate_densities()

    def _place_centers(self):
        """Place employment/activity centers based on distribution strategy"""
        from .polycentric_city import PolycentricCity

        # Use PolycentricCity's center placement logic
        temp_city = PolycentricCity(
            grid_rows=self.config.height,
            grid_cols=self.config.width,
            config=self.polycentric_config
        )
        temp_city._place_centers()
        self.centers = temp_city.centers

    def _calculate_densities(self):
        """Calculate housing units and population using additive exponential model"""
        for row in range(self.config.height):
            for col in range(self.config.width):
                block_density = 0.0

                # Sum contributions from all centers
                for center in self.centers:
                    distance = self._calculate_distance(
                        (row, col),
                        center['position']
                    )

                    # Exponential decay: D = D0 * exp(-b * distance)
                    contribution = center['peak_density'] * np.exp(
                        -self.polycentric_config.density_decay_rate * distance
                    )
                    block_density += contribution

                # Convert density to housing units
                units = int(block_density * self.config.block_area_acres)

                # Calculate population
                population = units * self.config.persons_per_unit

                # Update the block
                block = self.grid.get_block(col, row)
                if block:
                    block.units = units
                    block.population = population

    def _generate_uniform_density(self):
        """Generate uniform density across the city (fallback if no polycentric config)"""
        default_density = 10.0  # units per acre

        for block in self.grid.blocks:
            units = int(default_density * self.config.block_area_acres)
            block.units = units
            block.population = units * self.config.persons_per_unit

    def _generate_transportation_network(self):
        """Generate transportation corridors and apply density effects"""
        self.transport_network = TransportationNetwork(
            grid_rows=self.config.height,
            grid_cols=self.config.width,
            centers=self.centers,
            config=self.transport_config
        )
        self.transport_network.generate_corridors()

        # Apply corridor effects
        for block in self.grid.blocks:
            if self.transport_network.is_on_corridor(block.y, block.x):
                block.units = int(block.units * self.transport_config.density_multiplier)
                block.population = block.population * self.transport_config.density_multiplier

    def _apply_density_constraints(self):
        """Apply city-wide density constraints to all blocks"""
        max_units_per_block = int(self.config.max_density_units_per_acre *
                                   self.config.block_area_acres)
        min_units_per_block = int(self.config.min_density_units_per_acre *
                                   self.config.block_area_acres)

        for block in self.grid.blocks:
            # Apply maximum density constraint
            if block.units > max_units_per_block:
                ratio = max_units_per_block / block.units
                block.units = max_units_per_block
                block.population = block.population * ratio

            # Apply minimum density constraint
            if block.units < min_units_per_block:
                ratio = min_units_per_block / block.units if block.units > 0 else 1
                block.units = min_units_per_block
                block.population = block.population * ratio

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def visualize(self, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize the city.

        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        if not self._generated:
            raise RuntimeError("City must be generated before visualization. Call generate() first.")

        if self.transport_network:
            from .visualize import visualize_with_corridors
            visualize_with_corridors(
                self.grid,
                self.centers,
                self.transport_network,
                save_path=save_path,
                show=show
            )
        else:
            from .visualize import visualize_grid
            visualize_grid(self.grid, save_path=save_path, show=show)

    def visualize_population(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize only population distribution"""
        if not self._generated:
            raise RuntimeError("City must be generated before visualization. Call generate() first.")

        from .visualize import visualize_population
        visualize_population(self.grid, save_path=save_path, show=show)

    def visualize_units(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize only housing units distribution"""
        if not self._generated:
            raise RuntimeError("City must be generated before visualization. Call generate() first.")

        from .visualize import visualize_units
        visualize_units(self.grid, save_path=save_path, show=show)

    def summary(self):
        """Print summary statistics about the city"""
        if not self._generated:
            raise RuntimeError("City must be generated before viewing summary. Call generate() first.")

        print(f"{'='*60}")
        print(f"CITY SUMMARY")
        print(f"{'='*60}")
        print(f"Dimensions: {self.config.width}x{self.config.height} blocks")
        print(f"Block size: {self.config.block_size_meters}m x {self.config.block_size_meters}m")
        print(f"Block area: {self.config.block_area_acres:.2f} acres")
        print(f"Max density: {self.config.max_density_units_per_acre:.1f} units/acre")

        if self.centers:
            print(f"\nActivity Centers: {len(self.centers)}")
            for i, center in enumerate(self.centers):
                print(f"  Center {i+1}: Position {center['position']}, "
                      f"Strength {center['strength']:.2f}, "
                      f"Peak Density {center['peak_density']:.1f} units/acre")

        total_units = sum(block.units for block in self.grid.blocks)
        total_population = self.grid.total_population
        all_units = [block.units for block in self.grid.blocks]

        print(f"\nDensity Statistics:")
        print(f"  Total housing units: {total_units:,}")
        print(f"  Total population: {total_population:,.0f}")
        print(f"  Average density: {np.mean(all_units):.2f} units/block")
        print(f"  Max density: {max(all_units)} units/block")
        print(f"  Min density: {min(all_units)} units/block")

        if self.transport_network:
            corridor_info = self.transport_network.get_corridor_info()
            print(f"\nTransportation Network:")
            print(f"  Type: {corridor_info['corridor_type']}")
            print(f"  Corridor blocks: {corridor_info['total_corridor_blocks']}")
            print(f"  Coverage: {corridor_info['corridor_coverage_pct']:.1f}%")
            print(f"  Density boost: {corridor_info['average_density_boost']:.1f}%")

        print(f"{'='*60}\n")

    @property
    def total_population(self) -> float:
        """Get total city population"""
        return self.grid.total_population

    @property
    def total_units(self) -> int:
        """Get total housing units"""
        return sum(block.units for block in self.grid.blocks)

    @property
    def total_area_acres(self) -> float:
        """Get total city area in acres"""
        return self.config.width * self.config.height * self.config.block_area_acres

    @property
    def average_density(self) -> float:
        """Get average density in units per acre"""
        return self.total_units / self.total_area_acres if self.total_area_acres > 0 else 0
