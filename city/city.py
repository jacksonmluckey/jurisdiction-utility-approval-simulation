"""
City class - unified interface for creating and managing simulated cities
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Callable
from .grid import Grid
from .polycentric_city import PolycentricConfig
from .transportation_corridor import TransportationConfig, TransportationNetwork


@dataclass
class CityConfig:
    """
    Configuration parameters for city-wide properties.

    Attributes:
        width: Grid width in blocks (default: 50)
        height: Grid height in blocks (default: 50)
        block_size_meters: Side length of each block in meters (default: 100.0)
        block_area_acres: Area of each block in acres. 100m × 100m ≈ 2.47 acres (default: 2.47)
        max_density_units_per_acre: Maximum housing units per acre (default: 50.0).
            Typical ranges: 5-20 suburban, 20-60 urban, 60-150+ high-density urban.
        min_density_units_per_acre: Minimum housing units per acre (default: 0.5)
        persons_per_unit: Average household size or a function that takes (units, noise) and returns
            household size. Can be a float for constant value, or a callable for variable values
            based on block characteristics (default: 2.5)
        units_noise: Noise to add to units per block. Can be:
            - None: No noise (default)
            - float: Scaling factor for std dev (std_dev = units_noise * base_units).
              For example, 0.1 means 10% of base units as std dev
            - Callable[[int], float]: Function that takes base_units and returns noise value to add
        random_seed: Random seed for reproducibility. Set to an integer for consistent
            results across runs (default: None)
    """
    # Grid dimensions
    width: int = 50
    height: int = 50

    # Block physical properties
    block_size_meters: float = 100.0
    block_area_acres: float = 2.47

    # Density constraints
    max_density_units_per_acre: float = 50.0
    min_density_units_per_acre: float = 0.5

    # Population parameters
    persons_per_unit: Union[float, Callable[[int, Optional[float]], float]] = 2.5
    units_noise: Optional[Union[float, Callable[[int], float]]] = None

    # Random seed for reproducibility
    random_seed: Optional[int] = None


class City:
    """
    Main city class that integrates grid, polycentric density, and transportation.

    This class provides a unified interface for creating simulated cities with:
    - Block-level spatial grid structure
    - Polycentric density patterns with multiple activity centers
    - Transportation networks that boost density along corridors
    - City-wide constraints and parameters

    Basic Usage:
        >>> from city import City, CityConfig, PolycentricConfig
        >>>
        >>> config = CityConfig(width=50, height=50, max_density_units_per_acre=50)
        >>> polycentric = PolycentricConfig(num_centers=3, primary_density=25.0)
        >>>
        >>> city = City(config=config, polycentric_config=polycentric)
        >>> grid = city.generate()
        >>> city.summary()
        >>> city.visualize()

    With Single Transportation Corridor:
        >>> from city import TransportationConfig, CorridorType
        >>>
        >>> transport = TransportationConfig(
        ...     corridor_type=CorridorType.INTER_CENTER,
        ...     corridor_width_blocks=2,
        ...     density_multiplier=1.20
        ... )
        >>> city = City(config, polycentric, transport_configs=[transport])
        >>> city.generate()
        >>> city.visualize()

    With Multiple Transportation Corridors:
        >>> from city import TransportationConfig, CorridorType
        >>>
        >>> # Wide highways connecting centers
        >>> highways = TransportationConfig(
        ...     corridor_type=CorridorType.INTER_CENTER,
        ...     corridor_width_blocks=3,
        ...     density_multiplier=1.10
        ... )
        >>>
        >>> # Narrow transit lines in a grid pattern
        >>> transit = TransportationConfig(
        ...     corridor_type=CorridorType.GRID,
        ...     corridor_width_blocks=1,
        ...     density_multiplier=1.25
        ... )
        >>>
        >>> city = City(config, polycentric, transport_configs=[highways, transit])
        >>> city.generate()
        >>> city.visualize()

    High-Density Urban Example:
        >>> urban = CityConfig(
        ...     width=80,
        ...     height=80,
        ...     max_density_units_per_acre=100.0,
        ...     persons_per_unit=2.0
        ... )
        >>> polycentric = PolycentricConfig(
        ...     num_centers=7,
        ...     primary_density=40.0,
        ...     density_decay_rate=0.08
        ... )
        >>> city = City(config=urban, polycentric_config=polycentric)
        >>> city.generate()
    """

    def __init__(self,
                 config: Optional[CityConfig] = None,
                 polycentric_config: Optional[PolycentricConfig] = None,
                 transport_configs: Optional[List[TransportationConfig]] = None):
        """
        Initialize a city with configuration parameters.

        Args:
            config: City-wide configuration (dimensions, block size, max density, etc.)
            polycentric_config: Configuration for polycentric density patterns
            transport_configs: List of transportation corridor configurations. Can be a list of one or more configs.
        """
        self.config = config or CityConfig()
        self.polycentric_config = polycentric_config
        self.transport_configs = transport_configs

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
        if self.transport_configs and self.centers:
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
                base_units = int(block_density * self.config.block_area_acres)

                # Apply noise to units if configured
                noise_value = None
                units = base_units
                if self.config.units_noise is not None:
                    if callable(self.config.units_noise):
                        # Function that takes base_units and returns noise value
                        noise_value = self.config.units_noise(base_units)
                    else:
                        # Float scaling factor: std_dev = units_noise * base_units
                        std_dev = self.config.units_noise * base_units
                        noise_value = np.random.normal(0, std_dev)
                    units = max(0, int(base_units + noise_value))

                # Calculate population based on whether persons_per_unit is callable
                if callable(self.config.persons_per_unit):
                    population = units * self.config.persons_per_unit(units, noise_value)
                else:
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
            base_units = int(default_density * self.config.block_area_acres)

            # Apply noise to units if configured
            noise_value = None
            units = base_units
            if self.config.units_noise is not None:
                if callable(self.config.units_noise):
                    # Function that takes base_units and returns noise value
                    noise_value = self.config.units_noise(base_units)
                else:
                    # Float scaling factor: std_dev = units_noise * base_units
                    std_dev = self.config.units_noise * base_units
                    noise_value = np.random.normal(0, std_dev)
                units = max(0, int(base_units + noise_value))

            block.units = units

            # Calculate population based on whether persons_per_unit is callable
            if callable(self.config.persons_per_unit):
                block.population = units * self.config.persons_per_unit(units, noise_value)
            else:
                block.population = units * self.config.persons_per_unit

    def _generate_transportation_network(self):
        """Generate transportation corridors and apply density effects"""
        self.transport_network = TransportationNetwork(
            grid_rows=self.config.height,
            grid_cols=self.config.width,
            centers=self.centers,
            configs=self.transport_configs
        )
        self.transport_network.generate_corridors()

        # Apply corridor effects with max multiplier for overlapping corridors
        for block in self.grid.blocks:
            block_key = (block.y, block.x)
            if block_key in self.transport_network.corridor_details:
                corridor_info_list = self.transport_network.corridor_details[block_key]
                max_multiplier = max(info['density_multiplier'] for info in corridor_info_list)
                block.units = int(block.units * max_multiplier)

                # Recalculate population based on new units if persons_per_unit is callable
                if callable(self.config.persons_per_unit):
                    # For transportation corridors, we don't have the original noise value,
                    # so we pass None
                    block.population = block.units * self.config.persons_per_unit(block.units, None)
                else:
                    block.population = block.population * max_multiplier

    def _apply_density_constraints(self):
        """Apply city-wide density constraints to all blocks"""
        max_units_per_block = int(self.config.max_density_units_per_acre *
                                   self.config.block_area_acres)
        min_units_per_block = int(self.config.min_density_units_per_acre *
                                   self.config.block_area_acres)

        for block in self.grid.blocks:
            # Apply maximum density constraint
            if block.units > max_units_per_block:
                block.units = max_units_per_block
                # Recalculate population based on new units
                if callable(self.config.persons_per_unit):
                    block.population = block.units * self.config.persons_per_unit(block.units, None)
                else:
                    block.population = block.units * self.config.persons_per_unit

            # Apply minimum density constraint
            if block.units < min_units_per_block:
                block.units = min_units_per_block
                # Recalculate population based on new units
                if callable(self.config.persons_per_unit):
                    block.population = block.units * self.config.persons_per_unit(block.units, None)
                else:
                    block.population = block.units * self.config.persons_per_unit

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
            print(f"  Number of corridor types: {corridor_info['num_corridor_configs']}")
            print(f"  Total corridor blocks: {corridor_info['total_corridor_blocks']}")
            print(f"  Coverage: {corridor_info['corridor_coverage_pct']:.1f}%")
            print(f"  Average density boost: {corridor_info['average_density_boost']:.1f}%")

            # Show details for each corridor configuration
            for config_info in corridor_info['corridor_configs']:
                print(f"\n  Corridor {config_info['index'] + 1}:")
                print(f"    Type: {config_info['type']}")
                print(f"    Width: {config_info['width_blocks']} blocks")
                print(f"    Density boost: {config_info['density_boost_pct']:.1f}%")

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

    def get_center_info(self) -> List[dict]:
        """Get information about placed centers"""
        return self.centers
