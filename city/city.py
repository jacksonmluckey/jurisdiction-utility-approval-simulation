"""
City class - unified interface for creating and managing simulated cities
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Callable
from .grid import Grid
from .city_centers import CityCentersConfig, place_points
from .transportation_corridor import TransportationConfig, TransportationNetwork
from .zoning import ZoningConfig, generate_zoning, get_zoning_summary
from .generation import (
    generate_city_centers,
    generate_transportation_corridors,
    generate_parks as gen_parks
)
from .density import create_density_map


@dataclass
class ParkConfig:
    """
    Configuration for park generation.

    Attributes:
        num_parks: Number of parks to generate (default: 0)
        min_size_blocks: Minimum park size in blocks (default: 1)
        max_size_blocks: Maximum park size in blocks (default: 9)
        placement_strategy: How to place parks - "uniform" (evenly spaced),
            "clustered" (grouped together), or "random" (default: "random")
        shape: Park shape - "square" or "circle" (default: "square")
        min_separation_blocks: Minimum distance between parks (default: 3)
    """
    num_parks: int = 0
    min_size_blocks: int = 1
    max_size_blocks: int = 9
    placement_strategy: str = "random"
    shape: str = "square"
    min_separation_blocks: int = 3


@dataclass
class CityConfig:
    """
    Configuration parameters for city-wide properties.

    Attributes:
        width: Grid width in blocks (default: 50)
        height: Grid height in blocks (default: 50)
        block_size_meters: Side length of each block in meters (default: 100.0)
        max_density_units_per_km2: Maximum housing units per km² (default: 12355.0).
            Typical ranges: 1235-12355 suburban, 12355-37065 urban, 37065-91435+ high-density urban.
        min_density_units_per_km2: Minimum housing units per km² (default: 124.0)
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
        max_density_offices_per_km2: Maximum office units per km² (default: 7413.0)
        max_density_shops_per_km2: Maximum shop/retail units per km² (default: 4942.0)
        office_center_concentration: Exponential decay rate for offices from centers.
            Higher values = more concentrated at center (default: 0.15)
        shop_center_concentration: Exponential decay rate for shops from centers.
            Higher values = more concentrated at center (default: 0.10)
        shop_corridor_multiplier: Density boost for shops along transportation corridors.
            1.0 = no boost, 1.5 = 50% increase (default: 1.3)
    """
    # Grid dimensions
    width: int = 50
    height: int = 50

    # Block physical properties
    block_size_meters: float = 100.0

    # Density constraints (units per km²)
    max_density_units_per_km2: float = 12355.0
    min_density_units_per_km2: float = 124.0

    # Population parameters
    persons_per_unit: Union[float, Callable[[int, Optional[float]], float]] = 2.5
    units_noise: Optional[Union[float, Callable[[int], float]]] = None

    # Random seed for reproducibility
    random_seed: Optional[int] = None

    # Commercial density parameters (units per km²)
    max_density_offices_per_km2: float = 7413.0
    max_density_shops_per_km2: float = 4942.0
    office_center_concentration: float = 0.15
    shop_center_concentration: float = 0.10
    shop_corridor_multiplier: float = 1.3

    # Base density parameters (units per km²)
    base_housing_density_km2: float = 1235.0
    base_office_density_km2: float = 741.0
    base_shop_density_km2: float = 494.0

    # Density combination method
    density_combination_method: str = "additive"  # "additive", "max", or "multiplicative"

    @property
    def block_area_km2(self) -> float:
        """Calculate block area in km² from block size in meters"""
        return (self.block_size_meters / 1000.0) ** 2


class City:
    """
    Main city class that integrates grid, density patterns, and transportation.

    This class provides a unified interface for creating simulated cities with:
    - Block-level spatial grid structure
    - City center-based density patterns with multiple activity centers
    - Transportation networks that boost density along corridors
    - City-wide constraints and parameters

    Basic Usage:
        >>> from city import City, CityConfig, CityCentersConfig
        >>>
        >>> config = CityConfig(width=50, height=50, max_density_units_per_km2=12355.0)
        >>> centers_config = CityCentersConfig(num_centers=3, primary_density_km2=6178.0)
        >>>
        >>> city = City(config=config, centers_config=centers_config)
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
        >>> city = City(config, centers_config, transport_configs=[transport])
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
        >>> city = City(config, centers_config, transport_configs=[highways, transit])
        >>> city.generate()
        >>> city.visualize()

    High-Density Urban Example:
        >>> urban = CityConfig(
        ...     width=80,
        ...     height=80,
        ...     max_density_units_per_km2=24710.0,
        ...     persons_per_unit=2.0
        ... )
        >>> centers = CityCentersConfig(
        ...     num_centers=7,
        ...     primary_density_km2=9884.0,
        ...     density_decay_rate=0.08
        ... )
        >>> city = City(config=urban, centers_config=centers)
        >>> city.generate()
    """

    def __init__(self,
                 config: Optional[CityConfig] = None,
                 centers_config: Optional[CityCentersConfig] = None,
                 transport_configs: Optional[List[TransportationConfig]] = None,
                 park_configs: Optional[Union[ParkConfig, List[ParkConfig]]] = None,
                 zoning_config: Optional[ZoningConfig] = None):
        """
        Initialize a city with configuration parameters.

        Args:
            config: City-wide configuration (dimensions, block size, max density, etc.)
            centers_config: Configuration for city centers density patterns
            transport_configs: List of transportation corridor configurations. Can be a list of one or more configs.
            park_configs: Configuration(s) for park generation. Can be a single ParkConfig or a list of ParkConfigs.
            zoning_config: Configuration for zoning generation
        """
        self.config = config or CityConfig()
        self.centers_config = centers_config
        self.transport_configs = transport_configs

        # Normalize park_configs to a list
        if park_configs is None:
            self.park_configs = []
        elif isinstance(park_configs, ParkConfig):
            self.park_configs = [park_configs]
        else:
            self.park_configs = park_configs

        self.zoning_config = zoning_config or ZoningConfig()

        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Initialize grid
        self.grid = Grid(width=self.config.width, height=self.config.height)

        # Components (populated during generation)
        self.centers = []  # Will be List[CityCenter] after new generation
        self.corridors = []  # List[TransportationCorridor]
        self.parks = []  # Will be List[Park] after new generation
        self.density_map = None  # DensityMap object
        self.transport_network = None  # Legacy, kept for compatibility

        # Track if city has been generated
        self._generated = False

    def generate(self) -> Grid:
        """
        Generate the city with all configured features.

        Returns:
            Grid object with populated blocks
        """
        # Generate city components as objects (don't modify grid yet)
        if self.centers_config:
            self.centers = generate_city_centers(
                self.centers_config,
                self.config,
                self.config.height,
                self.config.width
            )
        else:
            # No centers - will use uniform density
            self.centers = []

        # Generate transportation corridors (if configured)
        if self.transport_configs and self.centers:
            self.corridors = generate_transportation_corridors(
                self.centers,
                self.transport_configs,
                self.config.height,
                self.config.width
            )
        else:
            self.corridors = []

        # Generate parks (if configured)
        if self.park_configs:
            self.parks = gen_parks(
                self.park_configs,
                self.config.height,
                self.config.width
            )
        else:
            self.parks = []

        # Create density map from all components
        if self.centers:
            self.density_map = create_density_map(
                self.centers,
                self.corridors,
                self.parks,
                self.config,
                self.config.density_combination_method
            )
            # Apply density map to grid
            self.density_map.apply_to_grid(self.grid, self.config)
        else:
            # No centers - use uniform density
            self._generate_uniform_density()

        # Mark park blocks on grid (BEFORE applying constraints so they can be skipped)
        for park in self.parks:
            for y, x in park.blocks:
                block = self.grid.get_block(x, y)
                if block:
                    block.is_park = True

        # Apply city-wide density constraints (will skip park blocks)
        self._apply_density_constraints()

        # Generate zoning (after parks, using centers and density info)
        if self.zoning_config.enabled:
            # Convert centers to dict format for compatibility with zoning
            centers_dict = self._centers_to_dict()
            generate_zoning(self.grid, centers_dict, self.zoning_config)

        self._generated = True
        return self.grid

    def _centers_to_dict(self) -> List[dict]:
        """Convert CityCenter objects to dict format for backwards compatibility."""
        if not self.centers:
            return []

        # Check if already in dict format (from old code)
        if self.centers and isinstance(self.centers[0], dict):
            return self.centers

        # Convert CityCenter objects to dicts
        centers_dict = []
        for center in self.centers:
            centers_dict.append({
                'position': center.position,
                'strength': center.strength,
                'peak_density': center.housing_peak_multiplier * self.config.base_housing_density_km2
            })
        return centers_dict

    def _generate_centers_density(self):
        """Generate density using city centers model by delegating to CityCenters"""
        from .city_centers import CityCenters

        # Update centers config with city parameters to ensure consistency
        self.centers_config.block_size_meters = self.config.block_size_meters
        self.centers_config.persons_per_unit = self.config.persons_per_unit
        self.centers_config.units_noise = self.config.units_noise

        # Create temporary CityCenters to handle density generation
        # Don't pass transport_configs - we'll handle that separately in City
        temp_city = CityCenters(
            grid_rows=self.config.height,
            grid_cols=self.config.width,
            config=self.centers_config,
            transport_configs=None
        )
        temp_city.generate()

        # Copy results back to our grid
        self.centers = temp_city.centers
        for i, block in enumerate(self.grid.blocks):
            temp_block = temp_city.grid.blocks[i]
            block.units = temp_block.units
            block.population = temp_block.population

    def _generate_uniform_density(self):
        """Generate uniform density across the city (fallback if no centers config)"""
        default_density = 2471.0  # units per km²

        for block in self.grid.blocks:
            base_units = int(default_density * self.config.block_area_km2)

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
        max_units_per_block = int(self.config.max_density_units_per_km2 *
                                   self.config.block_area_km2)
        min_units_per_block = int(self.config.min_density_units_per_km2 *
                                   self.config.block_area_km2)

        for block in self.grid.blocks:
            # Skip park blocks
            if hasattr(block, 'is_park') and block.is_park:
                continue

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

    def _generate_parks(self):
        """Generate parks throughout the city"""
        # Process each park configuration
        for park_config in self.park_configs:
            if park_config.num_parks == 0:
                continue

            # Use common placement utility
            park_positions = place_points(
                num_points=park_config.num_parks,
                grid_rows=self.config.height,
                grid_cols=self.config.width,
                placement_strategy=park_config.placement_strategy,
                min_separation=park_config.min_separation_blocks
            )

            # For each park position, determine size and mark blocks
            for park_center in park_positions:
                size = np.random.randint(
                    park_config.min_size_blocks,
                    park_config.max_size_blocks + 1
                )

                park_blocks = self._get_park_blocks(park_center, size, park_config.shape)

                # Mark blocks as parks
                for block_pos in park_blocks:
                    block = self.grid.get_block(block_pos[1], block_pos[0])
                    if block:
                        block.is_park = True
                        block.units = 0
                        block.population = 0

                # Store park info
                self.parks.append({
                    'center': park_center,
                    'size': size,
                    'blocks': park_blocks
                })

    def _get_park_blocks(self, center: tuple, size: int, shape: str) -> List[tuple]:
        """Get list of block positions for a park given its center and size"""
        blocks = []
        center_y, center_x = center

        if shape == "circle":
            # Circular park
            radius = np.sqrt(size / np.pi)
            for dy in range(-int(radius) - 1, int(radius) + 2):
                for dx in range(-int(radius) - 1, int(radius) + 2):
                    if dx*dx + dy*dy <= radius*radius:
                        y = center_y + dy
                        x = center_x + dx
                        if 0 <= x < self.config.width and 0 <= y < self.config.height:
                            blocks.append((y, x))
                            if len(blocks) >= size:
                                return blocks
        else:
            # Square park
            side = int(np.sqrt(size))
            for dy in range(-side // 2, side // 2 + 1):
                for dx in range(-side // 2, side // 2 + 1):
                    y = center_y + dy
                    x = center_x + dx
                    if 0 <= x < self.config.width and 0 <= y < self.config.height:
                        blocks.append((y, x))
                        if len(blocks) >= size:
                            return blocks

        return blocks

    def _generate_offices(self):
        """Generate office units concentrated at city centers"""
        from .zoning import Use

        if not self.centers:
            return

        max_offices_per_block = int(self.config.max_density_offices_per_km2 *
                                     self.config.block_area_km2)

        for block in self.grid.blocks:
            # Skip parks
            if block.is_park:
                continue

            # Check if office use is allowed by zoning
            if block.zoning and not block.zoning.allows_use(Use.OFFICE):
                continue

            # Calculate office density based on distance to nearest center
            min_distance = float('inf')
            for center in self.centers:
                center_y, center_x = center['position']
                distance = np.sqrt((block.x - center_x)**2 + (block.y - center_y)**2)
                min_distance = min(min_distance, distance)

            # Exponential decay from center
            office_density_factor = np.exp(-self.config.office_center_concentration * min_distance)

            # Calculate offices
            offices = int(max_offices_per_block * office_density_factor)
            block.offices = max(0, offices)

    def _generate_shops(self):
        """Generate shop/retail units concentrated at centers and along corridors"""
        from .zoning import Use

        if not self.centers:
            return

        max_shops_per_block = int(self.config.max_density_shops_per_km2 *
                                   self.config.block_area_km2)

        for block in self.grid.blocks:
            # Skip parks
            if block.is_park:
                continue

            # Check if commercial use is allowed by zoning
            if block.zoning and not block.zoning.allows_use(Use.COMMERCIAL):
                continue

            # Calculate shop density based on distance to nearest center
            min_distance = float('inf')
            for center in self.centers:
                center_y, center_x = center['position']
                distance = np.sqrt((block.x - center_x)**2 + (block.y - center_y)**2)
                min_distance = min(min_distance, distance)

            # Exponential decay from center
            shop_density_factor = np.exp(-self.config.shop_center_concentration * min_distance)

            # Apply corridor multiplier if on a corridor
            if self.transport_network and self.transport_network.is_on_corridor(block.y, block.x):
                shop_density_factor *= self.config.shop_corridor_multiplier

            # Calculate shops
            shops = int(max_shops_per_block * shop_density_factor)
            block.shops = max(0, shops)

    def visualize(self, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize the city.

        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        if not self._generated:
            raise RuntimeError("City must be generated before visualization. Call generate() first.")

        if self.corridors or self.transport_network:
            from .visualize import visualize_with_corridors
            visualize_with_corridors(
                self.grid,
                self._centers_to_dict(),
                self.transport_network,  # Still pass for backwards compat
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

    def visualize_zoning(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize zoning map showing density levels and allowed uses"""
        if not self._generated:
            raise RuntimeError("City must be generated before visualization. Call generate() first.")

        if not self.zoning_config.enabled:
            print("Warning: Zoning is not enabled for this city.")
            return

        from .visualize import visualize_zoning
        visualize_zoning(self, save_path=save_path, show=show)

    def summary(self):
        """Print summary statistics about the city"""
        if not self._generated:
            raise RuntimeError("City must be generated before viewing summary. Call generate() first.")

        print(f"{'='*60}")
        print(f"CITY SUMMARY")
        print(f"{'='*60}")
        print(f"Dimensions: {self.config.width}x{self.config.height} blocks")
        print(f"Block size: {self.config.block_size_meters}m x {self.config.block_size_meters}m")
        print(f"Block area: {self.config.block_area_km2:.6f} km²")
        print(f"Max density: {self.config.max_density_units_per_km2:.1f} units/km²")

        if self.centers:
            print(f"\nActivity Centers: {len(self.centers)}")
            for i, center in enumerate(self.centers):
                print(f"  Center {i+1}: Position {center['position']}, "
                      f"Strength {center['strength']:.2f}, "
                      f"Peak Density {center['peak_density']:.1f} units/km²")

        total_units = sum(block.units for block in self.grid.blocks)
        total_population = self.grid.total_population
        all_units = [block.units for block in self.grid.blocks]
        total_offices = sum(block.offices for block in self.grid.blocks)
        total_shops = sum(block.shops for block in self.grid.blocks)

        print(f"\nDensity Statistics:")
        print(f"  Total housing units: {total_units:,}")
        print(f"  Total population: {total_population:,.0f}")
        print(f"  Average density: {np.mean(all_units):.2f} units/block")
        print(f"  Max density: {max(all_units)} units/block")
        print(f"  Min density: {min(all_units)} units/block")

        print(f"\nCommercial Statistics:")
        print(f"  Total offices: {total_offices:,}")
        print(f"  Total shops: {total_shops:,}")
        blocks_with_offices = sum(1 for block in self.grid.blocks if block.offices > 0)
        blocks_with_shops = sum(1 for block in self.grid.blocks if block.shops > 0)
        print(f"  Blocks with offices: {blocks_with_offices} ({blocks_with_offices/(self.config.width*self.config.height)*100:.1f}%)")
        print(f"  Blocks with shops: {blocks_with_shops} ({blocks_with_shops/(self.config.width*self.config.height)*100:.1f}%)")

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

        if self.parks:
            total_park_blocks = sum(len(park['blocks']) for park in self.parks)
            park_coverage_pct = (total_park_blocks / (self.config.width * self.config.height)) * 100
            print(f"\nParks:")
            print(f"  Number of parks: {len(self.parks)}")
            print(f"  Total park blocks: {total_park_blocks}")
            print(f"  Park coverage: {park_coverage_pct:.1f}%")
            avg_park_size = total_park_blocks / len(self.parks)
            print(f"  Average park size: {avg_park_size:.1f} blocks")

        # Show zoning statistics if enabled
        if self.zoning_config.enabled:
            from .zoning import Use, Density
            zoning_summary = get_zoning_summary(self.grid)
            print(f"\nZoning:")
            print(f"  Density distribution:")
            print(f"    Low: {zoning_summary['density_percentages'][Density.LOW]:.1f}%")
            print(f"    Medium: {zoning_summary['density_percentages'][Density.MEDIUM]:.1f}%")
            print(f"    High: {zoning_summary['density_percentages'][Density.HIGH]:.1f}%")
            print(f"  Use permissions:")
            print(f"    Residential: {zoning_summary['use_percentages'][Use.RESIDENTIAL]:.1f}%")
            print(f"    Commercial: {zoning_summary['use_percentages'][Use.COMMERCIAL]:.1f}%")
            print(f"    Office: {zoning_summary['use_percentages'][Use.OFFICE]:.1f}%")
            print(f"  Mixed-use blocks: {zoning_summary['mixed_use_count']} ({zoning_summary['mixed_use_count']/zoning_summary['total_blocks']*100:.1f}%)")

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
    def total_area_km2(self) -> float:
        """Get total city area in km²"""
        return self.config.width * self.config.height * self.config.block_area_km2

    @property
    def average_density(self) -> float:
        """Get average density in units per km²"""
        return self.total_units / self.total_area_km2 if self.total_area_km2 > 0 else 0

    def get_center_info(self) -> List[dict]:
        """Get information about placed centers"""
        return self._centers_to_dict()
