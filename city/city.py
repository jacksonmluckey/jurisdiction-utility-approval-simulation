"""
City class - unified interface for creating and managing simulated cities
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Callable
from .grid import Grid
from .city_centers import CityCentersConfig, place_points
from .transportation_corridor import TransportationConfig
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
    City data structure containing grid, density patterns, and transportation.

    This class holds the generated city data including:
    - Block-level spatial grid structure
    - City centers with density multipliers
    - Transportation corridors
    - Parks and zoning information

    Use the generate_city() function to create and populate a City instance.

    Basic Usage:
        >>> from city import generate_city, CityConfig, CityCentersConfig
        >>>
        >>> config = CityConfig(width=50, height=50, max_density_units_per_km2=12355.0)
        >>> centers_config = CityCentersConfig(num_centers=3, primary_density_km2=6178.0)
        >>>
        >>> city = generate_city(config=config, centers_config=centers_config)
        >>> city.summary()
        >>> city.visualize()

    With Transportation Corridors:
        >>> from city import generate_city, TransportationConfig, CorridorType
        >>>
        >>> transport = TransportationConfig(
        ...     corridor_type=CorridorType.INTER_CENTER,
        ...     corridor_width_blocks=2,
        ...     density_multiplier=1.20
        ... )
        >>> city = generate_city(config, centers_config, transport_configs=[transport])
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
        >>> city = generate_city(config=urban, centers_config=centers)
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

        # Track if city has been generated
        self._generated = False

    def visualize(self, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize the city.

        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        if self.corridors:
            from .visualize import visualize_with_corridors
            visualize_with_corridors(
                self.grid,
                self.centers,
                self.corridors,
                save_path=save_path,
                show=show
            )
        else:
            from .visualize import visualize_grid
            visualize_grid(self.grid, save_path=save_path, show=show)

    def visualize_population(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize only population distribution"""
        from .visualize import visualize_population
        visualize_population(self.grid, save_path=save_path, show=show)

    def visualize_units(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize only housing units distribution"""
        from .visualize import visualize_units
        visualize_units(self.grid, save_path=save_path, show=show)

    def visualize_shops(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize only shop distribution"""
        from .visualize import visualize_shops
        visualize_shops(self.grid, save_path=save_path, show=show)

    def visualize_offices(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize only office distribution"""
        from .visualize import visualize_offices
        visualize_offices(self.grid, save_path=save_path, show=show)

    def visualize_zoning(self, save_path: Optional[str] = None, show: bool = True):
        """Visualize zoning map showing density levels and allowed uses"""
        if not self.zoning_config.enabled:
            print("Warning: Zoning is not enabled for this city.")
            return

        from .visualize import visualize_zoning
        visualize_zoning(self, save_path=save_path, show=show)

    def summary(self):
        """Print summary statistics about the city"""

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
                # Calculate peak housing density from multiplier
                peak_housing_density = center.housing_peak_multiplier * self.config.base_housing_density_km2
                print(f"  Center {i+1}: Position {center.position}, "
                      f"Strength {center.strength:.2f}, "
                      f"Peak Housing Density {peak_housing_density:.1f} units/km²")

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

        if self.corridors:
            total_corridor_blocks = sum(len(corridor.blocks) for corridor in self.corridors)
            corridor_coverage_pct = (total_corridor_blocks / (self.config.width * self.config.height)) * 100

            print(f"\nTransportation Network:")
            print(f"  Number of corridors: {len(self.corridors)}")
            print(f"  Total corridor blocks: {total_corridor_blocks}")
            print(f"  Coverage: {corridor_coverage_pct:.1f}%")

            # Show details for each corridor
            for i, corridor in enumerate(self.corridors):
                print(f"\n  Corridor {i + 1}:")
                print(f"    Type: {corridor.corridor_type}")
                print(f"    Width: {corridor.width_blocks} blocks")
                print(f"    Blocks: {len(corridor.blocks)}")
                print(f"    Housing multiplier: {corridor.housing_multiplier:.2f}")
                print(f"    Office multiplier: {corridor.office_multiplier:.2f}")
                print(f"    Shop multiplier: {corridor.shop_multiplier:.2f}")

        if self.parks:
            total_park_blocks = sum(len(park.blocks) for park in self.parks)
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
        """Get information about placed centers as dictionaries"""
        centers_dict = []
        for center in self.centers:
            centers_dict.append({
                'position': center.position,
                'strength': center.strength,
                'peak_density': center.housing_peak_multiplier * self.config.base_housing_density_km2
            })
        return centers_dict
