import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Callable
from .block import Block
from .grid import Grid
from .transportation_corridor import TransportationNetwork, TransportationConfig


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
class CityCentersConfig:
    """
    Configuration for polycentric city generation.

    Attributes:
        num_centers: Number of activity/employment centers (default: 3)
        center_distribution: Distribution pattern - "uniform" (evenly spaced),
            "clustered" (grouped together), or "random" (default: "uniform")
        primary_density_km2: Peak density at primary center in units per km² (default: 4448.0)
        density_decay_rate: Rate of density decay from centers. Lower = flatter/polycentric,
            higher = steeper/monocentric. Typical: 0.05-0.10 (gradual), 0.10-0.20 (moderate),
            0.20-0.30 (steep) (default: 0.20)
        center_strength_decay: Strength multiplier for each subsequent center relative
            to previous. 0.6 means second center is 60% as strong as first (default: 0.6)
        block_size_meters: Side length of each block in meters (default: 100.0)
        persons_per_unit: Average household size or a function that takes (units, noise) and returns
            household size. Can be a float for constant value, or a callable for variable values
            based on block characteristics (default: 2.5)
        units_noise: Noise to add to units per block. Can be:
            - None: No noise (default)
            - float: Scaling factor for std dev (std_dev = units_noise * base_units).
              For example, 0.1 means 10% of base units as std dev
            - Callable[[int], float]: Function that takes base_units and returns noise value to add
        min_center_separation_blocks: Minimum distance between centers (default: 5)
        starting_point: Optional (row, col) tuple for the first center. If None, defaults to
            center of grid (default: None)
    """
    num_centers: int = 3
    center_distribution: str = "uniform"
    primary_density_km2: float = 4448.0
    density_decay_rate: float = 0.20
    center_strength_decay: float = 0.6
    block_size_meters: float = 100.0
    persons_per_unit: Union[float, Callable[[int, Optional[float]], float]] = 2.5
    units_noise: Optional[Union[float, Callable[[int], float]]] = None
    min_center_separation_blocks: int = 5
    starting_point: Optional[Tuple[int, int]] = None

    # Office and shop density parameters (units per km²)
    office_density_km2: Optional[float] = None
    shop_density_km2: Optional[float] = None
    office_decay_rate: Optional[float] = None  # defaults to density_decay_rate if None
    shop_decay_rate: Optional[float] = None  # defaults to density_decay_rate if None

    @property
    def block_area_km2(self) -> float:
        """Calculate block area in km² from block size in meters"""
        return (self.block_size_meters / 1000.0) ** 2

class CityCenters:
    """
    Generates polycentric urban density patterns on a grid.

    Creates cities with multiple activity centers where density decays exponentially
    from each center. Supports uniform, clustered, or random center placement, and
    can integrate transportation corridors that boost density along routes.

    Note: For most use cases, prefer the City class which provides a simpler interface.
    Use CityCenters directly only when you need granular control over the generation
    process or want to work with the internal components.
    """

    def __init__(self, grid_rows: int, grid_cols: int,
                 config: CityCentersConfig = None,
                 transport_configs: Optional[List[TransportationConfig]] = None):
        self.rows = grid_rows
        self.cols = grid_cols
        self.config = config or CityCentersConfig()
        self.transport_configs = transport_configs
        self.centers = []
        self.grid = Grid(width=grid_cols, height=grid_rows)
        self.transport_network = None

    def generate(self) -> Grid:
        """Generate housing units and population distributions"""
        self._place_centers()
        self._calculate_densities()

        # Generate transportation corridors if configured
        if self.transport_configs:
            self._generate_transportation_corridors()
            self._apply_corridor_effects()

        return self.grid

    def _place_centers(self):
        """Place employment/activity centers based on distribution strategy"""
        # Use common placement utility
        positions = place_points(
            num_points=self.config.num_centers,
            grid_rows=self.rows,
            grid_cols=self.cols,
            placement_strategy=self.config.center_distribution,
            min_separation=self.config.min_center_separation_blocks,
            starting_point=self.config.starting_point
        )

        # Create center objects with strength and density info
        for i, position in enumerate(positions):
            strength = self.config.center_strength_decay ** i

            self.centers.append({
                'position': position,
                'strength': strength,
                'peak_density': self.config.primary_density_km2 * strength
            })

    def _calculate_densities(self):
        """Calculate housing units and population using additive exponential model"""
        for row in range(self.rows):
            for col in range(self.cols):
                block_density = 0.0

                # Sum contributions from all centers
                for center in self.centers:
                    distance = self._calculate_distance(
                        (row, col),
                        center['position']
                    )

                    # Exponential decay: D = D0 * exp(-b * distance)
                    contribution = center['peak_density'] * np.exp(
                        -self.config.density_decay_rate * distance
                    )
                    block_density += contribution

                # Convert density to housing units
                base_units = int(block_density * self.config.block_area_km2)

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

                # Update the block in the grid
                block = self.grid.get_block(col, row)
                if block:
                    block.units = units
                    block.population = population

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _generate_transportation_corridors(self):
        """Generate transportation network based on centers"""
        self.transport_network = TransportationNetwork(
            self.rows,
            self.cols,
            self.centers,
            self.transport_configs
        )
        self.transport_network.generate_corridors()

    def _apply_corridor_effects(self):
        """Apply density multiplier to blocks on transportation corridors.

        When multiple corridors overlap, applies the maximum density multiplier.
        """
        if not self.transport_network:
            return

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

    def get_center_info(self) -> List[dict]:
        """Get information about placed centers"""
        return self.centers

    def visualize_summary(self):
        """Print summary statistics"""
        print(f"City Grid: {self.rows}x{self.cols} blocks")
        print(f"Number of centers: {len(self.centers)}")
        print(f"\nCenter locations and strengths:")
        for i, center in enumerate(self.centers):
            print(f"  Center {i+1}: Position {center['position']}, "
                  f"Strength {center['strength']:.2f}, "
                  f"Peak Density {center['peak_density']:.1f} units/km²")

        total_units = sum(block.units for block in self.grid.blocks)
        total_population = self.grid.total_population
        all_units = [block.units for block in self.grid.blocks]

        print(f"\nTotal housing units: {total_units}")
        print(f"Total population: {total_population:.0f}")
        print(f"Average density: {np.mean(all_units):.2f} units/block")
        print(f"Max density: {max(all_units):.2f} units/block")
        print(f"Min density: {min(all_units):.2f} units/block")

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