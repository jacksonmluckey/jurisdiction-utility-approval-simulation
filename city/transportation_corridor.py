import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from enum import Enum

class CorridorType(Enum):
    """Types of transportation corridors"""
    RADIAL = "radial"  # Spoke pattern from primary center
    INTER_CENTER = "inter_center"  # Connecting all centers
    RING = "ring"  # Circular around primary center
    GRID = "grid"  # Orthogonal grid pattern

@dataclass
class TransportationConfig:
    """Configuration for transportation corridors"""
    corridor_type: CorridorType = CorridorType.INTER_CENTER
    corridor_width_blocks: int = 2  # Width of corridor in blocks
    density_multiplier: float = 1.15  # 15% density boost along corridors
    connect_all_centers: bool = True  # Connect all centers or just nearest
    max_corridor_distance: int = None  # Max distance to connect (None = no limit)
    include_ring_roads: bool = False  # Add ring road around primary center
    ring_road_radius_blocks: int = 10  # Radius of ring road from primary center
    radial_corridors_count: int = 4  # Number of radial corridors (if RADIAL type)

class TransportationNetwork:
    """Manages transportation corridors and their effects on density"""

    def __init__(self, grid_rows: int, grid_cols: int, 
                 centers: List[dict], config: TransportationConfig = None):
        self.rows = grid_rows
        self.cols = grid_cols
        self.centers = centers
        self.config = config or TransportationConfig()
        self.corridor_blocks = set()  # Set of (row, col) tuples on corridors

    def generate_corridors(self) -> Set[Tuple[int, int]]:
        """Generate all corridor blocks based on configuration"""
        self.corridor_blocks.clear()

        if self.config.corridor_type == CorridorType.RADIAL:
            self._generate_radial_corridors()
        elif self.config.corridor_type == CorridorType.INTER_CENTER:
            self._generate_inter_center_corridors()
        elif self.config.corridor_type == CorridorType.RING:
            self._generate_ring_corridors()
        elif self.config.corridor_type == CorridorType.GRID:
            self._generate_grid_corridors()

        if self.config.include_ring_roads:
            self._add_ring_road()

        return self.corridor_blocks

    def _generate_radial_corridors(self):
        """Generate radial corridors from primary center"""
        if not self.centers:
            return

        primary_center = self.centers[0]['position']
        angles = np.linspace(0, 2*np.pi, self.config.radial_corridors_count, 
                            endpoint=False)

        max_radius = max(self.rows, self.cols)

        for angle in angles:
            for r in range(max_radius):
                row = int(primary_center[0] + r * np.sin(angle))
                col = int(primary_center[1] + r * np.cos(angle))

                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self._add_corridor_block_with_width(row, col)

    def _generate_inter_center_corridors(self):
        """Generate corridors connecting centers"""
        if len(self.centers) < 2:
            return

        if self.config.connect_all_centers:
            # Connect all pairs of centers
            for i in range(len(self.centers)):
                for j in range(i + 1, len(self.centers)):
                    self._connect_two_centers(
                        self.centers[i]['position'],
                        self.centers[j]['position']
                    )
        else:
            # Connect each center to nearest neighbor
            for i, center in enumerate(self.centers):
                if i == 0:
                    continue  # Skip primary center
                # Find nearest center
                distances = [
                    self._distance(center['position'], other['position'])
                    for j, other in enumerate(self.centers) if j != i
                ]
                nearest_idx = np.argmin(distances)
                if nearest_idx >= i:
                    nearest_idx += 1
                self._connect_two_centers(
                    center['position'],
                    self.centers[nearest_idx]['position']
                )

    def _generate_ring_corridors(self):
        """Generate concentric ring corridors"""
        if not self.centers:
            return

        primary_center = self.centers[0]['position']

        # Multiple rings at different radii
        radii = [self.config.ring_road_radius_blocks * (i + 1) 
                for i in range(min(self.rows, self.cols) // 
                              (2 * self.config.ring_road_radius_blocks))]

        for radius in radii:
            self._add_ring_at_radius(primary_center, radius)

    def _generate_grid_corridors(self):
        """Generate orthogonal grid corridors"""
        # Vertical corridors
        spacing = max(5, self.cols // 6)
        for col in range(0, self.cols, spacing):
            for row in range(self.rows):
                self._add_corridor_block_with_width(row, col)

        # Horizontal corridors
        spacing = max(5, self.rows // 6)
        for row in range(0, self.rows, spacing):
            for col in range(self.cols):
                self._add_corridor_block_with_width(row, col)

    def _add_ring_road(self):
        """Add a ring road around the primary center"""
        if not self.centers:
            return
        primary_center = self.centers[0]['position']
        self._add_ring_at_radius(primary_center, 
                                 self.config.ring_road_radius_blocks)

    def _add_ring_at_radius(self, center: Tuple[int, int], radius: int):
        """Add a circular corridor at given radius from center"""
        center_row, center_col = center

        # Sample points around circle
        num_points = int(2 * np.pi * radius * 2)
        angles = np.linspace(0, 2*np.pi, num_points)

        for angle in angles:
            row = int(center_row + radius * np.sin(angle))
            col = int(center_col + radius * np.cos(angle))

            if 0 <= row < self.rows and 0 <= col < self.cols:
                self._add_corridor_block_with_width(row, col)

    def _connect_two_centers(self, pos1: Tuple[int, int], 
                            pos2: Tuple[int, int]):
        """Create corridor between two centers using Bresenham's line algorithm"""
        distance = self._distance(pos1, pos2)

        if (self.config.max_corridor_distance and 
            distance > self.config.max_corridor_distance):
            return

        # Bresenham's line algorithm
        x0, y0 = pos1[1], pos1[0]  # col, row
        x1, y1 = pos2[1], pos2[0]

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self._add_corridor_block_with_width(y0, x0)

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _add_corridor_block_with_width(self, row: int, col: int):
        """Add a block and its neighbors based on corridor width"""
        half_width = self.config.corridor_width_blocks // 2

        for dr in range(-half_width, half_width + 1):
            for dc in range(-half_width, half_width + 1):
                new_row = row + dr
                new_col = col + dc

                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    self.corridor_blocks.add((new_row, new_col))

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def apply_corridor_effects(self, housing_units: np.ndarray, 
                               population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply density multiplier to blocks on corridors"""
        modified_housing = housing_units.copy()
        modified_population = population.copy()

        for row, col in self.corridor_blocks:
            modified_housing[row, col] *= self.config.density_multiplier
            modified_population[row, col] *= self.config.density_multiplier

        return modified_housing, modified_population

    def is_on_corridor(self, row: int, col: int) -> bool:
        """Check if a block is on a corridor"""
        return (row, col) in self.corridor_blocks

    def get_corridor_info(self) -> dict:
        """Get statistics about the corridor network"""
        return {
            'total_corridor_blocks': len(self.corridor_blocks),
            'corridor_coverage_pct': (len(self.corridor_blocks) / 
                                     (self.rows * self.cols)) * 100,
            'corridor_type': self.config.corridor_type.value,
            'average_density_boost': (self.config.density_multiplier - 1) * 100
        }


# ===== INTEGRATION WITH YOUR EXISTING CODE =====

def apply_polycentric_with_corridors(grid: 'Grid', 
                                     polycentric_config: 'PolycentricConfig',
                                     transport_config: TransportationConfig) -> 'Grid':
    """
    Apply polycentric density patterns with transportation corridors to your Grid

    This function modifies the Grid in place and returns it
    """
    from .polycentric import PolycentricCity  # Import your polycentric class

    # Generate polycentric density
    poly_city = PolycentricCity(grid.height, grid.width, polycentric_config)
    housing_units, population = poly_city.generate()

    # Generate transportation network
    transport_net = TransportationNetwork(
        grid.height, 
        grid.width, 
        poly_city.get_center_info(),
        transport_config
    )
    transport_net.generate_corridors()

    # Apply corridor effects
    housing_units, population = transport_net.apply_corridor_effects(
        housing_units, population
    )

    # Update grid blocks
    for i, block in enumerate(grid.blocks):
        row = block.y
        col = block.x
        block.population = population[row, col]
        block.units = int(housing_units[row, col])

    # Store metadata
    grid.centers = poly_city.get_center_info()
    grid.corridor_network = transport_net

    return grid


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    from your_module.grid import Grid
    from your_module.polycentric import PolycentricConfig

    # Example 1: Inter-center corridors (most realistic)
    print("=== Example 1: Inter-Center Corridors ===")
    grid1 = Grid(width=30, height=30)

    poly_config1 = PolycentricConfig(
        num_centers=4,
        center_distribution="uniform",
        primary_density=18.0,
        density_decay_rate=0.18
    )

    transport_config1 = TransportationConfig(
        corridor_type=CorridorType.INTER_CENTER,
        corridor_width_blocks=2,
        density_multiplier=1.15,
        connect_all_centers=True,
        include_ring_roads=True,
        ring_road_radius_blocks=8
    )

    grid1 = apply_polycentric_with_corridors(grid1, poly_config1, transport_config1)

    print(f"Total population: {grid1.total_population:.0f}")
    print(f"Corridor coverage: {grid1.corridor_network.get_corridor_info()['corridor_coverage_pct']:.1f}%")
    print(f"Number of centers: {len(grid1.centers)}")

    # Example 2: Radial corridors (hub-and-spoke)
    print("\n=== Example 2: Radial Corridors ===")
    grid2 = Grid(width=30, height=30)

    poly_config2 = PolycentricConfig(
        num_centers=3,
        center_distribution="uniform",
        primary_density=20.0
    )

    transport_config2 = TransportationConfig(
        corridor_type=CorridorType.RADIAL,
        corridor_width_blocks=3,
        density_multiplier=1.20,
        radial_corridors_count=6
    )

    grid2 = apply_polycentric_with_corridors(grid2, poly_config2, transport_config2)

    print(f"Total population: {grid2.total_population:.0f}")
    print(f"Corridor coverage: {grid2.corridor_network.get_corridor_info()['corridor_coverage_pct']:.1f}%")

    # Example 3: Grid pattern (Manhattan-style)
    print("\n=== Example 3: Grid Pattern ===")
    grid3 = Grid(width=30, height=30)

    poly_config3 = PolycentricConfig(
        num_centers=2,
        primary_density=22.0,
        density_decay_rate=0.25
    )

    transport_config3 = TransportationConfig(
        corridor_type=CorridorType.GRID,
        corridor_width_blocks=1,
        density_multiplier=1.10
    )

    grid3 = apply_polycentric_with_corridors(grid3, poly_config3, transport_config3)

    print(f"Total population: {grid3.total_population:.0f}")
    print(f"Corridor coverage: {grid3.corridor_network.get_corridor_info()['corridor_coverage_pct']:.1f}%")


# ===== VISUALIZATION HELPER =====

def visualize_corridors(grid: 'Grid', title: str = "City with Transportation Corridors"):
    """Visualize the city with corridors highlighted"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Create density matrix
    housing_matrix = np.zeros((grid.height, grid.width))
    corridor_matrix = np.zeros((grid.height, grid.width))

    for block in grid.blocks:
        housing_matrix[block.y, block.x] = block.units
        if grid.corridor_network.is_on_corridor(block.y, block.x):
            corridor_matrix[block.y, block.x] = 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Housing density
    im1 = ax1.imshow(housing_matrix, cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('Housing Density')
    plt.colorbar(im1, ax=ax1, label='Housing Units')

    # Mark centers
    for i, center in enumerate(grid.centers):
        row, col = center['position']
        ax1.scatter(col, row, s=200 * center['strength'], c='blue', 
                   marker='*', edgecolors='white', linewidths=2)

    # Plot 2: Corridors
    ax2.imshow(corridor_matrix, cmap='Greys', interpolation='nearest')
    ax2.set_title('Transportation Corridors')

    # Plot 3: Combined
    im3 = ax3.imshow(housing_matrix, cmap='YlOrRd', interpolation='nearest', alpha=0.7)
    ax3.imshow(corridor_matrix, cmap='Blues', interpolation='nearest', alpha=0.3)
    ax3.set_title('Combined View')
    plt.colorbar(im3, ax=ax3, label='Housing Units')

    # Mark centers on combined
    for i, center in enumerate(grid.centers):
        row, col = center['position']
        ax3.scatter(col, row, s=200 * center['strength'], c='blue', 
                   marker='*', edgecolors='white', linewidths=2)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()