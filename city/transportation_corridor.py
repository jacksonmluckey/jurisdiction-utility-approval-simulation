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
    """
    Configuration for transportation corridors.

    Attributes:
        corridor_type: Type of corridor pattern (RADIAL, INTER_CENTER, RING, or GRID)
        corridor_width_blocks: Width of corridor in blocks. Use 1 for minor arterials,
            2-3 for major transit, 4+ for wide transportation zones (default: 2)
        density_multiplier: Density boost along corridors. 1.15 = 15% increase (default: 1.15)
        connect_all_centers: If True, connect all center pairs; if False, only nearest
            neighbors (default: True)
        max_corridor_distance: Maximum distance to connect centers. None = no limit (default: None)
        include_ring_roads: Add ring road around primary center (default: False)
        ring_road_radius_blocks: Radius of ring road from primary center (default: 10)
        radial_corridors_count: Number of radial corridors for RADIAL type (default: 4)
    """
    corridor_type: CorridorType = CorridorType.INTER_CENTER
    corridor_width_blocks: int = 2
    density_multiplier: float = 1.15
    connect_all_centers: bool = True
    max_corridor_distance: int = None
    include_ring_roads: bool = False
    ring_road_radius_blocks: int = 10
    radial_corridors_count: int = 4

class TransportationNetwork:
    """
    Manages transportation corridors and their effects on density.

    Generates corridor patterns (radial, inter-center, ring, or grid) and tracks which
    blocks are affected by transportation infrastructure.
    """

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