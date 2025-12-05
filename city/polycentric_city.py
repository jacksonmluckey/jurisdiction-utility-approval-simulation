import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .block import Block
from .grid import Grid

@dataclass
class PolycentricConfig:
    """Configuration for polycentric city generation"""
    num_centers: int = 3
    center_distribution: str = "uniform"  # "uniform", "clustered", or "random"
    primary_density: float = 18.0  # units per acre
    density_decay_rate: float = 0.20  # Lower = flatter gradient (polycentric)
    center_strength_decay: float = 0.6  # Each subcenter is X times the previous
    block_area_acres: float = 1.0  # Area of each block in acres
    persons_per_unit: float = 2.5
    min_center_separation_blocks: int = 5  # Minimum distance between centers

class PolycentricCity:
    """Generates polycentric urban density patterns on a grid"""

    def __init__(self, grid_rows: int, grid_cols: int, config: PolycentricConfig = None):
        self.rows = grid_rows
        self.cols = grid_cols
        self.config = config or PolycentricConfig()
        self.centers = []
        self.grid = Grid(width=grid_cols, height=grid_rows)

    def generate(self) -> Grid:
        """Generate housing units and population distributions"""
        self._place_centers()
        self._calculate_densities()
        return self.grid

    def _place_centers(self):
        """Place employment/activity centers based on distribution strategy"""
        if self.config.center_distribution == "uniform":
            self._place_uniform_centers()
        elif self.config.center_distribution == "clustered":
            self._place_clustered_centers()
        else:  # random
            self._place_random_centers()

    def _place_uniform_centers(self):
        """Place centers in a uniform grid pattern"""
        # Primary center at city center
        center_row = self.rows // 2
        center_col = self.cols // 2
        self.centers.append({
            'position': (center_row, center_col),
            'strength': 1.0,
            'peak_density': self.config.primary_density
        })

        # Secondary centers in a circle around primary
        if self.config.num_centers > 1:
            radius = min(self.rows, self.cols) // 3
            angles = np.linspace(0, 2*np.pi, self.config.num_centers, endpoint=False)

            for i, angle in enumerate(angles[1:], start=1):
                row = int(center_row + radius * np.sin(angle))
                col = int(center_col + radius * np.cos(angle))

                # Ensure within bounds
                row = np.clip(row, 0, self.rows - 1)
                col = np.clip(col, 0, self.cols - 1)

                strength = self.config.center_strength_decay ** i

                self.centers.append({
                    'position': (row, col),
                    'strength': strength,
                    'peak_density': self.config.primary_density * strength
                })

    def _place_clustered_centers(self):
        """Place centers clustered in one region (e.g., downtown area)"""
        base_row = self.rows // 2
        base_col = self.cols // 2

        for i in range(self.config.num_centers):
            # Add some randomness but keep clustered
            offset_row = np.random.randint(-self.rows//6, self.rows//6)
            offset_col = np.random.randint(-self.cols//6, self.cols//6)

            row = np.clip(base_row + offset_row, 0, self.rows - 1)
            col = np.clip(base_col + offset_col, 0, self.cols - 1)

            strength = self.config.center_strength_decay ** i

            self.centers.append({
                'position': (row, col),
                'strength': strength,
                'peak_density': self.config.primary_density * strength
            })

    def _place_random_centers(self):
        """Place centers randomly with minimum separation"""
        attempts = 0
        max_attempts = 1000

        for i in range(self.config.num_centers):
            placed = False

            while not placed and attempts < max_attempts:
                row = np.random.randint(0, self.rows)
                col = np.random.randint(0, self.cols)

                # Check minimum separation from existing centers
                if self._check_separation((row, col)):
                    strength = self.config.center_strength_decay ** i

                    self.centers.append({
                        'position': (row, col),
                        'strength': strength,
                        'peak_density': self.config.primary_density * strength
                    })
                    placed = True

                attempts += 1

    def _check_separation(self, position: Tuple[int, int]) -> bool:
        """Check if position maintains minimum separation from existing centers"""
        if not self.centers:
            return True

        for center in self.centers:
            distance = self._calculate_distance(position, center['position'])
            if distance < self.config.min_center_separation_blocks:
                return False
        return True

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
                units = int(block_density * self.config.block_area_acres)

                # Calculate population
                population = units * self.config.persons_per_unit

                # Update the block in the grid
                block = self.grid.get_block(col, row)
                if block:
                    block.units = units
                    block.population = population

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

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
                  f"Peak Density {center['peak_density']:.1f} units/acre")

        total_units = sum(block.units for block in self.grid.blocks)
        total_population = self.grid.total_population
        all_units = [block.units for block in self.grid.blocks]

        print(f"\nTotal housing units: {total_units}")
        print(f"Total population: {total_population:.0f}")
        print(f"Average density: {np.mean(all_units):.2f} units/block")
        print(f"Max density: {max(all_units):.2f} units/block")
        print(f"Min density: {min(all_units):.2f} units/block")