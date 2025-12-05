import numpy as np
from dataclasses import dataclass, field
from typing import List
from .block import Block


@dataclass
class Grid:
    """Represents the entire grid with all blocks"""
    width: int
    height: int
    blocks: List[Block] = field(default_factory=list)

    def __post_init__(self):
        """Initialize blocks if not provided"""
        if not self.blocks:
            self._create_blocks()

    def _create_blocks(self):
        """Create all blocks in the grid with random population and units"""
        for i in range(self.height):
            for j in range(self.width):
                block_id = i * self.width + j
                population = np.random.lognormal(mean=4, sigma=1)
                units = np.random.poisson(lam=2)  # Random number of housing units

                self.blocks.append(Block(
                    block_id=block_id,
                    x=j,
                    y=i,
                    population=population,
                    units=units
                ))

    @property
    def total_population(self) -> float:
        return sum(block.population for block in self.blocks)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def get_block(self, x: int, y: int) -> Block:
        """Get block at coordinates (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            block_id = y * self.width + x
            return self.blocks[block_id]
        return None
