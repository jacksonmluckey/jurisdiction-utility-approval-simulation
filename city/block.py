from dataclasses import dataclass


@dataclass
class Block:
    """
    Represents a single grid cell/block in the city.

    Attributes:
        block_id: Unique identifier for the block
        x: X-coordinate in the grid
        y: Y-coordinate in the grid
        population: Population living in this block
        units: Number of housing units in this block
    """
    block_id: int
    x: int
    y: int
    population: float
    units: int
