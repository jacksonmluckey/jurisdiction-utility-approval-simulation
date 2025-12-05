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
        is_park: Whether this block is part of a park
    """
    block_id: int
    x: int
    y: int
    population: float
    units: int
    is_park: bool = False
