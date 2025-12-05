from dataclasses import dataclass
from typing import Optional


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
        zoning: Zoning information for this block (optional)
        offices: Number of office units in this block
        shops: Number of retail/shop units in this block
    """
    block_id: int
    x: int
    y: int
    population: float
    units: int
    is_park: bool = False
    zoning: Optional['Zoning'] = None
    offices: int = 0
    shops: int = 0
