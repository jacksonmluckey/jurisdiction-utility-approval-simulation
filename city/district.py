from dataclasses import dataclass, field
from typing import List
from .block import Block


@dataclass
class District:
    """
    Represents a district composed of multiple blocks.

    Districts are groupings of blocks that can be used for administrative
    purposes, zoning analysis, or other spatial aggregations.

    Attributes:
        district_id: Unique identifier for the district
        blocks: List of Block objects belonging to this district
    """
    district_id: int
    blocks: List[Block] = field(default_factory=list)

    @property
    def total_population(self) -> float:
        return sum(block.population for block in self.blocks)

    @property
    def total_units(self) -> int:
        return sum(block.units for block in self.blocks)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)
