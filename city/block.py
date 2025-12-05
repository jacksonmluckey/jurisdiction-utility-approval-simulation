from dataclasses import dataclass


@dataclass
class Block:
    """Represents a single grid cell/block"""
    block_id: int
    x: int
    y: int
    population: float
    units: int
