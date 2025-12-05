import numpy as np
from typing import Dict
from city import Block, Grid


def calculate_distance(block1: Block, block2: Block) -> float:
    """Calculate Euclidean distance between two blocks"""
    return np.sqrt((block1.x - block2.x)**2 + (block1.y - block2.y)**2)


def voter_utility(distance: float, b: float = 1.0, k: float = 15.0, q: float = 0.4) -> float:
    """
    Calculate voter utility as function of distance from development

    Parameters match the research paper's utility function
    """
    return b - k * (1 / (distance + q))**2


def calculate_approval_score(grid: Grid, site_block_id: int, district_assignments: Dict[int, int]) -> float:
    """
    Calculate approval score for siting a development at a given block

    Args:
        grid: The grid object
        site_block_id: ID of the block where development is proposed
        district_assignments: Dictionary mapping block_id to district_id

    Returns:
        Weighted approval score (0 to 1)
    """
    site_block = grid.blocks[site_block_id]
    district_id = district_assignments[site_block_id]

    # Get all blocks in the same district
    district_blocks = [b for b in grid.blocks if district_assignments[b.block_id] == district_id]

    # Calculate approval for each block in the district
    approvals = []
    populations = []

    for block in district_blocks:
        distance = calculate_distance(block, site_block)
        utility = voter_utility(distance)
        approval = 1 if utility > 0 else 0

        approvals.append(approval)
        populations.append(block.population)

    # Weight by population
    total_pop = sum(populations)
    if total_pop == 0:
        return 0

    weighted_approval = sum(a * p for a, p in zip(approvals, populations)) / total_pop
    return weighted_approval
