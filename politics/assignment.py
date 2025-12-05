from typing import Dict
from city import Grid


def assign_districts_quadrants(grid: Grid, num_districts: int, district_assignments: Dict[int, int]) -> None:
    """
    Assign initial districts using quadrant-based approach
    Adjusts for population balance
    """
    if num_districts == 4:
        # Use quadrant approach for 4 districts
        midpoint_x = grid.width // 2
        midpoint_y = grid.height // 2

        for block in grid.blocks:
            if block.x < midpoint_x and block.y < midpoint_y:
                district_assignments[block.block_id] = 0
            elif block.x >= midpoint_x and block.y < midpoint_y:
                district_assignments[block.block_id] = 1
            elif block.x < midpoint_x and block.y >= midpoint_y:
                district_assignments[block.block_id] = 2
            else:
                district_assignments[block.block_id] = 3
    else:
        # Use stripe approach for other numbers
        assign_districts_by_population_stripes(grid, num_districts, district_assignments)


def assign_districts_by_population_stripes(grid: Grid, num_districts: int, district_assignments: Dict[int, int]) -> None:
    """
    Assign districts using horizontal stripes balanced by population
    """
    target_pop = grid.total_population / num_districts

    # Sort blocks by row (y), then column (x)
    sorted_blocks = sorted(grid.blocks, key=lambda b: (b.y, b.x))

    current_district = 0
    current_pop = 0

    for block in sorted_blocks:
        # Move to next district if we've reached target (except for last district)
        if current_district < num_districts - 1 and current_pop >= target_pop:
            current_district += 1
            current_pop = 0

        district_assignments[block.block_id] = current_district
        current_pop += block.population


def assign_districts_diagonal(grid: Grid, num_districts: int, district_assignments: Dict[int, int]) -> None:
    """
    Assign post-redistricting districts using diagonal approach
    """
    if num_districts == 4:
        # Use diagonal approach for 4 districts
        for block in grid.blocks:
            if block.x + block.y < max(grid.width, grid.height):
                district_assignments[block.block_id] = 0 if block.x < block.y else 1
            else:
                district_assignments[block.block_id] = 2 if block.x < block.y else 3
    else:
        # Use vertical stripes for other numbers
        assign_districts_by_population_stripes(grid, num_districts, district_assignments)
