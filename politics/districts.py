from typing import Dict
from city import Grid, District


def get_districts(grid: Grid, district_assignments: Dict[int, int]) -> Dict[int, District]:
    """Get districts grouped by district ID"""
    districts = {}

    for block in grid.blocks:
        district_id = district_assignments[block.block_id]
        if district_id not in districts:
            districts[district_id] = District(district_id=district_id)
        districts[district_id].blocks.append(block)

    return districts
