"""Tests for core City functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from city import City, CityConfig, PolycentricConfig, Grid, District, Block


def test_grid_get_block():
    """Test Grid.get_block() returns correct block and handles bounds."""
    grid = Grid(width=10, height=10)

    # Valid coordinates
    block = grid.get_block(5, 5)
    assert block is not None
    assert block.x == 5 and block.y == 5

    # Out of bounds
    assert grid.get_block(-1, 5) is None
    assert grid.get_block(5, 10) is None
    assert grid.get_block(10, 5) is None


def test_district_aggregation():
    """Test District correctly aggregates block data."""
    blocks = [
        Block(block_id=0, x=0, y=0, population=100, units=40),
        Block(block_id=1, x=1, y=0, population=200, units=80)
    ]
    district = District(district_id=1, blocks=blocks)

    assert district.total_population == 300
    assert district.total_units == 120
    assert district.num_blocks == 2


def test_city_requires_generate_before_viz():
    """Test City raises error if visualize called before generate."""
    city = City()

    try:
        city.visualize(show=False)
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "generate()" in str(e)


def test_city_applies_density_constraints():
    """Test City enforces max/min density constraints."""
    config = CityConfig(
        width=10,
        height=10,
        max_density_units_per_acre=20.0,
        min_density_units_per_acre=5.0,
        block_area_acres=2.0,
        random_seed=42
    )
    polycentric = PolycentricConfig(
        num_centers=2,
        primary_density=100.0  # Intentionally high
    )

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    max_units = int(config.max_density_units_per_acre * config.block_area_acres)
    min_units = int(config.min_density_units_per_acre * config.block_area_acres)

    for block in city.grid.blocks:
        assert block.units <= max_units
        assert block.units >= min_units


def test_city_properties():
    """Test City exposes correct aggregate properties."""
    config = CityConfig(width=5, height=5, random_seed=42)
    polycentric = PolycentricConfig(num_centers=1)

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    assert city.total_population == city.grid.total_population
    assert city.total_units == sum(b.units for b in city.grid.blocks)
    assert city.total_area_acres == 5 * 5 * config.block_area_acres
    assert city.average_density == city.total_units / city.total_area_acres


def test_city_centers_placement():
    """Test City places correct number of centers."""
    config = CityConfig(width=20, height=20, random_seed=42)
    polycentric = PolycentricConfig(num_centers=3)

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    assert len(city.centers) == 3
    assert len(city.get_center_info()) == 3


if __name__ == '__main__':
    tests = [
        test_grid_get_block,
        test_district_aggregation,
        test_city_requires_generate_before_viz,
        test_city_applies_density_constraints,
        test_city_properties,
        test_city_centers_placement
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print("✓ Passed\n")

    print("All tests passed! ✓")
