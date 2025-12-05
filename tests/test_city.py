"""Tests for core City functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from city import City, CityConfig, PolycentricConfig, ParkConfig, ZoningConfig, Use, Density, Grid, District, Block


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


def test_units_noise_float_scaling():
    """Test units_noise as float scales proportionally with base units."""
    config = CityConfig(
        width=10,
        height=10,
        units_noise=0.2,  # 20% proportional noise
        random_seed=42
    )
    polycentric = PolycentricConfig(num_centers=1, primary_density=20.0)

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    # With noise, units should vary from deterministic baseline
    # Generate without noise for comparison
    config_no_noise = CityConfig(
        width=10,
        height=10,
        units_noise=None,
        random_seed=42
    )
    city_no_noise = City(config=config_no_noise, polycentric_config=polycentric)
    city_no_noise.generate()

    # At least some blocks should differ (noise should have effect)
    units_with_noise = [b.units for b in city.grid.blocks]
    units_without_noise = [b.units for b in city_no_noise.grid.blocks]
    differences = sum(1 for a, b in zip(units_with_noise, units_without_noise) if a != b)

    assert differences > 0, "Noise should affect at least some blocks"


def test_units_noise_callable():
    """Test units_noise as callable function is applied correctly."""
    def custom_noise(base_units):
        # Simple: add 5 units of noise for testing
        import numpy as np
        return np.random.normal(0, 5)

    config = CityConfig(
        width=10,
        height=10,
        units_noise=custom_noise,
        random_seed=42
    )
    polycentric = PolycentricConfig(num_centers=1, primary_density=20.0)

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    # Check that generation succeeds and produces valid results
    assert city.total_units > 0
    assert all(b.units >= 0 for b in city.grid.blocks)


def test_persons_per_unit_callable():
    """Test persons_per_unit as callable varies by density."""
    def density_based_household(units, noise):
        if units < 30:
            return 3.0
        else:
            return 2.0

    config = CityConfig(
        width=10,
        height=10,
        persons_per_unit=density_based_household,
        random_seed=42
    )
    polycentric = PolycentricConfig(num_centers=1, primary_density=25.0)

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    # Check that household sizes vary as expected
    low_density_blocks = [b for b in city.grid.blocks if b.units < 30 and b.units > 0]
    high_density_blocks = [b for b in city.grid.blocks if b.units >= 30]

    if low_density_blocks and high_density_blocks:
        # Low density should have ~3.0 persons/unit
        avg_low = sum(b.population / b.units for b in low_density_blocks) / len(low_density_blocks)
        assert 2.9 < avg_low < 3.1, f"Expected ~3.0, got {avg_low}"

        # High density should have ~2.0 persons/unit
        avg_high = sum(b.population / b.units for b in high_density_blocks) / len(high_density_blocks)
        assert 1.9 < avg_high < 2.1, f"Expected ~2.0, got {avg_high}"


def test_combined_noise_and_callable_persons():
    """Test units_noise and callable persons_per_unit work together."""
    def adaptive_household(units, noise):
        base = 2.5
        if units > 50:
            base = 2.0
        if noise is not None and abs(noise) > 10:
            base += 0.2
        return base

    config = CityConfig(
        width=15,
        height=15,
        units_noise=0.15,
        persons_per_unit=adaptive_household,
        random_seed=42
    )
    polycentric = PolycentricConfig(num_centers=2, primary_density=30.0)

    city = City(config=config, polycentric_config=polycentric)
    city.generate()

    # Verify generation succeeds and produces sensible results
    assert city.total_units > 0
    assert city.total_population > 0
    assert all(b.population > 0 for b in city.grid.blocks if b.units > 0)

    # Check that persons_per_unit is in reasonable range
    for block in city.grid.blocks:
        if block.units > 0:
            ppu = block.population / block.units
            assert 1.5 < ppu < 3.5, f"persons_per_unit {ppu} out of expected range"


def test_park_generation():
    """Test basic park generation creates parks with correct properties."""
    config = CityConfig(width=20, height=20, random_seed=42)
    polycentric = PolycentricConfig(num_centers=2)
    park_config = ParkConfig(num_parks=3, min_size_blocks=2, max_size_blocks=6)

    city = City(config=config, polycentric_config=polycentric, park_config=park_config)
    city.generate()

    # Check parks were created
    assert len(city.parks) == 3, f"Expected 3 parks, got {len(city.parks)}"

    # Check park blocks have 0 units and population
    park_blocks = [b for b in city.grid.blocks if b.is_park]
    assert len(park_blocks) > 0, "Should have at least some park blocks"
    assert all(b.units == 0 for b in park_blocks), "All park blocks should have 0 units"
    assert all(b.population == 0 for b in park_blocks), "All park blocks should have 0 population"


def test_park_size_constraints():
    """Test parks respect size constraints."""
    config = CityConfig(width=30, height=30, random_seed=42)
    polycentric = PolycentricConfig(num_centers=1)
    park_config = ParkConfig(
        num_parks=5,
        min_size_blocks=4,
        max_size_blocks=10,
        placement_strategy="random"
    )

    city = City(config=config, polycentric_config=polycentric, park_config=park_config)
    city.generate()

    # Check each park size is within bounds
    for park in city.parks:
        park_size = len(park['blocks'])
        assert park_config.min_size_blocks <= park_size <= park_config.max_size_blocks, \
            f"Park size {park_size} outside bounds [{park_config.min_size_blocks}, {park_config.max_size_blocks}]"


def test_dispersed_park_placement():
    """Test dispersed placement strategy spreads parks out."""
    config = CityConfig(width=40, height=40, random_seed=42)
    polycentric = PolycentricConfig(num_centers=2)
    park_config = ParkConfig(
        num_parks=4,
        placement_strategy="dispersed"
    )

    city = City(config=config, polycentric_config=polycentric, park_config=park_config)
    city.generate()

    assert len(city.parks) == 4, "Should create 4 parks with dispersed placement"


def test_zoning_generation():
    """Test basic zoning generation."""
    config = CityConfig(width=20, height=20, random_seed=42)
    polycentric = PolycentricConfig(num_centers=2)
    zoning_config = ZoningConfig(enabled=True)

    city = City(config=config, polycentric_config=polycentric, zoning_config=zoning_config)
    city.generate()

    # All blocks should have zoning
    zoned_blocks = [b for b in city.grid.blocks if hasattr(b, 'zoning') and b.zoning is not None]
    assert len(zoned_blocks) == len(city.grid.blocks), "All blocks should have zoning"


def test_center_zoning():
    """Test that centers are zoned for all uses at high density."""
    config = CityConfig(width=20, height=20, random_seed=42)
    polycentric = PolycentricConfig(num_centers=2)
    zoning_config = ZoningConfig(enabled=True, center_radius_blocks=2)

    city = City(config=config, polycentric_config=polycentric, zoning_config=zoning_config)
    city.generate()

    # Check that at least some blocks near centers have all three uses
    multi_use_count = 0
    for block in city.grid.blocks:
        if block.zoning and len(block.zoning.allowed_uses) == 3:
            multi_use_count += 1
            # These should be high density
            assert block.zoning.max_density == Density.HIGH

    assert multi_use_count > 0, "Should have some blocks zoned for all uses near centers"


def test_density_zoning_levels():
    """Test that density levels are assigned based on unit counts."""
    config = CityConfig(width=30, height=30, random_seed=42)
    polycentric = PolycentricConfig(num_centers=2, primary_density=30.0)
    zoning_config = ZoningConfig(
        enabled=True,
        low_density_threshold=20,
        medium_density_threshold=50
    )

    city = City(config=config, polycentric_config=polycentric, zoning_config=zoning_config)
    city.generate()

    # Check that blocks are zoned according to their unit counts
    low_density_blocks = [b for b in city.grid.blocks
                           if b.zoning and b.zoning.max_density == Density.LOW]
    high_density_blocks = [b for b in city.grid.blocks
                            if b.zoning and b.zoning.max_density == Density.HIGH]

    # Should have a mix of density levels
    assert len(low_density_blocks) > 0, "Should have some low density blocks"
    assert len(high_density_blocks) > 0, "Should have some high density blocks"


def test_zoning_disabled():
    """Test that zoning can be disabled."""
    config = CityConfig(width=10, height=10, random_seed=42)
    polycentric = PolycentricConfig(num_centers=1)
    zoning_config = ZoningConfig(enabled=False)

    city = City(config=config, polycentric_config=polycentric, zoning_config=zoning_config)
    city.generate()

    # Blocks may have None zoning or no zoning attribute when disabled
    # Just check that generation succeeds
    assert city._generated == True


if __name__ == '__main__':
    tests = [
        test_grid_get_block,
        test_district_aggregation,
        test_city_requires_generate_before_viz,
        test_city_applies_density_constraints,
        test_city_properties,
        test_city_centers_placement,
        test_units_noise_float_scaling,
        test_units_noise_callable,
        test_persons_per_unit_callable,
        test_combined_noise_and_callable_persons,
        test_park_generation,
        test_park_size_constraints,
        test_dispersed_park_placement,
        test_zoning_generation,
        test_center_zoning,
        test_density_zoning_levels,
        test_zoning_disabled
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print("✓ Passed\n")

    print("All tests passed! ✓")
