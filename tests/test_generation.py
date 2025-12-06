"""Tests for generation functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from city.generation import (
    CityCenter, TransportationCorridor, Park,
    generate_city_centers, generate_parks, place_points
)
from city import CityConfig, CityCentersConfig, ParkConfig


def test_place_points_uniform():
    """Test uniform point placement strategy."""
    points = place_points(3, 20, 20, placement_strategy="uniform")
    assert len(points) == 3
    # First point should be at center
    assert points[0] == (10, 10)


def test_place_points_random():
    """Test random point placement with separation."""
    points = place_points(5, 30, 30, placement_strategy="random", min_separation=5)
    assert len(points) == 5
    # Check minimum separation
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                import numpy as np
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                assert dist >= 5


def test_city_center_dataclass():
    """Test CityCenter dataclass creation."""
    center = CityCenter(
        position=(10, 15),
        strength=1.0,
        housing_peak_multiplier=3.6,
        office_peak_multiplier=2.0,
        shop_peak_multiplier=1.5,
        decay_rate=0.20
    )
    assert center.position == (10, 15)
    assert center.strength == 1.0
    assert center.housing_peak_multiplier == 3.6
    assert center.office_peak_multiplier == 2.0
    assert center.shop_peak_multiplier == 1.5
    assert center.decay_rate == 0.20


def test_generate_city_centers_multiplier_calculation():
    """Test that generate_city_centers correctly converts densities to multipliers."""
    config = CityConfig(
        width=20,
        height=20,
        base_housing_density_km2=1000.0,
        base_office_density_km2=500.0,
        base_shop_density_km2=250.0,
        random_seed=42
    )
    centers_config = CityCentersConfig(
        num_centers=2,
        primary_density_km2=3000.0,  # 3x base housing
        office_density_km2=1000.0,  # 2x base office
        shop_density_km2=500.0,  # 2x base shop
        center_strength_decay=0.5  # second center is 50% of first
    )

    centers = generate_city_centers(centers_config, config, 20, 20)

    assert len(centers) == 2

    # First center (strength = 1.0)
    assert centers[0].strength == 1.0
    assert centers[0].housing_peak_multiplier == 3.0  # 3000 / 1000
    assert centers[0].office_peak_multiplier == 2.0  # 1000 / 500
    assert centers[0].shop_peak_multiplier == 2.0  # 500 / 250

    # Second center (strength = 0.5)
    assert centers[1].strength == 0.5
    assert centers[1].housing_peak_multiplier == 1.5  # (3000 * 0.5) / 1000
    assert centers[1].office_peak_multiplier == 1.0  # (1000 * 0.5) / 500
    assert centers[1].shop_peak_multiplier == 1.0  # (500 * 0.5) / 250


def test_generate_city_centers_no_office_shop():
    """Test that office/shop multipliers default to 1.0 when not configured."""
    config = CityConfig(
        width=20,
        height=20,
        base_housing_density_km2=1000.0,
        random_seed=42
    )
    centers_config = CityCentersConfig(
        num_centers=1,
        primary_density_km2=2000.0,
        office_density_km2=None,  # not configured
        shop_density_km2=None  # not configured
    )

    centers = generate_city_centers(centers_config, config, 20, 20)

    assert len(centers) == 1
    assert centers[0].housing_peak_multiplier == 2.0
    assert centers[0].office_peak_multiplier == 1.0  # neutral
    assert centers[0].shop_peak_multiplier == 1.0  # neutral


def test_park_dataclass():
    """Test Park dataclass creation."""
    park = Park(
        center=(5, 5),
        size=9,
        blocks={(4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6)},
        shape="square"
    )
    assert park.center == (5, 5)
    assert park.size == 9
    assert len(park.blocks) == 9
    assert park.shape == "square"


def test_generate_parks():
    """Test park generation."""
    park_config = ParkConfig(
        num_parks=3,
        min_size_blocks=4,
        max_size_blocks=10,
        placement_strategy="random"
    )

    parks = generate_parks([park_config], 30, 30)

    assert len(parks) == 3
    for park in parks:
        assert isinstance(park, Park)
        assert 4 <= park.size <= 10
        assert len(park.blocks) > 0


if __name__ == '__main__':
    tests = [
        test_place_points_uniform,
        test_place_points_random,
        test_city_center_dataclass,
        test_generate_city_centers_multiplier_calculation,
        test_generate_city_centers_no_office_shop,
        test_park_dataclass,
        test_generate_parks
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print("✓ Passed\n")

    print("All generation tests passed! ✓")
