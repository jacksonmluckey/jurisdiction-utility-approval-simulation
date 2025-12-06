"""Tests for density calculation functions."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from city.density import (
    DensityMap,
    calculate_center_multiplier_map,
    calculate_corridor_multiplier_map,
    combine_multiplier_maps,
    create_density_map
)
from city.generation import CityCenter, TransportationCorridor, Park
from city import CityConfig


def test_calculate_center_multiplier_map_exponential_decay():
    """Test that center multiplier map uses exponential decay."""
    center = CityCenter(
        position=(10, 10),
        strength=1.0,
        housing_peak_multiplier=2.0,
        decay_rate=0.2
    )

    multiplier_map = calculate_center_multiplier_map(center, 21, 21, "housing")

    # At center, multiplier should equal peak
    assert np.isclose(multiplier_map[10, 10], 2.0)

    # At distance 5 from center (10, 15)
    # multiplier = 2.0 * exp(-0.2 * 5) = 2.0 * exp(-1.0) = 0.736
    assert np.isclose(multiplier_map[10, 15], 2.0 * np.exp(-1.0), rtol=0.01)

    # Multiplier should decrease with distance
    assert multiplier_map[10, 15] < multiplier_map[10, 12]
    assert multiplier_map[10, 12] < multiplier_map[10, 10]


def test_calculate_center_multiplier_map_different_types():
    """Test center multiplier map for different density types."""
    center = CityCenter(
        position=(5, 5),
        strength=1.0,
        housing_peak_multiplier=3.0,
        office_peak_multiplier=2.0,
        shop_peak_multiplier=1.5,
        decay_rate=0.15
    )

    housing_map = calculate_center_multiplier_map(center, 11, 11, "housing")
    office_map = calculate_center_multiplier_map(center, 11, 11, "office")
    shop_map = calculate_center_multiplier_map(center, 11, 11, "shop")

    # At center, each should have their respective peak multipliers
    assert np.isclose(housing_map[5, 5], 3.0)
    assert np.isclose(office_map[5, 5], 2.0)
    assert np.isclose(shop_map[5, 5], 1.5)


def test_calculate_corridor_multiplier_map():
    """Test that corridor multiplier map sets correct values."""
    corridor = TransportationCorridor(
        corridor_type="inter_center",
        blocks={(5, 5), (5, 6), (5, 7)},
        housing_multiplier=1.15,
        office_multiplier=1.10,
        shop_multiplier=1.20
    )

    housing_map = calculate_corridor_multiplier_map(corridor, 10, 10, "housing")
    office_map = calculate_corridor_multiplier_map(corridor, 10, 10, "office")
    shop_map = calculate_corridor_multiplier_map(corridor, 10, 10, "shop")

    # Corridor blocks should have multiplier values
    assert housing_map[5, 5] == 1.15
    assert housing_map[5, 6] == 1.15
    assert housing_map[5, 7] == 1.15

    assert office_map[5, 5] == 1.10
    assert shop_map[5, 5] == 1.20

    # Non-corridor blocks should be 1.0 (neutral)
    assert housing_map[0, 0] == 1.0
    assert housing_map[9, 9] == 1.0
    assert office_map[0, 0] == 1.0
    assert shop_map[0, 0] == 1.0


def test_combine_multiplier_maps_additive():
    """Test additive combination: sum(M - 1) + 1."""
    map1 = np.array([[1.1, 1.2], [1.0, 1.5]])
    map2 = np.array([[1.2, 1.0], [1.3, 1.1]])
    map3 = np.array([[1.0, 1.1], [1.0, 1.0]])

    result = combine_multiplier_maps([map1, map2, map3], "additive")

    # [0,0]: (1.1-1) + (1.2-1) + (1.0-1) + 1 = 0.1 + 0.2 + 0.0 + 1 = 1.3
    assert np.isclose(result[0, 0], 1.3)

    # [0,1]: (1.2-1) + (1.0-1) + (1.1-1) + 1 = 0.2 + 0.0 + 0.1 + 1 = 1.3
    assert np.isclose(result[0, 1], 1.3)

    # [1,0]: (1.0-1) + (1.3-1) + (1.0-1) + 1 = 0.0 + 0.3 + 0.0 + 1 = 1.3
    assert np.isclose(result[1, 0], 1.3)

    # [1,1]: (1.5-1) + (1.1-1) + (1.0-1) + 1 = 0.5 + 0.1 + 0.0 + 1 = 1.6
    assert np.isclose(result[1, 1], 1.6)


def test_combine_multiplier_maps_multiplicative():
    """Test multiplicative combination: M1 * M2 * M3."""
    map1 = np.array([[1.1, 1.2], [1.0, 1.5]])
    map2 = np.array([[1.2, 1.0], [1.3, 1.1]])

    result = combine_multiplier_maps([map1, map2], "multiplicative")

    # [0,0]: 1.1 * 1.2 = 1.32
    assert np.isclose(result[0, 0], 1.32)

    # [0,1]: 1.2 * 1.0 = 1.2
    assert np.isclose(result[0, 1], 1.2)

    # [1,0]: 1.0 * 1.3 = 1.3
    assert np.isclose(result[1, 0], 1.3)

    # [1,1]: 1.5 * 1.1 = 1.65
    assert np.isclose(result[1, 1], 1.65)


def test_combine_multiplier_maps_max():
    """Test max combination: max(M1, M2, M3)."""
    map1 = np.array([[1.1, 1.2], [1.0, 1.5]])
    map2 = np.array([[1.2, 1.0], [1.3, 1.1]])
    map3 = np.array([[1.0, 1.5], [1.2, 1.0]])

    result = combine_multiplier_maps([map1, map2, map3], "max")

    # [0,0]: max(1.1, 1.2, 1.0) = 1.2
    assert result[0, 0] == 1.2

    # [0,1]: max(1.2, 1.0, 1.5) = 1.5
    assert result[0, 1] == 1.5

    # [1,0]: max(1.0, 1.3, 1.2) = 1.3
    assert result[1, 0] == 1.3

    # [1,1]: max(1.5, 1.1, 1.0) = 1.5
    assert result[1, 1] == 1.5


def test_combine_multiplier_maps_neutral_value():
    """Test that 1.0 is neutral for all combination methods."""
    map1 = np.array([[1.5]])
    map_neutral = np.array([[1.0]])

    # Additive: 1.5 + 1.0 should give 1.5 (because (1.5-1) + (1.0-1) + 1 = 0.5 + 0 + 1 = 1.5)
    result_add = combine_multiplier_maps([map1, map_neutral], "additive")
    assert np.isclose(result_add[0, 0], 1.5)

    # Multiplicative: 1.5 * 1.0 = 1.5
    result_mult = combine_multiplier_maps([map1, map_neutral], "multiplicative")
    assert np.isclose(result_mult[0, 0], 1.5)

    # Max: max(1.5, 1.0) = 1.5
    result_max = combine_multiplier_maps([map1, map_neutral], "max")
    assert result_max[0, 0] == 1.5


def test_create_density_map_basic():
    """Test basic density map creation."""
    config = CityConfig(
        width=10,
        height=10,
        base_housing_density_km2=1000.0,
        base_office_density_km2=500.0,
        base_shop_density_km2=250.0,
        density_combination_method="additive"
    )

    center = CityCenter(
        position=(5, 5),
        strength=1.0,
        housing_peak_multiplier=2.0,
        office_peak_multiplier=1.5,
        shop_peak_multiplier=1.0,
        decay_rate=0.1
    )

    density_map = create_density_map([center], [], [], config, "additive")

    assert density_map.housing_densities.shape == (10, 10)
    assert density_map.office_densities.shape == (10, 10)
    assert density_map.shop_densities.shape == (10, 10)

    # At center, density = base * peak_multiplier
    assert np.isclose(density_map.housing_densities[5, 5], 1000.0 * 2.0)
    assert np.isclose(density_map.office_densities[5, 5], 500.0 * 1.5)
    assert np.isclose(density_map.shop_densities[5, 5], 250.0 * 1.0)


def test_create_density_map_with_parks():
    """Test that parks zero out all density types."""
    config = CityConfig(
        width=10,
        height=10,
        base_housing_density_km2=1000.0,
        base_office_density_km2=500.0,
        base_shop_density_km2=250.0
    )

    center = CityCenter(
        position=(5, 5),
        strength=1.0,
        housing_peak_multiplier=2.0,
        office_peak_multiplier=1.5,
        shop_peak_multiplier=1.3,
        decay_rate=0.1
    )

    park = Park(
        center=(5, 5),
        size=1,
        blocks={(5, 5)},
        shape="square"
    )

    density_map = create_density_map([center], [], [park], config, "additive")

    # Park blocks should have zero density for all types
    assert density_map.housing_densities[5, 5] == 0.0
    assert density_map.office_densities[5, 5] == 0.0
    assert density_map.shop_densities[5, 5] == 0.0


def test_create_density_map_multiple_centers_additive():
    """Test that multiple centers combine additively."""
    config = CityConfig(
        width=11,
        height=11,
        base_housing_density_km2=1000.0,
        base_office_density_km2=500.0,
        base_shop_density_km2=250.0
    )

    center1 = CityCenter(
        position=(5, 5),
        strength=1.0,
        housing_peak_multiplier=2.0,
        decay_rate=0.5  # High decay for testing
    )

    center2 = CityCenter(
        position=(5, 7),
        strength=1.0,
        housing_peak_multiplier=1.5,
        decay_rate=0.5
    )

    density_map = create_density_map([center1, center2], [], [], config, "additive")

    # At (5, 6) - distance 1 from both centers
    # Center1 multiplier at (5,6): 2.0 * exp(-0.5 * 1) = 1.213
    # Center2 multiplier at (5,6): 1.5 * exp(-0.5 * 1) = 0.910
    # Combined additive: (1.213-1) + (0.910-1) + 1 = 0.213 + (-0.090) + 1 = 1.123
    # Final density: 1000 * 1.123 = 1123
    expected_multiplier = (2.0 * np.exp(-0.5) - 1) + (1.5 * np.exp(-0.5) - 1) + 1
    expected_density = 1000.0 * expected_multiplier
    assert np.isclose(density_map.housing_densities[5, 6], expected_density, rtol=0.01)


def test_density_map_get_density():
    """Test DensityMap.get_density method."""
    housing = np.array([[100, 200], [300, 400]])
    office = np.array([[10, 20], [30, 40]])
    shop = np.array([[1, 2], [3, 4]])

    density_map = DensityMap(
        housing_densities=housing,
        office_densities=office,
        shop_densities=shop,
        grid_rows=2,
        grid_cols=2
    )

    assert density_map.get_density(0, 0, "housing") == 100
    assert density_map.get_density(1, 1, "housing") == 400
    assert density_map.get_density(0, 1, "office") == 30
    assert density_map.get_density(1, 0, "shop") == 2


if __name__ == '__main__':
    tests = [
        test_calculate_center_multiplier_map_exponential_decay,
        test_calculate_center_multiplier_map_different_types,
        test_calculate_corridor_multiplier_map,
        test_combine_multiplier_maps_additive,
        test_combine_multiplier_maps_multiplicative,
        test_combine_multiplier_maps_max,
        test_combine_multiplier_maps_neutral_value,
        test_create_density_map_basic,
        test_create_density_map_with_parks,
        test_create_density_map_multiple_centers_additive,
        test_density_map_get_density
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print("✓ Passed\n")

    print("All density tests passed! ✓")
