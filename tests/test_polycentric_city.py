"""Tests for PolycentricCity class and transportation integration."""

import sys
from pathlib import Path
import io
from contextlib import redirect_stdout

# Add parent directory to path to import city modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from city.polycentric_city import PolycentricCity, PolycentricConfig
from city.transportation_corridor import TransportationConfig, CorridorType


def test_visualize_summary_without_transport():
    """Test that visualize_summary works without transportation config."""
    config = PolycentricConfig(
        num_centers=2,
        center_distribution="uniform",
        primary_density=18.0,
        density_decay_rate=0.15
    )

    city = PolycentricCity(
        grid_rows=20,
        grid_cols=20,
        config=config
    )

    grid = city.generate()

    # Capture output to verify no exceptions are raised
    output = io.StringIO()
    with redirect_stdout(output):
        city.visualize_summary()

    result = output.getvalue()
    assert "City Grid:" in result
    assert "Number of centers:" in result
    assert "Transportation Network:" not in result  # No transport config


def test_visualize_summary_with_single_transport():
    """Test that visualize_summary works with a single transportation config."""
    config = PolycentricConfig(
        num_centers=3,
        center_distribution="uniform",
        primary_density=18.0,
        density_decay_rate=0.15
    )

    transport_config = TransportationConfig(
        corridor_type=CorridorType.INTER_CENTER,
        corridor_width_blocks=2,
        density_multiplier=1.20,
        connect_all_centers=True
    )

    city = PolycentricCity(
        grid_rows=30,
        grid_cols=30,
        config=config,
        transport_configs=[transport_config]
    )

    grid = city.generate()

    # Capture output to verify no exceptions are raised
    output = io.StringIO()
    with redirect_stdout(output):
        city.visualize_summary()

    result = output.getvalue()
    assert "City Grid:" in result
    assert "Number of centers:" in result
    assert "Transportation Network:" in result
    assert "Corridor 1:" in result
    assert "Type:" in result
    assert "Width:" in result
    assert "Density boost:" in result


def test_visualize_summary_with_multiple_transports():
    """Test that visualize_summary works with multiple transportation configs."""
    config = PolycentricConfig(
        num_centers=4,
        center_distribution="uniform",
        primary_density=18.0,
        density_decay_rate=0.15
    )

    transport_configs = [
        TransportationConfig(
            corridor_type=CorridorType.INTER_CENTER,
            corridor_width_blocks=2,
            density_multiplier=1.20,
            connect_all_centers=True
        ),
        TransportationConfig(
            corridor_type=CorridorType.RADIAL,
            corridor_width_blocks=1,
            density_multiplier=1.10,
            radial_corridors_count=4
        )
    ]

    city = PolycentricCity(
        grid_rows=40,
        grid_cols=40,
        config=config,
        transport_configs=transport_configs
    )

    grid = city.generate()

    # Capture output to verify no exceptions are raised
    output = io.StringIO()
    with redirect_stdout(output):
        city.visualize_summary()

    result = output.getvalue()
    assert "City Grid:" in result
    assert "Number of centers:" in result
    assert "Transportation Network:" in result
    assert "Corridor 1:" in result
    assert "Corridor 2:" in result


def test_corridor_info_dict_structure():
    """Test that get_corridor_info returns the expected dictionary structure."""
    config = PolycentricConfig(
        num_centers=3,
        center_distribution="uniform",
        primary_density=18.0,
        density_decay_rate=0.15
    )

    transport_config = TransportationConfig(
        corridor_type=CorridorType.INTER_CENTER,
        corridor_width_blocks=2,
        density_multiplier=1.20,
        connect_all_centers=True
    )

    city = PolycentricCity(
        grid_rows=30,
        grid_cols=30,
        config=config,
        transport_configs=[transport_config]
    )

    grid = city.generate()

    corridor_info = city.transport_network.get_corridor_info()

    # Verify all expected keys exist
    assert 'total_corridor_blocks' in corridor_info
    assert 'corridor_coverage_pct' in corridor_info
    assert 'num_corridor_configs' in corridor_info
    assert 'corridor_configs' in corridor_info
    assert 'average_density_boost' in corridor_info

    # Verify corridor_configs structure
    assert isinstance(corridor_info['corridor_configs'], list)
    assert len(corridor_info['corridor_configs']) == 1

    config_info = corridor_info['corridor_configs'][0]
    assert 'index' in config_info
    assert 'type' in config_info  # This is the key that caused the bug!
    assert 'width_blocks' in config_info
    assert 'density_multiplier' in config_info
    assert 'density_boost_pct' in config_info

    # Verify the bug doesn't exist: 'corridor_type' should NOT be a key
    # in the top-level dict or in config_info
    assert 'corridor_type' not in corridor_info
    assert 'corridor_type' not in config_info


def test_corridor_info_with_multiple_configs():
    """Test corridor_info structure with multiple transport configurations."""
    config = PolycentricConfig(
        num_centers=4,
        center_distribution="uniform",
        primary_density=18.0,
        density_decay_rate=0.15
    )

    transport_configs = [
        TransportationConfig(
            corridor_type=CorridorType.INTER_CENTER,
            corridor_width_blocks=2,
            density_multiplier=1.20
        ),
        TransportationConfig(
            corridor_type=CorridorType.RADIAL,
            corridor_width_blocks=1,
            density_multiplier=1.15
        ),
        TransportationConfig(
            corridor_type=CorridorType.RING,
            corridor_width_blocks=1,
            density_multiplier=1.10
        )
    ]

    city = PolycentricCity(
        grid_rows=40,
        grid_cols=40,
        config=config,
        transport_configs=transport_configs
    )

    grid = city.generate()
    corridor_info = city.transport_network.get_corridor_info()

    # Verify we have info for all three configs
    assert corridor_info['num_corridor_configs'] == 3
    assert len(corridor_info['corridor_configs']) == 3

    # Verify each config has the correct structure
    for i, config_info in enumerate(corridor_info['corridor_configs']):
        assert config_info['index'] == i
        assert 'type' in config_info
        assert 'width_blocks' in config_info
        assert 'density_multiplier' in config_info
        assert 'density_boost_pct' in config_info

        # Verify corridor_type is NOT present (the bug)
        assert 'corridor_type' not in config_info


if __name__ == '__main__':
    print("Running test_visualize_summary_without_transport...")
    test_visualize_summary_without_transport()
    print("✓ Passed\n")

    print("Running test_visualize_summary_with_single_transport...")
    test_visualize_summary_with_single_transport()
    print("✓ Passed\n")

    print("Running test_visualize_summary_with_multiple_transports...")
    test_visualize_summary_with_multiple_transports()
    print("✓ Passed\n")

    print("Running test_corridor_info_dict_structure...")
    test_corridor_info_dict_structure()
    print("✓ Passed\n")

    print("Running test_corridor_info_with_multiple_configs...")
    test_corridor_info_with_multiple_configs()
    print("✓ Passed\n")

    print("All tests passed! ✓")
