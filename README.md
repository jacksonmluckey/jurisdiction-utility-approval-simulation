# Jurisdiction Utility Approval Simulation

A Python simulation framework for generating realistic urban environments with configurable density patterns, transportation networks, zoning, parks, and commercial development.

## Features

### Urban Density Patterns
- **Polycentric Cities**: Generate cities with multiple activity centers, each with configurable strength and density decay
- **Flexible Density Control**: Set maximum and minimum housing density in units per km�
- **Noise and Variation**: Add realistic variation to housing units using proportional or custom noise functions

### Commercial Development
- **Office Districts**: Offices concentrate near city centers with exponential decay, creating realistic central business districts
- **Retail/Shops**: Shops cluster at centers and along transportation corridors, simulating commercial activity patterns
- **Zoning Integration**: Commercial development respects zoning regulations when enabled

### Transportation Networks
- **Multiple Corridor Types**:
  - Inter-center corridors connecting activity centers
  - Grid-pattern transit networks
  - Radial highways from centers
- **Density Multipliers**: Transportation corridors boost residential and commercial density
- **Shop Corridor Boost**: Retail units receive additional density multiplier along transit corridors

### Zoning System
- **Density Levels**: Low, medium, and high-density zoning based on unit counts
- **Use Permissions**: Residential, commercial, and office use regulations
- **Mixed-Use Districts**: Central areas can be zoned for multiple uses
- **Center-Based Rules**: Activity centers automatically receive mixed-use high-density zoning
- **Automatic Upzoning**: Configurable feature that automatically upzones blocks when surrounded by higher-density or additional-use neighbors
  - Density upzoning: Increases density level when enough neighbors have higher density
  - Use upzoning: Adds commercial/office uses when enough neighbors allow those uses
  - Configurable thresholds for both density and use upzoning
  - Option to include or exclude diagonal neighbors
  - Multiple iteration support for spreading upzoning effects

### Parks and Green Space
- **Configurable Parks**: Set number, size, and distribution of parks
- **Placement Strategies**: Random or dispersed placement patterns
- **Shape Options**: Square or circular park boundaries

### Population Dynamics
- **Variable Household Size**: Set constant or density-dependent household sizes
- **Noise Functions**: Apply realistic variation to population estimates

## Installation

This project uses `uv` for dependency management. To run the simulation:

```bash
uv run simulation.py
```

## Basic Usage

```python
from city import City, CityConfig, CityCentersConfig, ParkConfig, ZoningConfig

# Configure the city
config = CityConfig(
    width=50,
    height=50,
    block_size_meters=100.0,
    max_density_units_per_km2=12355.0,
    min_density_units_per_km2=124.0,
    max_density_offices_per_km2=7413.0,
    max_density_shops_per_km2=4942.0,
    persons_per_unit=2.5
)

# Configure city centers density pattern
centers = CityCentersConfig(
    num_centers=3,
    primary_density_km2=6178.0,
    density_decay_rate=0.20
)

# Configure parks
parks = ParkConfig(
    num_parks=5,
    min_size_blocks=2,
    max_size_blocks=9,
    placement_strategy="dispersed"
)

# Configure zoning
zoning = ZoningConfig(enabled=True)

# Create and generate the city
city = City(
    config=config,
    centers_config=centers,
    park_configs=parks,
    zoning_config=zoning
)
city.generate()

# View summary and visualize
city.summary()
city.visualize()
```

## Configuration Details

### Block Size and Density
- **Block Size**: Specified in meters (default: 100m � 100m)
- **Block Area**: Automatically calculated as km� from block size
  - 100m � 100m = 0.01 km�
- **Density Units**: All density parameters are in units per km�
  - Suburban: 1235-12355 units/km�
  - Urban: 12355-37065 units/km�
  - High-Density Urban: 37065-91435+ units/km�

### Commercial Density Parameters
- `max_density_offices_per_km2`: Maximum office units per km� (default: 7413.0)
- `max_density_shops_per_km2`: Maximum retail units per km� (default: 4942.0)
- `office_center_concentration`: Controls office concentration at centers (default: 0.15)
- `shop_center_concentration`: Controls shop concentration at centers (default: 0.10)
- `shop_corridor_multiplier`: Density boost for shops along transit corridors (default: 1.3)

### Zoning Configuration
```python
zoning = ZoningConfig(
    enabled=True,
    center_radius_blocks=3,  # Mixed-use zone around centers
    low_density_threshold=30,  # Units per block
    medium_density_threshold=80,
    residential_weight=1.0,  # Weight for residential zoning
    commercial_weight=0.3,  # Weight for commercial zoning
    office_weight=0.5,  # Weight for office zoning in high-density areas
    # Automatic upzoning options
    auto_upzone_enabled=True,
    auto_upzone_density_threshold=3,  # Number of higher-density neighbors needed
    auto_upzone_use_threshold=3,  # Number of neighbors with additional use needed
    auto_upzone_include_diagonals=True,  # Include diagonal neighbors (8 total vs 4)
    auto_upzone_iterations=2  # Number of upzoning passes
)
```

#### Automatic Upzoning

The automatic upzoning feature allows zoning to spread organically from high-density centers:

**Density Upzoning**: A block's density level is automatically increased if enough adjacent blocks have higher density.
- LOW → MEDIUM when `auto_upzone_density_threshold` neighbors are MEDIUM or HIGH
- MEDIUM → HIGH when `auto_upzone_density_threshold` neighbors are HIGH

**Use Upzoning**: Additional uses (commercial, office) are added to blocks when enough neighbors allow those uses.
- COMMERCIAL use added when `auto_upzone_use_threshold` neighbors allow commercial
- OFFICE use added when `auto_upzone_use_threshold` neighbors allow office

**Neighbor Counting**:
- `auto_upzone_include_diagonals=True`: Counts all 8 surrounding blocks (default)
- `auto_upzone_include_diagonals=False`: Counts only 4 cardinal direction blocks

**Iterations**: Setting `auto_upzone_iterations` to higher values allows upzoning to spread further from activity centers. Each iteration applies upzoning rules to the entire grid, allowing changes to propagate outward.

**Example Use Cases**:
```python
# Conservative upzoning - only upzone with strong neighbor support
conservative = ZoningConfig(
    auto_upzone_enabled=True,
    auto_upzone_density_threshold=5,  # Need 5 neighbors
    auto_upzone_use_threshold=5,
    auto_upzone_include_diagonals=True,  # Out of 8 possible
    auto_upzone_iterations=1
)

# Aggressive upzoning - spread development widely
aggressive = ZoningConfig(
    auto_upzone_enabled=True,
    auto_upzone_density_threshold=2,  # Only need 2 neighbors
    auto_upzone_use_threshold=2,
    auto_upzone_include_diagonals=False,  # Out of 4 possible
    auto_upzone_iterations=5  # Multiple passes
)
```

## Output Statistics

The city summary includes:
- Total housing units, offices, and shops
- Total population and average density
- Activity center locations and peak densities
- Transportation network coverage
- Park distribution
- Zoning breakdown by density and use type

## Example: High-Density Urban City

```python
urban = CityConfig(
    width=80,
    height=80,
    block_size_meters=100.0,
    max_density_units_per_km2=24710.0,
    max_density_offices_per_km2=14826.0,
    max_density_shops_per_km2=9884.0,
    persons_per_unit=2.0
)

centers = CityCentersConfig(
    num_centers=7,
    primary_density_km2=9884.0,
    density_decay_rate=0.08
)

city = City(config=urban, centers_config=centers)
city.generate()
```

## Testing

Run the test suite:

```bash
uv run python tests/test_city.py
```

## License

See LICENSE file for details.
