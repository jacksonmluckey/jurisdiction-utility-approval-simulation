# City Module

A Python package for simulating urban spatial structure and land use patterns.

## Overview

This module creates grid-based city simulations with configurable population distribution, housing density, zoning regulations, transportation networks, and parks.

## Core Components

- **Block**: Individual grid cells representing city blocks with population, housing units, and land use
- **Grid**: Spatial container organizing blocks into a 2D city layout
- **District**: Collections of blocks for administrative or analytical grouping
- **City**: Main simulation class with monocentric (single-center) population patterns
- **PolycentricCity**: Multi-center city model with multiple employment/population hubs

## Features

- **Population Distribution**: Exponential decay from city center(s) with configurable parameters
- **Zoning**: Land use regulations (residential, commercial, mixed) with density controls
- **Transportation**: Road/transit corridors that influence development patterns
- **Parks**: Configurable green space with various placement strategies
- **Visualization**: Built-in plotting for population, density, zoning, and infrastructure

## Basic Usage

```python
from city import City, CityConfig

# Create a simple city
city = City(CityConfig(width=50, height=50))

# Add features
city.add_transportation(num_corridors=3)
city.add_parks(num_parks=5)
city.add_zoning()

# Visualize
city.visualize()
```
