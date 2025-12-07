import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, overload, TYPE_CHECKING
from .grid import Grid

if TYPE_CHECKING:
    from .city import City


@overload
def visualize_grid(city_or_grid: Grid, save_path: Optional[str] = None, show: bool = True) -> None: ...

@overload
def visualize_grid(city_or_grid: 'City', save_path: Optional[str] = None, show: bool = True) -> None: ...

def visualize_grid(city_or_grid: Union[Grid, 'City'], save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create a 2-panel visualization showing population and units distribution.

    Args:
        city_or_grid: Grid object or City object to visualize
        save_path: Optional path to save the figure (e.g., 'figures/city_grid.png')
        show: Whether to display the plot (default: True)
    """
    # Extract grid from City object if needed
    grid = city_or_grid
    has_parks = False
    if hasattr(city_or_grid, 'grid'):
        grid = city_or_grid.grid
        has_parks = hasattr(city_or_grid, 'parks') and len(city_or_grid.parks) > 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Create 2D arrays for population and units
    population_grid = np.zeros((grid.height, grid.width))
    units_grid = np.zeros((grid.height, grid.width))
    park_mask = np.zeros((grid.height, grid.width), dtype=bool)

    for block in grid.blocks:
        population_grid[block.y, block.x] = block.population
        units_grid[block.y, block.x] = block.units
        if hasattr(block, 'is_park') and block.is_park:
            park_mask[block.y, block.x] = True

    # --- Panel 1: Population Distribution ---
    ax_pop = axes[0]
    im_pop = ax_pop.imshow(population_grid, cmap='YlOrRd', origin='lower',
                           interpolation='nearest')
    cbar_pop = plt.colorbar(im_pop, ax=ax_pop, fraction=0.046, pad=0.04)
    cbar_pop.set_label('Population', rotation=270, labelpad=25, fontsize=12,
                       fontweight='bold')
    ax_pop.set_title('Population Distribution', fontsize=14, fontweight='bold', pad=20)
    ax_pop.set_xlabel('X Coordinate', fontsize=11)
    ax_pop.set_ylabel('Y Coordinate', fontsize=11)

    # Overlay parks if present
    if has_parks:
        park_overlay = np.ma.masked_where(~park_mask, np.ones_like(park_mask))
        ax_pop.imshow(park_overlay, cmap='Greens', alpha=0.6, origin='lower', vmin=0, vmax=1)

    # Add grid lines
    ax_pop.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax_pop.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax_pop.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_pop.set_xticks(np.arange(0, grid.width, 5))
    ax_pop.set_yticks(np.arange(0, grid.height, 5))

    # --- Panel 2: Housing Units Distribution ---
    ax_units = axes[1]
    im_units = ax_units.imshow(units_grid, cmap='viridis', origin='lower',
                               interpolation='nearest')
    cbar_units = plt.colorbar(im_units, ax=ax_units, fraction=0.046, pad=0.04)
    cbar_units.set_label('Housing Units', rotation=270, labelpad=25, fontsize=12,
                         fontweight='bold')
    ax_units.set_title('Housing Units Distribution', fontsize=14, fontweight='bold', pad=20)
    ax_units.set_xlabel('X Coordinate', fontsize=11)
    ax_units.set_ylabel('Y Coordinate', fontsize=11)

    # Overlay parks if present
    if has_parks:
        park_overlay = np.ma.masked_where(~park_mask, np.ones_like(park_mask))
        ax_units.imshow(park_overlay, cmap='Greens', alpha=0.6, origin='lower', vmin=0, vmax=1)

    # Add grid lines
    ax_units.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax_units.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax_units.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_units.set_xticks(np.arange(0, grid.width, 5))
    ax_units.set_yticks(np.arange(0, grid.height, 5))

    # Overall title
    fig.suptitle(f'City Grid Visualization ({grid.width}x{grid.height})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


@overload
def visualize_population(city_or_grid: Grid, save_path: Optional[str] = None, show: bool = True) -> None: ...

@overload
def visualize_population(city_or_grid: 'City', save_path: Optional[str] = None, show: bool = True) -> None: ...

def visualize_population(city_or_grid: Union[Grid, 'City'], save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create a single-panel visualization showing only population distribution.

    Args:
        city_or_grid: Grid object or City object to visualize
        save_path: Optional path to save the figure
        show: Whether to display the plot (default: True)
    """
    # Extract grid from City object if needed
    grid = city_or_grid
    if hasattr(city_or_grid, 'grid'):
        grid = city_or_grid.grid

    fig, ax = plt.subplots(figsize=(10, 9))

    # Create 2D array for population
    population_grid = np.zeros((grid.height, grid.width))
    for block in grid.blocks:
        population_grid[block.y, block.x] = block.population

    im = ax.imshow(population_grid, cmap='YlOrRd', origin='lower',
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Population', rotation=270, labelpad=25, fontsize=12,
                   fontweight='bold')

    ax.set_title(f'Population Distribution ({grid.width}x{grid.height})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(0, grid.width, 5))
    ax.set_yticks(np.arange(0, grid.height, 5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


@overload
def visualize_units(city_or_grid: Grid, save_path: Optional[str] = None, show: bool = True) -> None: ...

@overload
def visualize_units(city_or_grid: 'City', save_path: Optional[str] = None, show: bool = True) -> None: ...

def visualize_units(city_or_grid: Union[Grid, 'City'], save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create a single-panel visualization showing only housing units distribution.

    Args:
        city_or_grid: Grid object or City object to visualize
        save_path: Optional path to save the figure
        show: Whether to display the plot (default: True)
    """
    # Extract grid from City object if needed
    grid = city_or_grid
    if hasattr(city_or_grid, 'grid'):
        grid = city_or_grid.grid

    fig, ax = plt.subplots(figsize=(10, 9))

    # Create 2D array for units
    units_grid = np.zeros((grid.height, grid.width))
    for block in grid.blocks:
        units_grid[block.y, block.x] = block.units

    im = ax.imshow(units_grid, cmap='viridis', origin='lower',
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Housing Units', rotation=270, labelpad=25, fontsize=12,
                   fontweight='bold')

    ax.set_title(f'Housing Units Distribution ({grid.width}x{grid.height})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(0, grid.width, 5))
    ax.set_yticks(np.arange(0, grid.height, 5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


@overload
def print_grid_summary(city_or_grid: Grid) -> None: ...

@overload
def print_grid_summary(city_or_grid: 'City') -> None: ...

def print_grid_summary(city_or_grid: Union[Grid, 'City']) -> None:
    """
    Print summary statistics about the grid.

    Args:
        city_or_grid: Grid object or City object to summarize
    """
    # Extract grid from City object if needed
    grid = city_or_grid
    if hasattr(city_or_grid, 'grid'):
        grid = city_or_grid.grid

    populations = [block.population for block in grid.blocks]
    units = [block.units for block in grid.blocks]

    print(f"\n{'='*60}")
    print(f"GRID SUMMARY")
    print(f"{'='*60}")
    print(f"Grid Size: {grid.width}x{grid.height} ({grid.num_blocks} blocks)")
    print(f"\nPopulation Statistics:")
    print(f"  Total Population: {sum(populations):,.2f}")
    print(f"  Average per Block: {np.mean(populations):.2f}")
    print(f"  Median per Block: {np.median(populations):.2f}")
    print(f"  Min: {min(populations):.2f}")
    print(f"  Max: {max(populations):.2f}")
    print(f"  Std Dev: {np.std(populations):.2f}")

    print(f"\nHousing Units Statistics:")
    print(f"  Total Units: {sum(units):,}")
    print(f"  Average per Block: {np.mean(units):.2f}")
    print(f"  Median per Block: {np.median(units):.0f}")
    print(f"  Min: {min(units)}")
    print(f"  Max: {max(units)}")
    print(f"  Std Dev: {np.std(units):.2f}")

    # Calculate density (persons per unit)
    total_pop = sum(populations)
    total_units = sum(units)
    if total_units > 0:
        avg_density = total_pop / total_units
        print(f"\nAverage Density: {avg_density:.2f} persons/unit")

    print(f"{'='*60}\n")


@overload
def visualize_with_corridors(city_or_grid: Grid,
                             centers: List[dict],
                             corridors: List,
                             save_path: Optional[str] = None,
                             show: bool = True) -> None: ...

@overload
def visualize_with_corridors(city_or_grid: 'City',
                             centers: Optional[List[dict]] = None,
                             corridors: Optional[List] = None,
                             save_path: Optional[str] = None,
                             show: bool = True) -> None: ...

def visualize_with_corridors(city_or_grid: Union[Grid, 'City'],
                             centers: Optional[List[dict]] = None,
                             corridors: Optional[List] = None,
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
    """
    Create a 3-panel visualization showing population, corridors, and combined view.

    Args:
        city_or_grid: Grid object or City object to visualize
        centers: List of center dictionaries with 'position' and 'strength' keys
                 (optional if City object is provided)
        corridors: List of TransportationCorridor objects (optional if City object is provided)
        save_path: Optional path to save the figure
        show: Whether to display the plot (default: True)
    """
    # Extract grid, centers, and corridors from City object if needed
    grid = city_or_grid
    if hasattr(city_or_grid, 'grid'):
        city = city_or_grid
        grid = city.grid
        if centers is None:
            centers = city.centers
        if corridors is None:
            corridors = city.corridors

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Create 2D arrays
    population_grid = np.zeros((grid.height, grid.width))
    corridor_grid = np.zeros((grid.height, grid.width))

    # Mark corridor blocks
    if corridors:
        for corridor in corridors:
            for y, x in corridor.blocks:
                corridor_grid[y, x] = 1

    for block in grid.blocks:
        population_grid[block.y, block.x] = block.population

    # --- Panel 1: Population Distribution ---
    ax_pop = axes[0]
    im_pop = ax_pop.imshow(population_grid, cmap='YlOrRd', origin='lower',
                           interpolation='nearest')
    cbar_pop = plt.colorbar(im_pop, ax=ax_pop, fraction=0.046, pad=0.04)
    cbar_pop.set_label('Population', rotation=270, labelpad=20, fontsize=11)
    ax_pop.set_title('Population Distribution', fontsize=13, fontweight='bold', pad=15)
    ax_pop.set_xlabel('X Coordinate', fontsize=10)
    ax_pop.set_ylabel('Y Coordinate', fontsize=10)

    # Mark centers
    for i, center in enumerate(centers):
        # Handle both dict format (legacy) and CityCenter objects
        if isinstance(center, dict):
            row, col = center['position']
            strength = center['strength']
        else:
            row, col = center.position
            strength = center.strength
        ax_pop.scatter(col, row, s=200 * strength, c='blue',
                      marker='*', edgecolors='white', linewidths=2,
                      label=f"Center {i+1}" if i < 3 else "")

    ax_pop.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_pop.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # --- Panel 2: Transportation Corridors ---
    ax_corr = axes[1]
    ax_corr.imshow(corridor_grid, cmap='Greys', origin='lower',
                   interpolation='nearest', vmin=0, vmax=1)
    ax_corr.set_title('Transportation Corridors', fontsize=13, fontweight='bold', pad=15)
    ax_corr.set_xlabel('X Coordinate', fontsize=10)
    ax_corr.set_ylabel('Y Coordinate', fontsize=10)

    # Mark centers
    for i, center in enumerate(centers):
        # Handle both dict format (legacy) and CityCenter objects
        if isinstance(center, dict):
            row, col = center['position']
            strength = center['strength']
        else:
            row, col = center.position
            strength = center.strength
        ax_corr.scatter(col, row, s=200 * strength, c='red',
                       marker='*', edgecolors='white', linewidths=2)

    ax_corr.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_corr.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # --- Panel 3: Combined View ---
    ax_comb = axes[2]
    im_comb = ax_comb.imshow(population_grid, cmap='YlOrRd', origin='lower',
                             interpolation='nearest', alpha=0.7)
    ax_comb.imshow(corridor_grid, cmap='Blues', origin='lower',
                   interpolation='nearest', alpha=0.4)
    cbar_comb = plt.colorbar(im_comb, ax=ax_comb, fraction=0.046, pad=0.04)
    cbar_comb.set_label('Population', rotation=270, labelpad=20, fontsize=11)
    ax_comb.set_title('Combined View', fontsize=13, fontweight='bold', pad=15)
    ax_comb.set_xlabel('X Coordinate', fontsize=10)
    ax_comb.set_ylabel('Y Coordinate', fontsize=10)

    # Mark centers
    for i, center in enumerate(centers):
        # Handle both dict format (legacy) and CityCenter objects
        if isinstance(center, dict):
            row, col = center['position']
            strength = center['strength']
        else:
            row, col = center.position
            strength = center.strength
        ax_comb.scatter(col, row, s=200 * strength, c='blue',
                       marker='*', edgecolors='white', linewidths=2)

    ax_comb.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_comb.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # Overall title
    corridor_types = []
    if corridors:
        corridor_types = [c.corridor_type for c in corridors]
    corridor_type_str = ', '.join(corridor_types) if corridor_types else 'N/A'
    fig.suptitle(f'City with Transportation Corridors ({corridor_type_str})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_zoning(city, save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create visualizations showing zoning information (density and uses).

    Args:
        city: City object to visualize
        save_path: Optional path to save the figure
        show: Whether to display the plot (default: True)
    """
    from .zoning import Use, Density

    grid = city.grid

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Create grids for zoning data
    density_grid = np.zeros((grid.height, grid.width))
    residential_grid = np.zeros((grid.height, grid.width))
    commercial_grid = np.zeros((grid.height, grid.width))
    office_grid = np.zeros((grid.height, grid.width))

    for block in grid.blocks:
        if hasattr(block, 'zoning') and block.zoning:
            # Map density to numeric values
            density_grid[block.y, block.x] = block.zoning.max_density.value

            # Mark allowed uses
            if block.zoning.allows_use(Use.RESIDENTIAL):
                residential_grid[block.y, block.x] = 1
            if block.zoning.allows_use(Use.COMMERCIAL):
                commercial_grid[block.y, block.x] = 1
            if block.zoning.allows_use(Use.OFFICE):
                office_grid[block.y, block.x] = 1

    # Panel 1: Density Levels
    ax_density = axes[0, 0]
    im_density = ax_density.imshow(density_grid, cmap='RdYlGn', origin='lower',
                                     interpolation='nearest', vmin=1, vmax=3)
    cbar_density = plt.colorbar(im_density, ax=ax_density, fraction=0.046, pad=0.04,
                                  ticks=[1, 2, 3])
    cbar_density.set_ticklabels(['Low', 'Medium', 'High'])
    cbar_density.set_label('Max Density', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    ax_density.set_title('Zoning Density Levels', fontsize=14, fontweight='bold', pad=20)
    ax_density.set_xlabel('X Coordinate', fontsize=11)
    ax_density.set_ylabel('Y Coordinate', fontsize=11)
    ax_density.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_density.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # Panel 2: Residential Zoning
    ax_res = axes[0, 1]
    im_res = ax_res.imshow(residential_grid, cmap='Blues', origin='lower',
                            interpolation='nearest', vmin=0, vmax=1)
    ax_res.set_title('Residential Zoning', fontsize=14, fontweight='bold', pad=20)
    ax_res.set_xlabel('X Coordinate', fontsize=11)
    ax_res.set_ylabel('Y Coordinate', fontsize=11)
    ax_res.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_res.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # Panel 3: Commercial Zoning
    ax_com = axes[1, 0]
    im_com = ax_com.imshow(commercial_grid, cmap='Oranges', origin='lower',
                            interpolation='nearest', vmin=0, vmax=1)
    ax_com.set_title('Commercial Zoning', fontsize=14, fontweight='bold', pad=20)
    ax_com.set_xlabel('X Coordinate', fontsize=11)
    ax_com.set_ylabel('Y Coordinate', fontsize=11)
    ax_com.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_com.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # Panel 4: Office Zoning
    ax_off = axes[1, 1]
    im_off = ax_off.imshow(office_grid, cmap='Purples', origin='lower',
                            interpolation='nearest', vmin=0, vmax=1)
    ax_off.set_title('Office Zoning', fontsize=14, fontweight='bold', pad=20)
    ax_off.set_xlabel('X Coordinate', fontsize=11)
    ax_off.set_ylabel('Y Coordinate', fontsize=11)
    ax_off.set_xticks(np.arange(0, grid.width, max(1, grid.width // 10)))
    ax_off.set_yticks(np.arange(0, grid.height, max(1, grid.height // 10)))

    # Overall title
    fig.suptitle(f'City Zoning Map ({grid.width}x{grid.height})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
