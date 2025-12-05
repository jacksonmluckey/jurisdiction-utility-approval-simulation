import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .grid import Grid


def visualize_grid(grid: Grid, save_path: Optional[str] = None, show: bool = True):
    """
    Create a 2-panel visualization showing population and units distribution.

    Args:
        grid: Grid object to visualize
        save_path: Optional path to save the figure (e.g., 'figures/city_grid.png')
        show: Whether to display the plot (default: True)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Create 2D arrays for population and units
    population_grid = np.zeros((grid.height, grid.width))
    units_grid = np.zeros((grid.height, grid.width))

    for block in grid.blocks:
        population_grid[block.y, block.x] = block.population
        units_grid[block.y, block.x] = block.units

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


def visualize_population(grid: Grid, save_path: Optional[str] = None, show: bool = True):
    """
    Create a single-panel visualization showing only population distribution.

    Args:
        grid: Grid object to visualize
        save_path: Optional path to save the figure
        show: Whether to display the plot (default: True)
    """
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


def visualize_units(grid: Grid, save_path: Optional[str] = None, show: bool = True):
    """
    Create a single-panel visualization showing only housing units distribution.

    Args:
        grid: Grid object to visualize
        save_path: Optional path to save the figure
        show: Whether to display the plot (default: True)
    """
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


def print_grid_summary(grid: Grid):
    """
    Print summary statistics about the grid.

    Args:
        grid: Grid object to summarize
    """
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
