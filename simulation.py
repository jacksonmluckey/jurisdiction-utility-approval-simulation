import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from city import Grid
from politics import (
    assign_districts_quadrants,
    assign_districts_diagonal,
    calculate_approval_score,
    voter_utility,
    get_districts
)

# Set random seed for reproducibility
np.random.seed(19972025)


def to_dataframe(grid: Grid, district_assignments_pre: Dict[int, int],
                 district_assignments_post: Dict[int, int],
                 approval_scores_pre: Dict[int, float],
                 approval_scores_post: Dict[int, float]) -> pd.DataFrame:
    """Convert blocks and assignments to pandas DataFrame"""
    return pd.DataFrame([
        {
            'block_id': b.block_id,
            'x': b.x,
            'y': b.y,
            'population': b.population,
            'units': b.units,
            'district_pre': district_assignments_pre[b.block_id],
            'district_post': district_assignments_post[b.block_id],
            'approval_pre': approval_scores_pre[b.block_id],
            'approval_post': approval_scores_post[b.block_id],
            'approval_change': approval_scores_post[b.block_id] - approval_scores_pre[b.block_id]
        }
        for b in grid.blocks
    ])


def create_figures(grid: Grid, blocks_df: pd.DataFrame, num_districts: int):
    """Create all visualization figures"""

    # --- FIGURE 1: Utility Function ---
    test_distances = np.linspace(0.1, 10, 100)
    test_utilities = [voter_utility(d) for d in test_distances]

    # Find where utility crosses zero
    zero_crossing_idx = np.where(np.diff(np.sign(test_utilities)))[0]
    if len(zero_crossing_idx) > 0:
        idx = zero_crossing_idx[0]
        x1, x2 = test_distances[idx], test_distances[idx + 1]
        y1, y2 = test_utilities[idx], test_utilities[idx + 1]
        zero_distance = x1 - y1 * (x2 - x1) / (y2 - y1)
    else:
        zero_distance = None

    plt.figure(figsize=(10, 6))
    plt.plot(test_distances, test_utilities, linewidth=2, color='#2E86AB')

    if zero_distance is not None:
        plt.axvline(x=zero_distance, color='#A23B72', linestyle='--', alpha=0.7, linewidth=2,
                    label=f'Approval threshold ({zero_distance:.2f} blocks)')

    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('Distance from development site (blocks)', fontsize=12)
    plt.ylabel('Voter Utility', fontsize=12)
    plt.title('Voter Utility as Function of Distance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/utility_function.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- FIGURE 2: 2x2 Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Generate colormap for districts
    colors = plt.cm.Set3(np.linspace(0, 1, num_districts))
    district_cmap = ListedColormap(colors)
    legend_elements = [Patch(facecolor=colors[i], label=f'District {i}')
                       for i in range(num_districts)]

    for ax_idx, (ax, dist_col, app_col, title_dist, title_app) in enumerate([
        (axes[0, 0], 'district_pre', 'approval_pre', 'Pre-Redistricting Districts',
         'Pre-Redistricting Approval Scores'),
        (axes[1, 0], 'district_post', 'approval_post', 'Post-Redistricting Districts',
         'Post-Redistricting Approval Scores')
    ]):
        # Districts map
        grid_dist = np.zeros((grid.height, grid.width))
        for idx, row in blocks_df.iterrows():
            grid_dist[int(row['y']), int(row['x'])] = row[dist_col]

        ax.imshow(grid_dist, cmap=district_cmap, origin='lower', interpolation='nearest',
                  vmin=0, vmax=num_districts-1)
        ax.set_title(title_dist, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
        ax.set_xticks(np.arange(0, grid.width, 1))
        ax.set_yticks(np.arange(0, grid.height, 1))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

        # Approval scores map
        ax_app = axes[ax_idx, 1]
        grid_app = np.zeros((grid.height, grid.width))
        for idx, row in blocks_df.iterrows():
            grid_app[int(row['y']), int(row['x'])] = row[app_col]

        im = ax_app.imshow(grid_app, cmap='RdYlGn', vmin=0, vmax=1, origin='lower',
                           interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax_app, fraction=0.046, pad=0.04)
        cbar.set_label('Approval Score', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
        ax_app.set_title(title_app, fontsize=14, fontweight='bold', pad=20)
        ax_app.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
        ax_app.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
        ax_app.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_app.set_xticks(np.arange(0, grid.width, 1))
        ax_app.set_yticks(np.arange(0, grid.height, 1))

    fig.suptitle('Impact of Redistricting on Development Approval Scores',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('figures/redistricting_comparison_2x2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- FIGURE 3: Approval Change ---
    grid_change = np.zeros((grid.height, grid.width))
    for idx, row in blocks_df.iterrows():
        grid_change[int(row['y']), int(row['x'])] = row['approval_change']

    plt.figure(figsize=(10, 8))
    im = plt.imshow(grid_change, cmap='RdBu_r', origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, label='Approval Change', shrink=0.8)
    plt.title('Change in Approval Score (Post - Pre)', fontsize=14, fontweight='bold')
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/approval_change.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_statistics(grid: Grid, num_districts: int,
                            district_assignments_pre: Dict[int, int],
                            district_assignments_post: Dict[int, int]):
    """Print summary statistics about the grid and districts"""
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Grid Size: {grid.width}x{grid.height} ({grid.num_blocks} blocks)")
    print(f"Number of Districts: {num_districts}")
    print(f"Total Population: {grid.total_population:,.2f}")
    target_population = grid.total_population / num_districts
    print(f"Target Population per District: {target_population:,.2f}")

    # Pre-redistricting statistics
    print(f"\n{'-'*60}")
    print("PRE-REDISTRICTING DISTRICT POPULATIONS:")
    print(f"{'-'*60}")
    districts_pre = get_districts(grid, district_assignments_pre)
    for i in range(num_districts):
        district = districts_pre.get(i)
        if district:
            pct_of_target = (district.total_population / target_population) * 100
            print(f"  District {i}: {district.total_population:>8,.2f} pop "
                  f"({district.num_blocks:>3} blocks, {district.total_units:>3} units) "
                  f"[{pct_of_target:.1f}% of target]")

    # Post-redistricting statistics
    print(f"\n{'-'*60}")
    print("POST-REDISTRICTING DISTRICT POPULATIONS:")
    print(f"{'-'*60}")
    districts_post = get_districts(grid, district_assignments_post)
    for i in range(num_districts):
        district = districts_post.get(i)
        if district:
            pct_of_target = (district.total_population / target_population) * 100
            print(f"  District {i}: {district.total_population:>8,.2f} pop "
                  f"({district.num_blocks:>3} blocks, {district.total_units:>3} units) "
                  f"[{pct_of_target:.1f}% of target]")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Configuration parameters
    GRID_WIDTH = 50
    GRID_HEIGHT = 50
    NUM_DISTRICTS = 10

    print("Creating grid and assigning initial populations...")
    grid = Grid(width=GRID_WIDTH, height=GRID_HEIGHT)

    # Initialize district assignments
    district_assignments_pre = {}
    district_assignments_post = {}
    approval_scores_pre = {}
    approval_scores_post = {}

    print("Assigning pre-redistricting districts...")
    assign_districts_quadrants(grid, NUM_DISTRICTS, district_assignments_pre)

    print("Assigning post-redistricting districts...")
    assign_districts_diagonal(grid, NUM_DISTRICTS, district_assignments_post)

    print("Calculating approval scores...")
    # Calculate approval scores for all blocks
    for block in grid.blocks:
        approval_scores_pre[block.block_id] = calculate_approval_score(grid, block.block_id, district_assignments_pre)
        approval_scores_post[block.block_id] = calculate_approval_score(grid, block.block_id, district_assignments_post)

    print("Converting to DataFrame...")
    blocks_df = to_dataframe(grid, district_assignments_pre, district_assignments_post,
                            approval_scores_pre, approval_scores_post)

    print("Creating visualizations...")
    create_figures(grid, blocks_df, NUM_DISTRICTS)

    print("Generating summary statistics...")
    print_summary_statistics(grid, NUM_DISTRICTS, district_assignments_pre, district_assignments_post)

    print("✓ All figures saved to 'figures/' directory")
    print("✓ Simulation complete!")
