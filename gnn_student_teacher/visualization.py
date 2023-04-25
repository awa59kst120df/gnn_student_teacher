import os
import typing as t
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_node_importances(g: dict,
                          node_importances: np.ndarray,
                          node_coordinates: np.ndarray,
                          ax: plt.Axes,
                          radius: float = 30,
                          thickness: float = 4,
                          color='black',
                          vmin: float = 0,
                          vmax: float = 1):
    node_normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i, (x, y), ni in zip(g['node_indices'], node_coordinates, node_importances):
        circle = plt.Circle(
            (x, y),
            radius=radius,
            lw=thickness,
            color=color,
            fill=False,
            alpha=node_normalize(ni)
        )
        ax.add_artist(circle)


def plot_edge_importances(g: dict,
                          edge_importances: np.ndarray,
                          node_coordinates: np.ndarray,
                          ax: plt.Axes,
                          radius: float = 30,
                          thickness: float = 4,
                          color='black',
                          vmin: float = 0,
                          vmax: float = 1):
    edge_normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for (i, j), ei in zip(g['edge_indices'], edge_importances):
        coords_i = node_coordinates[i]
        coords_j = node_coordinates[j]
        # Here we determine the actual start and end points of the line to draw. Now we cannot simply use
        # the node coordinates, because that would look pretty bad. The lines have to start at the very
        # edge of the node importance circle which we have already drawn (presumably) at this point. This
        # circle is identified by it's radius. So what we do here is we reduce the length of the line on
        # either side of it by exactly this radius. We do this by first calculating the unit vector for that
        # line segment and then moving radius times this vector into the corresponding directions
        diff = (coords_j - coords_i)
        delta = (radius / np.linalg.norm(diff)) * diff
        x_i, y_i = coords_i + delta
        x_j, y_j = coords_j - delta

        ax.plot(
            [x_i, x_j],
            [y_i, y_j],
            color=color,
            lw=thickness,
            alpha=edge_normalize(ei)
        )


def plot_multi_variant_box_plot(ax: plt.Axes,
                                data_list: t.List[t.Dict[str, t.List[float]]],
                                variant_colors: t.Dict[str, str] = defaultdict(lambda: 'black'),
                                variant_color_alphas: t.Dict[str, str] = defaultdict(lambda: 0.4),
                                do_white_background: bool = True,
                                plot_between: bool = False,
                                between_ls: str = '--',
                                inter_spacing: float = 0.5,
                                intra_spacing: float = 0.1,
                                **kwargs):
    """
    Draws multiple comparative box plots into a single plot based on multiple lists of values.

    :returns: A list of float values, which are the x-positions of the centers of each individual box plot
        group.
    """
    variants = list(data_list[0].keys())
    variant_medians: t.List[t.Dict[str, float]] = []

    centers = []
    for center, data in enumerate(data_list):
        variants = list(data.keys())
        num_variants = len(variants)

        centers.append(center)
        total_width = 1 - inter_spacing
        width = (total_width - (num_variants - 1) * intra_spacing) / num_variants
        positions = [(center - (total_width / 2) + (width / 2)) + i * (width + intra_spacing)
                     for i in range(num_variants)]
        for position, (variant, values) in zip(positions, data.items()):
            color = variant_colors[variant]
            alpha = variant_color_alphas[variant]
            fill_color = mcolors.to_rgba(color, alpha)

            results = ax.boxplot(
                values,
                positions=[position],
                widths=[width],
                manage_ticks=False,
                patch_artist=True,
                medianprops={
                    'color': color
                },
                boxprops={
                    'facecolor': fill_color
                },
                **kwargs
            )

            # Adding a white background for all the boxes. We achieve a lighter color of the box content
            # by using a certain alpha value. This creates the problem however that the boxes are no longer
            # solid and cannot hide other content behind them. But that is the behavior we want most of the
            # time
            if do_white_background:
                for path_patch in results['boxes']:
                    p = mpl.patches.PathPatch(path_patch.get_path(), facecolor='white', zorder=-1)
                    ax.add_patch(p)

        # For the current position we construct a dictionary which assigns the median value of each
        # variant's value list to the name of that variant. We will later need this to (optionally) plot
        # the in-between lines.
        variant_medians.append({variant: np.median(values) for variant, values in data.items()})

    ax.set_xticks(centers)

    # If this optional flag is set, we plot normal lines between each of the entries of the data list
    # (aka all the center x-positions in the plot), which go between the median values of the distributions
    # We do that for each variant and for each variant we use it's own color
    if plot_between and len(variants) > 1:
        for variant in variants:
            ax.plot(
                centers,
                [data[variant] for data in variant_medians],
                color=variant_colors[variant],
                alpha=variant_color_alphas[variant],
                ls=between_ls,
                zorder=-10
            )

    # We return the list which contains the center x-coordinates of all the box groups, because in a
    # post-processing step we will most likely want to change the tick labels to reflect the sweep
    # configuration
    return centers
