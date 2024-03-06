import matplotlib.pyplot as plt
import numpy as np
from utils.util_calc import calc_tour_length

def add_arrow(tour, coords, step, color, ax):
    if len(tour) > 1:
        x = coords[:, 0]
        y = coords[:, 1]
        x0 = x[tour[step]]; y0 = y[tour[step]]
        x1 = x[tour[step+1]]; y1 = y[tour[step+1]] 
        ax.annotate('', xy=[x1, y1], xytext=[x0, y0],
                    arrowprops=dict(shrink=0, width=1, headwidth=8,
                                    headlength=10, connectionstyle="arc3",
                                    facecolor=color, edgecolor=color))

def visualize_tsp_tour(coords, tour, ax, linestyle="--"):
    """
    Parameters
    ----------
    instance: 2d list [num_nodes x coordinates]
    tour: 1d list [seq_length]
    """
    points = np.array(coords)
    tour = np.array(tour)
    # tour = tour - 1 # offset to make the first index 0
    x = points[:, 0]
    y = points[:, 1]

    # visualize points
    ax.scatter(x, y, c="black", zorder=2)

    # visualize pathes
    ax.plot(x[tour], y[tour], linestyle, c='black', zorder=1)

    # add an arrow indicating initial direction
    add_arrow(tour, points, 0, "black", ax)

def visualize_factual_and_cf_tours(factual_tour, cf_tour, coords, cf_step, vis_filename):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    visualize_tsp_tour(coords, factual_tour, ax1)
    visualize_tsp_tour(coords, cf_tour, ax2)
    visualize_tsp_tour(coords, factual_tour[:cf_step], ax1, linestyle="-")
    visualize_tsp_tour(coords, cf_tour[:cf_step], ax2, linestyle="-")
    add_arrow(factual_tour, coords, cf_step-1, "red", ax1) # factual visit
    add_arrow(cf_tour, coords, cf_step-1, "blue", ax2) # counterfactual visit
    factual_tour_length = calc_tour_length(factual_tour, coords)
    cf_tour_length = calc_tour_length(cf_tour, coords)
    ax1.set_title(f"Factual tour\nTour length={factual_tour_length:.3f}")
    ax2.set_title(f"Counterfactual tour\nTour length={cf_tour_length:.3f}")
    plt.savefig(vis_filename)