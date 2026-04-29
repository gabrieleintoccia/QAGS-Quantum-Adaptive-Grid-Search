
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_qags_search_2d(f, history, initial_bounds, best_point, save_path, resolution=300):
    """
    Plot:
    - 2D contour of objective function
    - Initial domain
    - Shrinking QAGS domains
    - Final minimum found
    """

    if len(initial_bounds) != 2:
        print("2D visualization only works for 2-dimensional problems.")
        return

    # Grid for contour plot
    x = np.linspace(initial_bounds[0][0], initial_bounds[0][1], resolution)
    y = np.linspace(initial_bounds[1][0], initial_bounds[1][1], resolution)

    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f([X[i, j], Y[i, j]])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Function contour
    contour = ax.contourf(X, Y, Z, levels=60, cmap="viridis")
    plt.colorbar(contour, ax=ax, label="Function Value")

    # Initial domain
    init_rect = Rectangle(
        (initial_bounds[0][0], initial_bounds[1][0]),
        initial_bounds[0][1] - initial_bounds[0][0],
        initial_bounds[1][1] - initial_bounds[1][0],
        linewidth=2,
        edgecolor="white",
        linestyle="--",
        facecolor="none",
        label="Initial Domain"
    )
    ax.add_patch(init_rect)

    # Shrinking domains
    colors = plt.cm.autumn(np.linspace(0, 1, len(history)))

    for i, step in enumerate(history):
        b = step["bounds"]

        rect = Rectangle(
            (b[0][0], b[1][0]),
            b[0][1] - b[0][0],
            b[1][1] - b[1][0],
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            alpha=0.85
        )
        ax.add_patch(rect)

    # Best point found
    ax.plot(
        best_point[0],
        best_point[1],
        "ro",
        markersize=10,
        label="QAGS Minimum"
    )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("QAGS Adaptive Domain Shrinkage")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()