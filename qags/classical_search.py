
import numpy as np


def classical_grid_search(f, bounds, grid_points=10, max_iter=8, max_points_threshold=1e7):
    """
    Classical adaptive grid search with safety limits.

    Stops automatically if total grid points exceed threshold.
    """

    current_bounds = np.array(bounds, dtype=float)
    history = []

    dim = len(bounds)

    for i in range(max_iter):

        total_points = grid_points ** dim

        # Safety check
        if total_points > max_points_threshold:
            print(
                f"[WARNING] Classical search aborted at iteration {i}: "
                f"{total_points:.2e} grid points exceed safe threshold."
            )
            return None

        try:
            # Create grid
            grid = [np.linspace(b[0], b[1], grid_points) for b in current_bounds]
            mesh = np.meshgrid(*grid, indexing='ij')

            points = np.stack(mesh, axis=-1).reshape(-1, dim)

            # Evaluate function
            values = np.array([f(p) for p in points])

            idx = np.argmin(values)
            best_point = points[idx]
            best_value = values[idx]

            # Shrink domain
            new_bounds = []

            for d in range(dim):
                width = current_bounds[d][1] - current_bounds[d][0]

                new_min = max(bounds[d][0], best_point[d] - width / 4)
                new_max = min(bounds[d][1], best_point[d] + width / 4)

                new_bounds.append((new_min, new_max))

            history.append({
                "iteration": i,
                "bounds": current_bounds.copy(),
                "best_point": best_point,
                "best_value": best_value
            })

            current_bounds = np.array(new_bounds)

        except MemoryError:
            print(f"[WARNING] Classical search stopped due to memory overflow.")
            return None

        except Exception as e:
            print(f"[WARNING] Classical search failed: {e}")
            return None

    if len(history) == 0:
        return None

    return min(history, key=lambda x: x["best_value"])