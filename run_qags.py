
import time
import numpy as np

from qags.quantum_search import QuantumAdaptiveSearcher
from qags.classical_search import classical_grid_search


# =========================
# USER INPUT SECTION
# =========================

def objective_function(x):
    # Example: Styblinski-Tang
    return 0.5 * sum(xi**4 - 16*xi**2 + 5*xi for xi in x)

dimension = 4
bounds = [(-5, 5)] * dimension
n_qubits = 2
max_iter = 5
grid_points = 20

# =========================
# RUN QAGS
# =========================
qags = QuantumAdaptiveSearcher(objective_function, bounds, n_qubits=n_qubits)

start_q = time.time()
q_result = qags.run(max_iter=max_iter)
quantum_time = time.time() - start_q

# =========================
# RUN CLASSICAL
# =========================
start_c = time.time()
c_result = classical_grid_search(
    objective_function,
    bounds,
    grid_points=grid_points,
    max_iter=max_iter
)
classical_time = time.time() - start_c

# =========================
# RESULTS
# =========================
print("\n===== QAGS RESULT =====")
print("Best point:", q_result["classical_solution"])
print("Best value:", q_result["classical_value"])
print("Runtime:", quantum_time)

if c_result is not None:
    print("\n===== CLASSICAL RESULT =====")
    print("Best point:", c_result["best_point"])
    print("Best value:", c_result["best_value"])
    print("Runtime:", classical_time)
else:
    print("\n===== CLASSICAL RESULT =====")
    print("Classical search unavailable due to computational complexity.")

# =========================
# PLOTS
# =========================

from qags.visualization import plot_qags_search_2d

# Only for 2D problems
if dimension == 2:
    plot_qags_search_2d(
        objective_function,
        qags.history,
        bounds,
        q_result["classical_solution"],
        "outputs/qags_search_visualization.png"
    )

print("2D search visualization saved in outputs/")