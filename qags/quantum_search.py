
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

class QuantumAdaptiveSearcher:
    def __init__(self, f, bounds, n_qubits=4):
        self.f = f
        self.bounds = np.array(bounds, dtype=float)
        self.n_qubits = n_qubits
        self.dim = len(bounds)
        self.current_bounds = self.bounds.copy()
        self.history = []

    def quantum_probability_estimation(self):
        grid = [np.linspace(b[0], b[1], 2**self.n_qubits) for b in self.current_bounds]
        mesh = np.meshgrid(*grid, indexing='ij')
        points = np.stack(mesh, axis=-1).reshape(-1, self.dim)

        values = np.array([self.f(p) for p in points])

        min_val = np.min(values)
        std_val = np.std(values) if np.std(values) > 0 else 1.0

        amplitudes = np.exp(-(values - min_val) / std_val)
        amplitudes /= np.linalg.norm(amplitudes)

        state = Statevector(amplitudes)
        probs = state.probabilities()

        return probs.reshape([2**self.n_qubits] * self.dim)

    def adjust_bounds(self, mask):
        grid = [np.linspace(b[0], b[1], 2**self.n_qubits) for b in self.current_bounds]
        mesh = np.meshgrid(*grid, indexing='ij')

        new_bounds = []
        for d in range(self.dim):
            active_points = mesh[d][mask]

            if active_points.size == 0:
                new_bounds.append(self.current_bounds[d])
                continue

            new_min = max(self.bounds[d][0], np.min(active_points))
            new_max = min(self.bounds[d][1], np.max(active_points))
            new_bounds.append((new_min, new_max))

        return np.array(new_bounds)

    def run(self, max_iter=8):
        for i in range(max_iter):
            prob_map = self.quantum_probability_estimation()

            threshold = np.percentile(prob_map, 75)
            mask = prob_map >= threshold

            new_bounds = self.adjust_bounds(mask)

            x0 = np.mean(new_bounds, axis=1)
            result = minimize(self.f, x0, bounds=new_bounds)

            self.history.append({
                "iteration": i,
                "bounds": self.current_bounds.copy(),
                "classical_solution": result.x,
                "classical_value": result.fun
            })

            self.current_bounds = new_bounds

        best = min(self.history, key=lambda x: x["classical_value"])
        return best
