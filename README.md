# QAGS — Quantum Adaptive Grid Search

QAGS (Quantum Adaptive Grid Search) is a quantum-inspired optimization framework for multivariate continuous function minimization based on adaptive domain contraction through amplitude-encoded probabilistic search.

This repository implements the method described in:

**Frontiers in Applied Mathematics and Statistics (2025)**  
Quantum Adaptive Grid Search (QAGS)

## Features

- Minimize arbitrary multivariate functions
- User-configurable:
  - Function
  - Dimension
  - Bounds
  - Number of qubits per dimension
  - Maximum iterations
- Runtime benchmark:
  - Classical adaptive grid search
  - QAGS
- Visualization:
  - Runtime comparison
  - Domain shrinkage toward the minimum (2D visualization)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python run_qags.py
```

You can edit:

- target function
- search bounds
- dimension
- qubits
- iterations

inside `run_qags.py`.

---

## Outputs

Generated plots:

- `runtime_comparison.png`
- `domain_shrinkage.png`

---

## Repository Structure

```bash
QAGS-Repository/
│
├── run_qags.py
├── requirements.txt
├── README.md
│
├── qags/
│   ├── __init__.py
│   ├── quantum_search.py
│   ├── classical_search.py
│   └── visualization.py
│
└── outputs/
```

---

## Citation

If you use this method, cite the original QAGS publication.
