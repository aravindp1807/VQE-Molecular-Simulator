import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

def generate_dataset(n_samples=2000):
    np.random.seed(42)
    
    num_atoms = np.random.randint(1, 30, size=n_samples)
    num_electrons = np.random.randint(2, 60, size=n_samples)
    num_qubits = np.random.randint(2, 40, size=n_samples)
    basis_set_size = np.random.randint(5, 100, size=n_samples)
    molecular_complexity = np.random.uniform(1.0, 10.0, size=n_samples)
    
    # Heuristic for Quantum vs Classical advantage:
    # High qubit count (>14), high electron count (>16), or high complexity (>5.5) -> Quantum (1)
    score = (
        (num_qubits / 20.0) * 0.35 +
        (molecular_complexity / 10.0) * 0.35 +
        (num_electrons / 40.0) * 0.2 +
        (num_atoms / 20.0) * 0.1
    )
    
    # Target: 1 for Quantum, 0 for Classical
    y = (score > 0.45).astype(int)
    
    X = np.column_stack([
        num_atoms,
        num_electrons,
        num_qubits,
        basis_set_size,
        molecular_complexity
    ])
    
    return X, y

def main():
    X, y = generate_dataset()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    output_path = os.path.join(os.path.dirname(__file__), "quantum_classical_predictor.joblib")
    dump(clf, output_path)
    print(f"Model successfully trained and saved to {output_path}")

if __name__ == "__main__":
    main()
