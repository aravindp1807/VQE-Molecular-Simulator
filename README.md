# ⚛️ VQE Molecular Simulator & AI Behavior Predictor (Google Colab Edition)

A 100% self-contained Google Colab solution for simulating molecular energy curves using quantum algorithms (VQE), predicting classical vs. quantum behavior using Machine Learning, rendering molecular 3D geometries and potential energy curves, and exploring curated element/molecule database insights.

---

## 🌟 Key Features

1. **Simulate Molecular Energy Curves using VQE (Variational Quantum Eigensolver)**
   - Computes Exact Ground State Energies (FCI / NumPyMinimumEigensolver) vs. VQE Energies across bond distances ($0.4\text{ Å}$ to $2.2\text{ Å}$).
   - Integrates with Qiskit Nature, PySCF, and ParityMapper.
   - Pre-configured with IBM Quantum API Key for optional cloud execution.

2. **Predict Molecular Behavior using Machine Learning**
   - Trained `RandomForestClassifier` predicting whether a molecule requires Quantum (VQE) algorithms or if Classical solvers suffice.
   - Features: `num_atoms`, `num_electrons`, `num_qubits`, `basis_set_size`, `molecular_complexity`.
   - Returns recommendation, confidence percentage, and feature importance breakdown.

3. **Visualize Molecular Structures & Energy Plots**
   - 3D spatial molecular scatter plots with covalent bond rendering.
   - Potential Energy Surface (PES) dissociation curve plotting (Exact vs. VQE).

4. **Curated Elements & Molecules Database with Insights**
   - In-depth chemical insights for Hydrogen, Lithium, Beryllium, Carbon, Oxygen, $H_2$, $LiH$, $BeH_2$, $H_2O$, $CH_4$.
   - Details atomic structure, electron configurations, molecular geometry, bonding, qubits required, complexity score, and real-world applications.

5. **Interactive Google Colab Dashboard (`ipywidgets`)**
   - Graphical 3-tab widget dashboard running inside Colab notebook cells.

---

## 🚀 How to Run in Google Colab

### Option A: Open `.ipynb` Notebook Directly in Colab
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `VQE_Molecular_Simulator_Colab.ipynb`.
3. Select **Runtime > Run all** (or press `Ctrl + F9`).

### Option B: Run via Python Script
1. Upload `vqe_colab_simulator.py` to your Colab session files.
2. In a Colab cell, run:
```python
!pip install -q qiskit qiskit-algorithms qiskit-nature pyscf rdkit matplotlib scikit-learn joblib ipywidgets qiskit-ibm-runtime

from vqe_colab_simulator import ColabDashboard

dashboard = ColabDashboard(ibm_token="7hk4AmU8jgQqqDjkkY4fpYD-3h6VtCjSPJqfSU0xmf9f")
dashboard.display()
```

---

## 🔐 IBM Quantum API Token
The IBM Quantum API Key `7hk4AmU8jgQqqDjkkY4fpYD-3h6VtCjSPJqfSU0xmf9f` is pre-loaded into the simulator.
To execute jobs on real IBM Quantum Cloud hardware:
1. Check the box **"Run on IBM Quantum Cloud Hardware"** in Tab 2 of the Colab GUI dashboard.
2. The simulator connects to IBM Quantum Runtime (`ibmq_qasm_simulator` / least-busy quantum backend).

---

## 📁 Repository Files

- **`VQE_Molecular_Simulator_Colab.ipynb`**: Complete Google Colab Notebook with text cells, code snippets, and interactive GUI.
- **`vqe_colab_simulator.py`**: Standalone modular Python library containing `CuratedDatabase`, `MoleculeVisualizer`, `VQESimulator`, `QuantumBehaviorPredictor`, and `ColabDashboard`.
- **`README.md`**: User guide & technical reference.
