from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
import numpy as np
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import QuantumCircuit
from pymongo import MongoClient
from datetime import datetime
import os
from io import BytesIO
import base64
from dotenv import load_dotenv
import warnings
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for non-interactive plotting
import matplotlib.pyplot as plt
from joblib import load
import logging

# RDKit import with fallback for systems where App Control policy blocks native C++ DLLs
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDetermineBonds
    RDKIT_AVAILABLE = True
except Exception as e:
    logging.warning(f"RDKit native DLL loading skipped or unavailable: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# Initialize MongoDB services if configured
MONGO_URI = os.getenv("MONGODB_ATLAS_URI")
client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client["quantum-sim"] if client else None
results_collection = db["results"] if db else None

# Load ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "quantum_classical_predictor.joblib")
try:
    model = load(MODEL_PATH)
    logger.info("Successfully loaded trained model")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Optional IBM Quantum Service
IBM_API_TOKEN = os.getenv("IBMQ_API_TOKEN")
ibm_service = None
if IBM_API_TOKEN:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        ibm_service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_API_TOKEN)
        logger.info("Connected to IBM Quantum service")
    except Exception as e:
        logger.error(f"IBM Quantum connection error: {str(e)}")

app = FastAPI(title="VQE Molecular Simulator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class AtomCoord(BaseModel):
    element: str
    x: float = Field(0.0, ge=-100, le=100)
    y: float = Field(0.0, ge=-100, le=100)
    z: float = Field(0.0, ge=-100, le=100)

class MoleculeInput(BaseModel):
    atoms: List[AtomCoord]
    charge: int = Field(0, ge=-5, le=5)
    spin: int = Field(0, ge=0, le=10)
    use_quantum_hardware: bool = False
    basis_set: str = Field("sto3g", pattern=r"^[a-zA-Z0-9]+$")

class SimulationSuccess(BaseModel):
    molecule_name: str
    exact_energy: float
    vqe_energy: float
    ansatz_type: str
    status: str = "success"
    source: Optional[str] = None
    cached_at: Optional[str] = None
    molecule_image: Optional[str] = None
    energy_plot: Optional[str] = None
    backend: Optional[str] = None
    qubit_count: Optional[int] = None
    elements: Optional[List[str]] = None

class SimulationError(BaseModel):
    status: str = "failed"
    error: str
    suggestion: Optional[str] = None

class PredictionInput(BaseModel):
    num_atoms: int = Field(..., gt=0, description="Number of atoms in molecule")
    num_electrons: int = Field(..., gt=0, description="Number of electrons")
    num_qubits: int = Field(..., gt=0, description="Number of qubits required")
    basis_set_size: int = Field(..., gt=0, description="Size of basis set")
    molecular_complexity: float = Field(..., gt=0, le=10, description="Complexity score (1-10)")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    features: Dict[str, float]
    status: str = "success"

# Quantum Molecular Operator Builder
ATOMIC_NUMBERS = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18
}

ELEMENT_COLORS = {
    'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050',
    'CL': '#1FF01F', 'BR': '#A62929', 'I': '#940094', 'HE': '#D9FFFF', 'LI': '#CC80FF',
    'BE': '#C2FF00', 'B': '#FFB5B5', 'S': '#FFFF30', 'P': '#FF8000'
}

def build_molecular_qubit_operator(atoms: List[AtomCoord], charge: int = 0):
    num_atoms = len(atoms)
    total_z = sum(ATOMIC_NUMBERS.get(a.element.upper(), 1) for a in atoms) - charge
    num_qubits = max(2, min(8, int(np.ceil(total_z / 2) * 2)))

    v_nn = 0.0
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            r = np.sqrt(
                (atoms[i].x - atoms[j].x)**2 +
                (atoms[i].y - atoms[j].y)**2 +
                (atoms[i].z - atoms[j].z)**2
            )
            if r > 1e-4:
                z_i = ATOMIC_NUMBERS.get(atoms[i].element.upper(), 1)
                z_j = ATOMIC_NUMBERS.get(atoms[j].element.upper(), 1)
                v_nn += (z_i * z_j) / r

    r_eff = max(0.5, v_nn if v_nn > 0 else 1.0)
    g0 = -1.05 - (total_z * 0.2) + (0.5 / r_eff) + v_nn
    g1 = 0.39 / (r_eff**0.8)
    g2 = -0.39 / (r_eff**0.8)
    g3 = 0.01 / (r_eff**2)
    g4 = 0.18 * np.exp(-0.8 * r_eff)
    g5 = 0.18 * np.exp(-0.8 * r_eff)

    if num_qubits == 2:
        pauli_list = [
            ("II", g0),
            ("IZ", g1),
            ("ZI", g2),
            ("ZZ", g3),
            ("XX", g4),
            ("YY", g5)
        ]
    else:
        pauli_list = [
            ("I" * num_qubits, g0),
            ("Z" + "I" * (num_qubits - 1), g1),
            ("I" + "Z" + "I" * (num_qubits - 2), g2),
            ("ZZ" + "I" * (num_qubits - 2), g3),
            ("XX" + "I" * (num_qubits - 2), g4),
            ("YY" + "I" * (num_qubits - 2), g5)
        ]

    qubit_op = SparsePauliOp.from_list(pauli_list)
    return qubit_op, num_qubits

def get_cache_key(data: MoleculeInput) -> str:
    atoms_str = ";".join(
        f"{atom.element}{atom.x:.4f}{atom.y:.4f}_{atom.z:.4f}"
        for atom in sorted(data.atoms, key=lambda x: x.element)
    )
    return f"{atoms_str}|{data.charge}|{data.spin}|{data.basis_set}|{data.use_quantum_hardware}"

def create_energy_plot(distances, exact_energies, vqe_energies) -> str:
    plt.figure(figsize=(9, 5))
    plt.plot(distances, exact_energies, 'b-o', linewidth=2, label='Exact Energy (FCI/Full Eigensolver)')
    plt.plot(distances, vqe_energies, 'r--s', linewidth=2, label='VQE Energy (Variational Solver)')
    plt.xlabel('Bond Distance (Å)', fontsize=12)
    plt.ylabel('Ground State Energy (Hartree)', fontsize=12)
    plt.title('Molecular Potential Energy Dissociation Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def simulate_energy_curve(atoms: List[AtomCoord], charge: int, spin: int, basis_set: str, use_quantum_hardware: bool):
    distances = np.linspace(0.5, 2.5, 10)
    exact_energies = []
    vqe_energies = []
    
    for dist in distances:
        temp_atoms = [AtomCoord(element=a.element, x=a.x, y=a.y, z=a.z) for a in atoms]
        if len(temp_atoms) >= 2:
            temp_atoms[1].x = float(dist)
            
        qubit_op, _ = build_molecular_qubit_operator(temp_atoms, charge)
        
        exact_res = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
        exact_val = float(exact_res.eigenvalue.real)
        exact_energies.append(exact_val)
        
        ansatz = EfficientSU2(qubit_op.num_qubits, reps=1, entanglement="linear")
        optimizer = COBYLA(maxiter=40)
        estimator = Estimator()
        vqe = VQE(estimator, ansatz, optimizer)
        vqe_res = vqe.compute_minimum_eigenvalue(qubit_op)
        vqe_val = float(vqe_res.eigenvalue.real)
        vqe_energies.append(vqe_val)
    
    return distances.tolist(), exact_energies, vqe_energies

def generate_molecule_image(atoms: List[AtomCoord], charge: int) -> str:
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.RWMol()
            for atom in atoms:
                atom_obj = Chem.Atom(atom.element)
                if atom_obj.GetAtomicNum() == 0:
                    atom_obj = Chem.Atom("C")
                mol.AddAtom(atom_obj)
                
            conf = Chem.Conformer(len(atoms))
            for i, atom in enumerate(atoms):
                conf.SetAtomPosition(i, (float(atom.x), float(atom.y), float(atom.z)))
            mol.AddConformer(conf)
            
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=charge)
            except Exception:
                if len(atoms) > 1:
                    for i in range(len(atoms) - 1):
                        if not mol.GetBondBetweenAtoms(i, i + 1):
                            mol.AddBond(i, i + 1, Chem.BondType.SINGLE)
            
            img = Draw.MolToImage(mol, size=(400, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.warning(f"RDKit image generation fallback: {str(e)}")

    # Matplotlib 2D Atomic Structure Visualization Fallback
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_facecolor('#1e1e2e')
    fig.patch.set_facecolor('#1e1e2e')
    
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    
    # Draw bonds
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.sqrt((atoms[i].x - atoms[j].x)**2 + (atoms[i].y - atoms[j].y)**2 + (atoms[i].z - atoms[j].z)**2)
            if dist < 2.5:
                ax.plot([atoms[i].x, atoms[j].x], [atoms[i].y, atoms[j].y], color='#89b4fa', linewidth=2.5, zorder=1)

    # Draw atoms
    for a in atoms:
        elem = a.element.upper()
        color = ELEMENT_COLORS.get(elem, '#cba6f7')
        ax.scatter(a.x, a.y, s=600, color=color, edgecolors='#ffffff', linewidth=1.5, zorder=2)
        ax.text(a.x, a.y, elem, color='#11111b' if color in ['#FFFFFF', '#D9FFFF', '#FFFF30'] else '#ffffff',
                fontsize=11, fontweight='bold', ha='center', va='center', zorder=3)

    ax.axis('off')
    ax.autoscale()
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# API Routes
@app.get("/")
async def root():
    return {"message": "VQE Molecular Simulator API is running"}

@app.post("/simulate/", response_model=Union[SimulationSuccess, SimulationError])
async def simulate_molecule(data: MoleculeInput):
    cache_key = get_cache_key(data)
    
    try:
        if client and results_collection:
            cached = results_collection.find_one({"cache_key": cache_key})
            if cached and "result" in cached:
                cached_result = cached["result"]
                if cached_result.get("status") == "success":
                    cached_result["source"] = "cache"
                    cached_result["cached_at"] = cached["timestamp"].isoformat()
                    return SimulationSuccess(**cached_result)
                return SimulationError(**cached_result)
        
        elements = {a.element for a in data.atoms}
        molecule_name = "".join(f"{e}{sum(1 for a in data.atoms if a.element == e)}" for e in sorted(elements))
        
        qubit_op, num_qubits = build_molecular_qubit_operator(data.atoms, data.charge)
        
        exact_res = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
        exact_energy = float(exact_res.eigenvalue.real)
        
        ansatz = EfficientSU2(num_qubits, reps=1, entanglement="linear")
        optimizer = COBYLA(maxiter=80)
        estimator = Estimator()
        vqe = VQE(estimator, ansatz, optimizer)
        vqe_res = vqe.compute_minimum_eigenvalue(qubit_op)
        vqe_energy = float(vqe_res.eigenvalue.real)
        
        backend_name = "Local Quantum Simulator (Statevector)"
        
        distances, exact_energies, vqe_energies = simulate_energy_curve(
            data.atoms, data.charge, data.spin, data.basis_set, data.use_quantum_hardware
        )
        energy_plot = create_energy_plot(distances, exact_energies, vqe_energies)
        
        molecule_image = generate_molecule_image(data.atoms, data.charge)
        
        result_data = SimulationSuccess(
            molecule_name=molecule_name,
            exact_energy=exact_energy,
            vqe_energy=vqe_energy,
            ansatz_type="EfficientSU2",
            backend=backend_name,
            qubit_count=num_qubits,
            elements=list(elements),
            molecule_image=molecule_image,
            energy_plot=energy_plot,
            source="computation"
        )
        
        if client and results_collection:
            results_collection.update_one(
                {"cache_key": cache_key},
                {"$set": {"result": result_data.dict(), "timestamp": datetime.utcnow()}},
                upsert=True
            )
        
        return result_data
    
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Simulation error: {error_message}")
        
        return SimulationError(
            error=error_message,
            suggestion="Try adjusting molecular geometry or atom coordinates"
        )

@app.post("/predict/", response_model=Union[PredictionOutput, SimulationError])
async def predict_behavior(data: PredictionInput):
    if not model:
        return SimulationError(
            error="Prediction model not available",
            suggestion="Please run train_model.py to build the predictor model"
        )
    
    try:
        features = [
            data.num_atoms,
            data.num_electrons,
            data.num_qubits,
            data.basis_set_size,
            data.molecular_complexity
        ]
        
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        confidence = round(float(max(probabilities)), 4)
        
        feature_dict = {
            "num_atoms": float(data.num_atoms),
            "num_electrons": float(data.num_electrons),
            "num_qubits": float(data.num_qubits),
            "basis_set_size": float(data.basis_set_size),
            "molecular_complexity": float(data.molecular_complexity)
        }
        
        return PredictionOutput(
            prediction="Quantum Advantage" if prediction == 1 else "Classical Efficient",
            confidence=confidence,
            features=feature_dict
        )
        
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        logger.error(error_message)
        return SimulationError(
            error=error_message,
            suggestion="Check your input values and try again"
        )

@app.get("/model/info")
async def get_model_info():
    if not model:
        return {"status": "error", "message": "Model not loaded"}
    
    try:
        model_info = {
            "model_type": str(type(model).__name__),
            "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown",
            "classes": list(map(str, model.classes_)) if hasattr(model, 'classes_') else "Unknown",
            "status": "loaded"
        }
        return model_info
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/cache/stats")
async def get_cache_stats():
    if results_collection:
        return {"entries": results_collection.count_documents({})}
    return {"entries": 0}

@app.delete("/cache/clear")
async def clear_cache():
    if results_collection:
        results_collection.delete_many({})
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
