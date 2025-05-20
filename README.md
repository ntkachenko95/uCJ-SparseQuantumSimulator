# uCJ-SparseQuantumSimulator

A Python-based quantum simulator based on SciPy's `dok_matrix`. This project includes a fully custom simulator backend, gate application, and implementations of real, imaginary and generalized unitary Cluster Jastrow (uCJ) ansatze.

## Project Structure

```
.
├── simulator.py               # Main simulator + auxilary functions
├── examples                   # Examples: H_2 molecule with im- re- and g-uCJ ansatze
    ├── h2_im_uCJ.py
    ├── h2_re_uCJ.py
    ├── h2_g_uCJ.py          
├── README.md
```

## Dependencies

```bash
pip install numpy scipy matplotlib openfermion openfermionpyscf pyscf
```

## Getting Started

### Run the H₂ simulation:

```bash
python h2_im_uCJ.py
```

This will:
- Run VQE for H_2 in STO-3G using the im-uCJ ansatz
- Plot the energy error vs FCI across bond lengths
- Save the result to `H2_sto3G_im_uCJ_RHF.jpg`

## Simulator Example Use

```python
import simulator as s

sim = s.SparseQuantumSimulator(num_qubits=4)
sim.apply_single_qubit_gate(0, s.Paulis['X'])  # Pauli-X
state = sim.get_statevector()
```
## Author

Developed by Nikolay Tkachenko, University of California Berkeley. Citation: arXiv:2505.10963.

