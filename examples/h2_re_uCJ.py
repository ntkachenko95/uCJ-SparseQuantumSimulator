import os
import sys
libdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(libdir)
import simulator as s
import numpy as np

# =================================
# Auxilary functions to define VQE
# =================================

from scipy.optimize import minimize

hamiltonian_dict = None
k_indices = None
j_indices = None
def energy(param, hamiltonian_dict, size=4, initial_state="0011", k_indices=k_indices, j_indices=j_indices, K_mat_implementation='paper', real=True, imag=False, thresh=1e-16, debug=False):
    if imag:
        K_mat=np.zeros((size,size), dtype=np.complex128)
    else:
        K_mat=np.zeros((size,size))
    if real and imag and len(param)<(len(k_indices)*2+len(j_indices)):
        print("ERROR: smaller number of parameters than needed")
        return None
    elif len(param)<(len(k_indices)+len(j_indices)):
        print("ERROR: smaller number of parameters than needed")
        return None
    if real:
        for d, (ind1,ind2) in enumerate(k_indices):
            K_mat[ind1][ind2] -= param[d]
            K_mat[ind2][ind1] += param[d]
    if real and imag:
        for d,(ind1,ind2) in enumerate(k_indices):
            K_mat[ind1][ind2] += 1j*param[d+len(k_indices)]
            K_mat[ind2][ind1] += 1j*param[d+len(k_indices)]
    if (not real) and imag:
        for d,(ind1,ind2) in enumerate(k_indices):
            K_mat[ind1][ind2] += 1j*param[d]
            K_mat[ind2][ind1] += 1j*param[d] 
    J_mat=np.zeros((size,size))
    for d, (ind1,ind2) in enumerate(j_indices):
        J_mat[ind1][ind2] += param[-1-d]
        J_mat[ind2][ind1] += param[-1-d]  
    simulator = s.SparseQuantumSimulator(num_qubits=size, initial_state={int(initial_state, 2):1.0})
    s.create_ucj_circuit(simulator, K_mat, J_mat, K_mat_implementation=K_mat_implementation, thresh=thresh, debug=debug)
    return np.real(sum(coef * simulator.expectation_value(pauli_word) for pauli_word, coef in hamiltonian_dict.items()))

if __name__ == '__main__':
    from pyscf import gto, scf, ao2mo
    from openfermionpyscf import run_pyscf
    from openfermion.transforms import jordan_wigner
    from openfermion.ops import FermionOperator
    from openfermion.chem import MolecularData
    import matplotlib.pyplot as plt
    
    results_re = []
    overlaps_re = {"T_1":[],
    'T_2' : [],
    'T_3' : [],
    'S_1' : [],
    'S_2' : [],
    'S_3' : [],}
    de_re = []
    simulators_re = []
    size=4
    CSFs = {"T_1":{'1001':1/np.sqrt(2),
                   '0110':1/np.sqrt(2)},
    'T_2' : {'1010':1},
    'T_3' : {'0101':1},
    'S_1' : {'1001':1/np.sqrt(2),
        '0110':-1/np.sqrt(2)},
    'S_2' : {'0011':1},
    'S_3' : {'1100':1}}

    for bond_length in np.linspace(0.5,3.0,26):

        # Define molecule
        basis = 'STO-3G'
        geometry = [["H", [0, 0, 0]], ["H", [0, 0, bond_length]]]
        charge = 0
        multiplicity = 1  # Singlet state

        # Create PySCF molecule
        mol = gto.Mole()
        mol.build(
            atom=geometry,
            basis=basis,
            charge=charge,
            spin=(multiplicity - 1)//2
        )

        # Perform Hartree-Fock calculation
        mf = scf.RHF(mol)
        mf.kernel()

        # Use OpenFermion to extract molecular integrals
        molecular_data = MolecularData(geometry, basis, multiplicity, charge)
        molecular_data = run_pyscf(molecular_data, run_scf=True, run_fci=True)
        fermionic_hamiltonian = molecular_data.get_molecular_hamiltonian()
        H = str(jordan_wigner(fermionic_hamiltonian))
        
        hamiltonian_dict = s.parse_hamiltonian(H,num_qubits=size)
        k_indices = [(0,2),(1,3)]
        j_indices = [(0,1),(0,3),(2,1),(2,3)]
        better_than_HF = False
        simulator_HF = s.SparseQuantumSimulator(num_qubits=size, initial_state={int("0011", 2):1.0})
        HF_res = np.real(sum(coef * simulator_HF.expectation_value(pauli_word) for pauli_word, coef in hamiltonian_dict.items()))
        print(f'HF: {HF_res} at R: {bond_length}')
        fci_energy = molecular_data.fci_energy
        print("FCI Energy:", fci_energy, "Hartree")
        
        while not better_than_HF:
            if len(results_re)!=0:
                res = minimize(energy, results_re[-1].x, method="SLSQP",tol=1e-8, options={'maxiter':10000}, args=(hamiltonian_dict, size, "0011", k_indices, j_indices, 'paper', True, False))
            else:
                res = minimize(energy, np.random.rand(6), method="SLSQP",tol=1e-8, options={'maxiter':10000}, args=(hamiltonian_dict, size, "0011", k_indices, j_indices, 'paper', True, False))
            print(f'uCJ: {res.fun}')
            if not np.isclose(res.fun,HF_res) and res.fun < HF_res:
                better_than_HF = True
            else:
                print(f"I am stuck in HF at R: {bond_length}")
        results_re.append(res)
        de_re.append(res['fun']-fci_energy)
        
        real = False
        imag = True
        param=results_re[-1]['x']
        initial_state='0011'
        k_indices = [(0,2),(1,3)]
        j_indices = [(0,1),(0,3),(2,1),(2,3)]
        if imag:
            K_mat=np.zeros((size,size), dtype=np.complex128)
        else:
            K_mat=np.zeros((size,size))
        if real:
            for d, (ind1,ind2) in enumerate(k_indices):
                K_mat[ind1][ind2] -= param[d]
                K_mat[ind2][ind1] += param[d]
        if real and imag:
            for d,(ind1,ind2) in enumerate(k_indices):
                K_mat[ind1][ind2] += 1j*param[d+len(k_indices)]
                K_mat[ind2][ind1] += 1j*param[d+len(k_indices)]
        if (not real) and imag:
            for d,(ind1,ind2) in enumerate(k_indices):
                K_mat[ind1][ind2] += 1j*param[d]
                K_mat[ind2][ind1] += 1j*param[d] 
        J_mat=np.zeros((size,size))
        for d, (ind1,ind2) in enumerate(j_indices):
            J_mat[ind1][ind2] += param[-1-d]
            J_mat[ind2][ind1] += param[-1-d]  
        
        simulator = s.SparseQuantumSimulator(num_qubits=size, initial_state={int(initial_state, 2):1.0})
        s.create_ucj_circuit(simulator, K_mat, J_mat)
        simulator.state_purification(1e-6)
        for csf_key in CSFs.keys():
            simulator2 = s.SparseQuantumSimulator(4,initial_state={int(key, 2): value for key, value in CSFs[csf_key].items()})
            overlaps_re[csf_key].append(simulator.overlap(simulator2.state_vector,'IIII'))
        simulators_re.append(simulator) 

    plt.rcParams['font.family'] = 'Serif'
    plt.figure(figsize=(5,5))
    plt.plot(np.linspace(0.5,3.0,26),np.array(de_re),'--',color="crimson", label="Re-uCJ dissociation",marker='*')
    plt.plot([0.5,3.0],[1/627.5,1/627.5],'--',c='black',label='Chemical accuracy')

    plt.xlim([0.5,3.0])
    plt.ylim([-0.005,0.08])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.ylabel(r"E-E$_{FCI}$ [a.u.]", fontsize=14)
    plt.xlabel(r"H-H internuclear distance [Å]", fontsize=14)
    plt.tight_layout()
    plt.savefig("H2_sto3G_re_uCJ_RHF.jpg", dpi=600)       