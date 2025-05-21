import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
import scipy
import re

# ==============================
# SparseQuantumSimulator Class
# ==============================

class SparseQuantumSimulator:
    def __init__(self, num_qubits, initial_state=None):
        """
        Initialize the quantum simulator.
        Args:
            num_qubits (int): Number of qubits.
            initial_state (dict or None): Optional dictionary specifying initial state amplitudes.
        """
        self.num_qubits = num_qubits
        self.state_vector = dok_matrix((2**num_qubits, 1), dtype=np.complex128)

        #To track gate counts
        self.one_qubit_gate_count = 0
        self.two_qubit_gate_count = 0
        
        if initial_state:
            norm = sum(abs(amplitude)**2 for amplitude in initial_state.values())
            if not np.isclose(norm, 1.0):
                print("Warning: Initial state is not normalized. Normalizing.")
                norm_factor = np.sqrt(norm)
                for index, amplitude in initial_state.items():
                    self.state_vector[index, 0] = amplitude / norm_factor
            else:
                for index, amplitude in initial_state.items():
                    self.state_vector[index, 0] = amplitude
        else:
            self.state_vector[0, 0] = 1.0  # Default to |0...0>

    def is_unitary(self, matrix):
        """
        Check if a matrix is unitary.
        Args:
            matrix (numpy array): Matrix to check.
        Returns:
            bool: True if unitary, False otherwise.
        """
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix @ matrix.conj().T, identity)

    def normalize_state_vector(self):
        """
        Normalize the state vector.
        """
        norm = np.sqrt(sum(abs(amplitude)**2 for amplitude in self.state_vector.values()))
        if not np.isclose(norm, 1.0):
            for index in self.state_vector.keys():
                self.state_vector[index] /= norm

    def get_statevector(self):
        """
        Get the current statevector as a dictionary of bitstrings and corresponding coefficients.
        Returns:
            dict: Bitstrings as keys and complex amplitudes as values.
        """
        state_dict = {}
        for index, amplitude in self.state_vector.items():
            bitstring = format(index[0], f'0{self.num_qubits}b')
            state_dict[bitstring] = amplitude
        return state_dict

    def apply_single_qubit_gate(self, qubit_idx, gate_matrix, one_qubit_gate_weight = 1):
        """
        Apply a single-qubit gate to the specified qubit.
        Args:
            qubit_idx (int): Index of the qubit to apply the gate to.
            gate_matrix (2x2 numpy array): The gate matrix.
        """
        if not self.is_unitary(gate_matrix):
            print("Error: Gate matrix is not unitary. Operation aborted.")
            return

        dim = 2**self.num_qubits
        new_state = dok_matrix((dim, 1), dtype=np.complex128)

        for index, amplitude in self.state_vector.items():
            binary_index = format(index[0], f'0{self.num_qubits}b')
            bit = int(binary_index[self.num_qubits - 1 - qubit_idx])
            new_amplitudes = gate_matrix[:, bit] * amplitude

            for new_bit in [0, 1]:
                new_index = list(binary_index)
                new_index[self.num_qubits - 1 - qubit_idx] = str(new_bit)
                new_index_str = ''.join(new_index)
                new_index_decimal = int(new_index_str, 2)
                new_state[new_index_decimal, 0] += new_amplitudes[new_bit]

        self.state_vector = new_state
        self.normalize_state_vector()

        #Increment single-qubit gate counter
        self.one_qubit_gate_count += one_qubit_gate_weight

    def apply_two_qubit_gate(self, qubit1_idx, qubit2_idx, gate_matrix, two_qubit_gate_weight = 1, one_qubit_gate_weight = 1):
        """
        Apply an arbitrary two-qubit gate.
        Args:
            qubit1_idx (int): Index of the first qubit.
            qubit2_idx (int): Index of the second qubit.
            gate_matrix (4x4 numpy array): The two-qubit gate matrix.
        """
        if not self.is_unitary(gate_matrix):
            print("Error: Gate matrix is not unitary. Operation aborted.")
            return

        dim = 2**self.num_qubits
        new_state = dok_matrix((dim, 1), dtype=np.complex128)

        for index, amplitude in self.state_vector.items():
            binary_index = format(index[0], f'0{self.num_qubits}b')
            bit1 = int(binary_index[self.num_qubits - 1 - qubit1_idx])
            bit2 = int(binary_index[self.num_qubits - 1 - qubit2_idx])
            combined_index = (bit1 << 1) | bit2

            new_amplitudes = gate_matrix[:, combined_index] * amplitude

            for new_combined_index in range(4):
                new_bit1 = (new_combined_index >> 1) & 1
                new_bit2 = new_combined_index & 1
                new_index = list(binary_index)
                new_index[self.num_qubits - 1 - qubit1_idx] = str(new_bit1)
                new_index[self.num_qubits - 1 - qubit2_idx] = str(new_bit2)
                new_index_str = ''.join(new_index)
                new_index_decimal = int(new_index_str, 2)
                new_state[new_index_decimal, 0] += new_amplitudes[new_combined_index]

        self.state_vector = new_state
        self.normalize_state_vector()
        
        #Increment two-qubit gate counter
        self.two_qubit_gate_count += two_qubit_gate_weight
        self.one_qubit_gate_count += one_qubit_gate_weight
    
    def apply_controlled_gate(self, control_qubit, target_qubit, gate_matrix, two_qubit_gate_weight = 1, one_qubit_gate_weight = 1):
        """
        Apply a controlled gate (e.g., CNOT, Toffoli) to the specified qubits.
        Args:
            control_qubit (int): Index of the control qubit.
            target_qubit (int): Index of the target qubit.
            gate_matrix (2x2 numpy array): The single-qubit gate matrix to apply if the control qubit is 1.
        """
        if not self.is_unitary(gate_matrix):
            print("Error: Gate matrix is not unitary. Operation aborted.")
            return

        dim = 2**self.num_qubits
        new_state = dok_matrix((dim, 1), dtype=np.complex128)

        for index, amplitude in self.state_vector.items():
            binary_index = format(index[0], f'0{self.num_qubits}b')
            control_bit = int(binary_index[self.num_qubits - 1 - control_qubit])
            target_bit = int(binary_index[self.num_qubits - 1 - target_qubit])

            if control_bit == 1:
                new_amplitudes = gate_matrix[:, target_bit] * amplitude
                for new_bit in [0, 1]:
                    new_index = list(binary_index)
                    new_index[self.num_qubits - 1 - target_qubit] = str(new_bit)
                    new_index_str = ''.join(new_index)
                    new_index_decimal = int(new_index_str, 2)
                    new_state[new_index_decimal, 0] += new_amplitudes[new_bit]
            else:
                new_state[index] = amplitude

        self.state_vector = new_state
        self.normalize_state_vector()
        
        #Increment two-qubit gate counter
        self.two_qubit_gate_count += two_qubit_gate_weight
        self.one_qubit_gate_count += one_qubit_gate_weight

    def single_one_qubit_measurement(self, qubit_idx):
        """
        Measure the specified qubit and return 0 or 1 with corresponding probabilities (State is collapsed).
        Args:
            qubit_idx (int): Index of the qubit to measure.
        Returns:
            int: Measurement outcome (0 or 1).
        """
        probabilities = [0.0, 0.0]

        for index, amplitude in self.state_vector.items():
            binary_index = format(index[0], f'0{self.num_qubits}b')
            bit = int(binary_index[self.num_qubits - 1 - qubit_idx])
            probabilities[bit] += abs(amplitude)**2

        outcome = np.random.choice([0, 1], p=probabilities)

        new_state = dok_matrix((2**self.num_qubits, 1), dtype=np.complex128)
        for index, amplitude in self.state_vector.items():
            binary_index = format(index[0], f'0{self.num_qubits}b')
            bit = int(binary_index[self.num_qubits - 1 - qubit_idx])
            if bit == outcome:
                new_state[index] = amplitude / np.sqrt(probabilities[outcome])

        self.state_vector = new_state
        return outcome
        
    def single_full_state_measurement(self):
        """
        Measure the entire state and return a bitstring corresponding to a full measurement (State is not collapsed).
        Returns:
            str: Bitstring representing the measurement outcome for all qubits.
        """
        probabilities = {}
    
        for index, amplitude in self.state_vector.items():
            bitstring = format(index[0], f'0{self.num_qubits}b')
            probabilities[bitstring] = abs(amplitude)**2
    
        # Normalize probabilities to sum to 1 (in case of rounding issues)
        total_prob = sum(probabilities.values())
        for key in probabilities:
            probabilities[key] /= total_prob
    
        # Sample a bitstring according to the probability distribution
        bitstrings = list(probabilities.keys())
        prob_values = list(probabilities.values())
    
        sampled_bitstring = np.random.choice(bitstrings, p=prob_values)
        return sampled_bitstring

    def apply_full_unitary(self, unitary_matrix):
        """
        Apply a full 2^n_qubit by 2^n_qubit unitary matrix to the quantum state.
        Args:
            unitary_matrix (numpy array): A dense 2^n_qubit by 2^n_qubit unitary matrix.
        """
        # Check if the unitary matrix is square and of the correct size
        dim = 2 ** self.num_qubits
        if unitary_matrix.shape != (dim, dim):
            raise ValueError(f"Unitary matrix must be of shape ({dim}, {dim}).")

        # Check if the matrix is unitary
        if not np.allclose(unitary_matrix @ unitary_matrix.conj().T, np.eye(dim)):
            raise ValueError("The provided matrix is not unitary.")

        # Convert the dense unitary matrix to a sparse representation
        sparse_unitary = csc_matrix(unitary_matrix)

        # Apply the sparse matrix multiplication to the state vector
        new_state_vector = sparse_unitary @ self.state_vector

        # Update the simulator's state vector
        self.state_vector = dok_matrix(new_state_vector)

        # Normalize the state vector
        self.normalize_state_vector()
        
    def apply_pauli_word(self, pauli_word):
        """
        Apply a Pauli word (e.g., 'XIZ') to the state.
        Args:
            pauli_word (str): Pauli operators (e.g., 'XIY').
        """
        if len(pauli_word) != self.num_qubits:
            print("Error: The length of the Pauli word does not match the number of qubits. Operation aborted.")
            return None
            
        for i, pauli in enumerate(pauli_word[::-1]):
            if pauli == 'I':
                continue
            elif pauli == 'X':
                self.apply_single_qubit_gate(i, np.array([[0, 1], [1, 0]]))  # Pauli-X
            elif pauli == 'Y':
                self.apply_single_qubit_gate(i, np.array([[0, -1j], [1j, 0]]))  # Pauli-Y
            elif pauli == 'Z':
                self.apply_single_qubit_gate(i, np.array([[1, 0], [0, -1]]))  # Pauli-Z

    def apply_exponential_pauli_word(self, pauli_word, theta):
        """
        Apply exp(-i*theta*P) where P is a Pauli word.
        Args:
            pauli_word (str): Pauli operators (e.g., 'ZI').
            theta (float): Angle of rotation.
        """
        # Save the original state vector
        original_state = self.state_vector.copy()
        
        # Apply the Pauli word to compute P|psi⟩
        self.apply_pauli_word(pauli_word)
        p_psi = self.state_vector.copy()  # P|psi⟩
        # Calculate the new state as exp(-itP)|psi⟩ = cos(t)|psi⟩ - i*sin(t)P|psi⟩
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        new_state = dok_matrix(self.state_vector.shape, dtype=np.complex128)
    
        for index in original_state.keys() | p_psi.keys():
            psi_amplitude = original_state.get(index, 0) * cos_theta
            p_psi_amplitude = -1j * sin_theta * p_psi.get(index, 0)
            new_state[index] = psi_amplitude + p_psi_amplitude
    
        # Update the state vector and normalize
        self.state_vector = new_state
        self.normalize_state_vector()

    def parse_pauli_expression(self, expression):
        """
        Parse expressions like "(0.5+0.5j)*XYII + 0.5*YXII - 1j*ZZII" into a list of (coefficient, Pauli word).
        """
        expression = expression.replace(' ', '')

        # Match full Pauli terms
        term_pattern = re.compile(r'([+-]?(?:\([^\)]+\)|[\d\.]+(?:[+-][\d\.]+)?[jJ]?))\*([IXYZ]+)')
        parsed_terms = []

        for match in term_pattern.finditer(expression):
            coef_str, pauli_word = match.groups()

            # Remove surrounding parentheses if present
            if coef_str.startswith('(') and coef_str.endswith(')'):
                coef_str = coef_str[1:-1]

            try:
                coefficient = complex(coef_str.replace('J', 'j'))
            except ValueError as e:
                raise ValueError(f"Invalid complex number format in: '{coef_str}'") from e

            parsed_terms.append((coefficient, pauli_word))

        return parsed_terms
    
    def apply_exponential_pauli_expression(self, pauli_expression):
        """
        Apply exp(H) where H is a sum of Pauli words. Note the expression should result in anti Hermitian matrix.
        Args:
            pauli_expression (str): A string of Pauli terms with coefficients (e.g., "0.5*XYII + 0.5*YXII").
        """
        pauli_dict = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        # Parse the Pauli expression
        parsed_terms = self.parse_pauli_expression(pauli_expression)
        dim = 2 ** self.num_qubits

        # Build the resultinh matrix H
        H = np.zeros((dim, dim), dtype=np.complex128)
        for coeff, pauli_word in parsed_terms:
            full_pauli_matrix = 1
            for pauli in pauli_word:  # Apply Kronecker in reverse order
                full_pauli_matrix = np.kron(pauli_dict[pauli], full_pauli_matrix)
            H += coeff * full_pauli_matrix

        # Compute exp(-i*H)
        U = scipy.linalg.expm(H)
        if not self.is_unitary(U):
            raise ValueError("Pauli expression is not anti-Hermitian.")
        
        # Convert to sparse matrix and apply
        sparse_U = csc_matrix(U)
        self.state_vector = dok_matrix(sparse_U @ self.state_vector)

        # Normalize the state vector
        self.normalize_state_vector()
    
    def sample_measurements(self, num_shots):
        """
        Perform measurements on the full quantum state and return the outcome frequencies.
        Args:
            num_shots (int): Number of measurements (shots) to perform.
        Returns:
            dict: Dictionary with bitstrings as keys and counts as values.
        """
        probabilities = {}
        for index, amplitude in self.state_vector.items():
            bitstring = format(index[0], f'0{self.num_qubits}b')
            probabilities[bitstring] = abs(amplitude)**2

        bitstrings = list(probabilities.keys())
        prob_values = list(probabilities.values())

        # Sample bitstrings according to their probabilities
        measurements = np.random.choice(bitstrings, size=num_shots, p=prob_values)

        # Count the number of occurrences of each bitstring
        result_counts = {bitstring: 0 for bitstring in bitstrings}
        for measurement in measurements:
            result_counts[measurement] += 1

        # Filter out zero counts for conciseness
        return {k: v for k, v in result_counts.items() if v > 0}

    def sample_expectation_value(self, pauli_word, num_shots):
        """
        Estimate the expectation value of a Pauli word using sample measurements.
        Args:
            pauli_word (str): Pauli operators (e.g., 'XIY').
            num_shots (int): Number of measurements (shots) to perform.
        Returns:
            float: Estimated expectation value based on measurements.
        """
        if len(pauli_word) != self.num_qubits:
            print("Error: The length of the Pauli word does not match the number of qubits. Operation aborted.")
            return None
            
        # Copy the current state to avoid modifying it
        transformed_sim = SparseQuantumSimulator(self.num_qubits)
        transformed_sim.state_vector = self.state_vector.copy()

        # Apply rotations to bring each Pauli operator to the Z basis
        for i, pauli in enumerate(pauli_word[::-1]):
            if pauli == 'I':
                continue  # Identity, no rotation
            elif pauli == 'X':
                # Apply Hadamard to convert X basis to Z basis
                h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                transformed_sim.apply_single_qubit_gate(i, h_gate)
            elif pauli == 'Y':
                # Apply S†H to convert Y basis to Z basis
                s_dagger_h = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
                transformed_sim.apply_single_qubit_gate(i, s_dagger_h)
            elif pauli == 'Z':
                pass  # Z is already in the Z basis

        # Perform measurements
        counts = transformed_sim.sample_measurements(num_shots)
        #print(counts)
        # Calculate expectation value: +1 for even parity, -1 for odd parity
        expectation = 0
        for bitstring, count in counts.items():
            sign = 1
            for i, pauli in enumerate(pauli_word):
                if pauli in ['X', 'Y', 'Z'] and bitstring[i] == '1':
                    sign *= -1
            expectation += sign * count

        # Normalize expectation by the total number of shots
        return expectation / num_shots

    def overlap(self, state, pauli_word):
        """
        Compute the overlap element for a Pauli word <psi_i|P|psi_j>.
        Args:
            state (dok_matrix): bra state.
            pauli_word (str): Pauli operators (e.g., 'XIY').
        Returns:
            float: Expectation value.
        """
        if len(pauli_word) != self.num_qubits:
            print("Error: The length of the Pauli word does not match the number of qubits. Operation aborted.")
            return None
            
        original_state = self.state_vector.copy()
        self.apply_pauli_word(pauli_word)
        expectation = 0
        for index, amplitude in self.state_vector.items():
            expectation += amplitude*np.conjugate(state.get(index))
        self.state_vector = original_state
        return expectation
    
    def expectation_value(self, pauli_word):
        """
        Compute the expectation value of a Pauli word.
        Args:
            pauli_word (str): Pauli operators (e.g., 'XIY').
        Returns:
            float: Expectation value.
        """
        if len(pauli_word) != self.num_qubits:
            print("Error: The length of the Pauli word does not match the number of qubits. Operation aborted.")
            return None
            
        original_state = self.state_vector.copy()
        self.apply_pauli_word(pauli_word)
        expectation = 0
        for index, amplitude in self.state_vector.items():
            expectation += amplitude*np.conjugate(original_state.get(index))
        self.state_vector = original_state
        return expectation
    
    def state_purification(self, threshold):
        """
        Purifies the state vector by removing components with absolute values below the threshold.
        Args:
            threshold (float): The threshold below which state amplitudes are removed.
        """
        # Remove components with small coefficients
        for index, amplitude in list(self.state_vector.items()):
            if abs(amplitude) < threshold:
                del self.state_vector[index]

        # Re-normalize the state vector after purification
        self.normalize_state_vector()

# ==============================
# Utils
# ==============================

def format_complex(z, tol=1e-7, width=10, digits=6):
    re = z.real
    im = z.imag
    re = 0 if abs(re) < tol else re
    im = 0 if abs(im) < tol else im

    if im == 0:
        s = f"{re:.6f}"  # real only
    elif re == 0:
        s = f"{im:.6f}j"  # imag only
    else:
        sign = '+' if im > 0 else '-'
        s = f"{re:.3f}{sign}{abs(im):.3f}j"  # full complex

    return f"{s:>{width}}"

def random_antihermitian_matrix(N, seed=None):
    """
    Generate a random anti-Hermitian matrix of size NxN.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    A_antiH = 0.5 * (A - A.conj().T)
    return A_antiH


# ==============================
# Circuit Builders
# ==============================

def create_ucj_circuit(simulator, K_mat, J_mat, K_mat_implementation='paper', thresh=1e-10, debug=False):
    """
    Creates a circuit with exact orbital rotations (K-matrix) and Jastrow correlator (J-matrix).
    Args:
        simulator (SparseQuantumSimulator): Quantum simulator instance.
        K_mat (numpy array): Anti-Hermitian matrix for orbital rotations.
        J_mat (numpy array): Symmetric matrix for Jastrow correlator.
    """
    # Apply exact orbital rotation circuit
    apply_exact_orbital_rotation(simulator, K_mat, K_mat_implementation=K_mat_implementation, thresh=thresh, debug=debug)
    
    # Apply Jastrow correlator
    apply_exact_jastrow_correlator(simulator, J_mat)

    # Apply exact orbital rotation circuit
    apply_exact_orbital_rotation(simulator, -K_mat, K_mat_implementation=K_mat_implementation, thresh=thresh, debug=debug)

def apply_exact_orbital_rotation(simulator, K_mat, K_mat_implementation='paper', exponentiated=False, thresh=1e-10, debug=False):
    """
    Applies the exact orbital rotations using the K matrix.
    Args:
        simulator (SparseQuantumSimulator): Quantum simulator instance.
        K_mat (numpy array): Anti-Hermitian matrix for orbital rotations.
    """
    # 1. Check anti-Hermicity (K^dag = -K)
    if not np.allclose(K_mat.T.conj(), -K_mat) and not exponentiated:
        raise ValueError("K matrix is not anti-Hermitian.")
    elif exponentiated and not simulator.is_unitary(K_mat):
        raise ValueError("Matrix is not Unitary.")        

    # 2. Check if K is real or complex
    is_real = np.all(np.isreal(K_mat))

    # 3. Compute matrix exponential exp(K)
    if not exponentiated:
        k_matrix = scipy.linalg.expm(K_mat)  # This gives the matrix `k` from exp(K)
    else:
        k_matrix = np.array(K_mat) 
    
    if debug:
        for row in k_matrix:
            print(' '.join(format_complex(0) if np.abs(x) < 1e-7 else format_complex(x) for x in row))                       
    # 4. Perform Givens rotations and QR-like decomposition
    if is_real:
        theta_list = []
        for i in range(len(k_matrix)-1):
            for j in range(len(k_matrix)-1,i,-1):
                if np.abs(k_matrix[j][i]) > thresh:
                    if np.abs(k_matrix[j-1][i]) > thresh:
                        theta = np.arctan(-k_matrix[j][i]/k_matrix[j-1][i])
                    else:
                        theta = np.pi/2
                    rotation_matrix = np.eye(len(k_matrix))
                    rotation_matrix[j-1][j-1]=np.cos(theta)
                    rotation_matrix[j-1][j]=-np.sin(theta)
                    rotation_matrix[j][j-1]=np.sin(theta)
                    rotation_matrix[j][j]=np.cos(theta)
                    k_matrix = rotation_matrix@k_matrix
                    theta_list.append(((j,j-1),theta))
                    if debug:
                        print(f"Step_1: i={i}, j={j}, theta={theta}")
                        for row in k_matrix:
                            print(' '.join(format_complex(0) if np.abs(x) < 1e-7 else format_complex(x) for x in row))

    elif K_mat_implementation == 'paper':
        phi_theta_list = []
        for i in range(len(k_matrix)-1):        
            for j in range(len(k_matrix)-1,i,-1):
                if np.abs(k_matrix[j][i]) > thresh:
                    if np.abs(k_matrix[j-1][i]) > thresh:
                        r1 = np.abs(k_matrix[j-1][i])
                        phi1 = np.angle(k_matrix[j-1][i])
                        r2 = np.abs(k_matrix[j][i])
                        phi2 = np.angle(k_matrix[j][i])
                        phi = phi2 - phi1
                        theta = np.arctan(-r2/r1)
                    else:
                        #phi2 = np.angle(k_matrix[j][i])
                        phi = 0
                        theta = np.pi/2
                    rotation_matrix = np.eye(len(k_matrix),dtype=np.complex128)
                    rotation_matrix[j-1][j-1]=np.cos(theta)
                    rotation_matrix[j-1][j]=-np.sin(theta)*np.exp(-1j*phi)
                    rotation_matrix[j][j-1]=np.sin(theta)*np.exp(1j*phi)
                    rotation_matrix[j][j]=np.cos(theta)
                    k_matrix = rotation_matrix@k_matrix 
                    phi_theta_list.append(((j,j-1),phi,theta))
                    if debug:
                        try: print(f"Step_1: i={i}, j={j}, phi={phi}, theta={theta}, phi1={phi1}, phi2={phi2}, r1={r1}, r2={r2}")
                        except: print(f"Step_1: i={i}, j={j}, phi={phi}, theta={theta}")
                        for row in k_matrix:
                            print(' '.join(format_complex(0) if np.abs(x) < 1e-7 else format_complex(x) for x in row))                   

    else:
    #This is an alternative way of implementation of e^K_g and e^K_Im
        theta_list = []
        for i in range(len(k_matrix)-1):
            for j in range(len(k_matrix)-1,i,-1):
                if np.abs(k_matrix[j][i]) > thresh:
                    a=np.real(k_matrix[j-1][i])
                    b=np.imag(k_matrix[j-1][i])
                    c=np.real(k_matrix[j][i])
                    d=np.imag(k_matrix[j][i])
                    if np.abs(a*d-b*c) > thresh:
                        theta_1 = np.arctan(float((-(d**2+c**2-a**2-b**2)-np.sqrt((d**2+c**2-a**2-b**2)**2-4*(c*b-a*d)*(a*d-b*c)))/(2*(a*d-b*c))))
                        rotation_matrix = np.eye(len(k_matrix),dtype=np.complex128)
                        rotation_matrix[j-1][j-1]=np.cos(theta_1)
                        rotation_matrix[j-1][j]=1j*np.sin(theta_1)
                        rotation_matrix[j][j-1]=1j*np.sin(theta_1)
                        rotation_matrix[j][j]=np.cos(theta_1)
                        k_matrix = rotation_matrix@k_matrix
                        if debug:
                            print(f"Step_1: i={i}, j={j}, theta={theta_1}")
                            for row in k_matrix:
                                print(' '.join(format_complex(0) if np.abs(x) < 1e-7 else format_complex(x) for x in row))
                    else:
                        theta_1 = 0

                    if np.abs(np.real(k_matrix[j-1][i])) > thresh:
                        theta_2 = np.arctan(-np.real(k_matrix[j][i])/np.real(k_matrix[j-1][i]))
                    elif np.abs(np.imag(k_matrix[j-1][i])) > thresh:
                        theta_2 = np.arctan(-np.imag(k_matrix[j][i])/np.imag(k_matrix[j-1][i]))                        
                    else:
                        theta_2 = np.pi/2
                    rotation_matrix = np.eye(len(k_matrix))
                    rotation_matrix[j-1][j-1]=np.cos(theta_2)
                    rotation_matrix[j-1][j]=-np.sin(theta_2)
                    rotation_matrix[j][j-1]=np.sin(theta_2)
                    rotation_matrix[j][j]=np.cos(theta_2)
                    k_matrix = rotation_matrix@k_matrix 
                    if debug:
                        print(f"Step_2: i={i}, j={j}, theta={theta_2}")
                        for row in k_matrix:
                            print(' '.join(format_complex(0) if np.abs(x) < 1e-7 else format_complex(x) for x in row))
                    theta_list.append(((j,j-1),theta_1, theta_2))  
    if debug:
        print("FINAL K MAT")
        for row in k_matrix:
            print(' '.join(format_complex(0) if np.abs(x) < 1e-7 else format_complex(x) for x in row))
    # 5. Apply rotations to the simulator (nearest-neighbor qubits)
    if is_real:
        phases = np.diag(k_matrix)
        for p in range(len(phases)): 
            phase_angle = np.angle(phases[p])
            single_qubit_rotation = np.array([[1, 0], [0, np.exp(1j * phase_angle)]])
            simulator.apply_single_qubit_gate(p, single_qubit_rotation)  
            
        for ind, theta in theta_list[::-1]:

            if theta !=0:
                rotation_matrix = np.eye(4)
                rotation_matrix[1][1]=np.cos(theta)
                rotation_matrix[1][2]=np.sin(theta)
                rotation_matrix[2][1]=-np.sin(theta)
                rotation_matrix[2][2]=np.cos(theta)
                simulator.apply_two_qubit_gate(ind[0], ind[1], rotation_matrix, two_qubit_gate_weight = 4, one_qubit_gate_weight = 2)

    elif K_mat_implementation == 'paper':
        phases = np.diag(k_matrix)
        for p in range(len(phases)): 
            phase_angle = np.angle(phases[p])
            single_qubit_rotation = np.array([[1, 0], [0, np.exp(1j * phase_angle)]])
            simulator.apply_single_qubit_gate(p, single_qubit_rotation) 
        for ind, phi, theta in phi_theta_list[::-1]:
            if ((theta != 0) or (phi != 0)):        
                rotation_matrix = np.eye(4,dtype=np.complex128)
                rotation_matrix[1][1]=np.cos(theta)
                rotation_matrix[1][2]=np.sin(theta)*np.exp(-1j*phi)
                rotation_matrix[2][1]=-np.sin(theta)*np.exp(1j*phi)
                rotation_matrix[2][2]=np.cos(theta)
                simulator.apply_two_qubit_gate(ind[0], ind[1], rotation_matrix, two_qubit_gate_weight = 3, one_qubit_gate_weight = 8)
    else:
        phases = np.diag(k_matrix)
        for p in range(len(phases)): 
            phase_angle = np.angle(phases[p])
            single_qubit_rotation = np.array([[1, 0], [0, np.exp(1j * phase_angle)]])
            simulator.apply_single_qubit_gate(p, single_qubit_rotation)            
        for ind, theta_1, theta_2 in theta_list[::-1]:
            if theta_2!=0:
                rotation_matrix = np.eye(4)
                rotation_matrix[1][1]=np.cos(theta_2)
                rotation_matrix[1][2]=np.sin(theta_2)
                rotation_matrix[2][1]=-np.sin(theta_2)
                rotation_matrix[2][2]=np.cos(theta_2)
                simulator.apply_two_qubit_gate(ind[0], ind[1], rotation_matrix, two_qubit_gate_weight = 4, one_qubit_gate_weight = 2)
            if theta_1!=0:
                rotation_matrix = np.eye(4, dtype=np.complex128)
                rotation_matrix[1][1]=np.cos(theta_1)
                rotation_matrix[1][2]=-1j*np.sin(theta_1)
                rotation_matrix[2][1]=-1j*np.sin(theta_1)
                rotation_matrix[2][2]=np.cos(theta_1)
                simulator.apply_two_qubit_gate(ind[0], ind[1], rotation_matrix, two_qubit_gate_weight = 4, one_qubit_gate_weight = 2)

def apply_exact_jastrow_correlator(simulator, J_mat):
    """
    Applies the Jastrow correlator using the J matrix.
    Args:
        simulator (SparseQuantumSimulator): Quantum simulator instance.
        J_mat (numpy array): Symmetric real matrix for Jastrow correlator.
    """
    # 1. Check if J is symmetric
    if not np.allclose(J_mat, J_mat.T):
        raise ValueError("J matrix is not symmetric.")

    if not np.all(np.isreal(J_mat)):
        raise ValueError("J matrix should be real. It will be multiplied by 1j inside the function.")

    # 2. Apply correlated phase rotations between pairs of qubits
    for i in range(J_mat.shape[0]):
        for j in range(i):
            if i!=j: 
                angle = J_mat[i][j]/2
                if angle!=0:
                    #Application of exp(1j*Jij*1/4(1-Z_i)(1-Z_j))
                    I_gate = np.array([[np.exp(1j*angle),0],[0,np.exp(1j*angle)]])
                    Z_gate = np.array([[np.exp(-1j*angle),0],[0,np.exp(1j*angle)]])
                    ZZ_gate = np.array([[np.exp(1j*angle),0,0,0],[0,np.exp(-1j*angle),0,0],[0,0,np.exp(-1j*angle),0],[0,0,0,np.exp(1j*angle)]])
                    simulator.apply_single_qubit_gate(i, I_gate)
                    simulator.apply_single_qubit_gate(i, Z_gate)
                    simulator.apply_single_qubit_gate(j, Z_gate)
                    simulator.apply_two_qubit_gate(i,j,ZZ_gate, two_qubit_gate_weight = 2, one_qubit_gate_weight = 1)

# ==============================
# Pauli Logic
# ==============================

Paulis = {"X": np.array([[0,1],[1,0]], dtype=np.complex128),
"Y": np.array([[0,-1j],[1j,0]], dtype=np.complex128),
"Z": np.array([[1,0],[0,-1]], dtype=np.complex128),
"I": np.array([[1,0],[0,1]], dtype=np.complex128)}

def pw_product(PW1, PW2, phase = False):
    '''Returns a PW = PW1 * PW2.'''

    # Define Plauli Matrix Multiplication Group
    Pauli_group_multiplications = np.array([[(np.complex128(1.0),'I'),(1.0,'X'),(1.0,'Y'),(1.0,'Z')],
                                            [(1.0,'X'),(1.0,'I'),(1.0j,'Z'),(-1.0j,'Y')],
                                            [(1.0,'Y'),(-1.0j,'Z'),(1.0,'I'),(1.0j,'X')],
                                            [(1.0,'Z'),(1.0j,'Y'),(-1.0j,'X'),(1.0,'I')]]) # 0 = I; 1 = X; 2 = Y; 3 = Z
    Paulis_ind = {"X": 1,"Y": 2,"Z": 3,"I": 0}
    
    phase_value = 1.0
    new_PW = ''
    if len(PW1)!=len(PW2):
        print('Cannot multiply Pauliwords of a different length')
        return None
    for pg1, pg2 in zip(PW1, PW2):
        ph, new_pg = Pauli_group_multiplications[Paulis_ind[pg1]][Paulis_ind[pg2]]
        ph = np.complex128(ph)
        new_PW += new_pg
        phase_value = phase_value*ph
    if  phase: return new_PW, phase_value
    else: return new_PW


def jw_pauli_terms(i, j, num_qubits):
    """
    Generate the four Pauli words for a_i^dag a_j using the Jordan-Wigner transformation.
    Returns:
        list of tuples: [(coeff1, pauli_word1), (coeff2, pauli_word2), ...].
    """
    pauli_words = []

    # Initialize base Pauli string as identity
    
    creation = []
    anihilation = []

    base_pauli = ["I"] * num_qubits

    for pg in ["X", "Y"]:
        base_pauli = ["I"] * num_qubits
        for k in range(i):
            base_pauli[k] = "Z"
        base_pauli[i]=pg
        if pg == "Y":
            creation.append((-1j, "".join(base_pauli)))
        else:
            creation.append((1, "".join(base_pauli)))

    for pg in ["X", "Y"]:
        base_pauli = ["I"] * num_qubits
        for k in range(j):
            base_pauli[k] = "Z"
        base_pauli[j]=pg
        if pg == "Y":
            anihilation.append((1j, "".join(base_pauli)))
        else:
            anihilation.append((1, "".join(base_pauli)))
    
    for coef1, pw1 in creation:
        for coef2, pw2 in anihilation:
            new_pw, phase = pw_product(pw1, pw2, phase = True)
            new_coef = coef1*coef2*phase/4
            pauli_words.append((new_coef, new_pw))

    return pauli_words

def jw_occupancy_operators(i,j,num_qubits):
    """
    Generate the four Pauli words for n_i n_j using the Jordan-Wigner transformation.
    Returns:
        list of tuples: [(coeff1, pauli_word1), (coeff2, pauli_word2), ...].
    """
    pauli_words = []

    # Initialize base Pauli string as identity
    
    pauliwords1 = jw_pauli_terms(i,i,num_qubits)
    pauliwords2 = jw_pauli_terms(j,j,num_qubits)
    
    for coef1, pw1 in pauliwords1:
        for coef2, pw2 in pauliwords2:
            new_pw, phase = pw_product(pw1, pw2, phase = True)
            new_coef = coef1*coef2*phase
            pauli_words.append((new_coef, new_pw))

    return pauli_words
    
def k_matrix_to_pauli_string(K_mat):
    """
    Converts a K matrix to a Pauli string using the Jordan-Wigner transformation.
    Args:
        K_mat (numpy array): Anti-Hermitian K matrix of size (n, n).
    Returns:
        str: Pauli string representation of the K matrix.
    """
    num_qubits = K_mat.shape[0]
    pauli_terms = []

    # Process each non-diagonal element of K_mat
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j or abs(K_mat[i, j]) < 1e-12:  # Skip diagonal and insignificant terms
                continue

            coeff = K_mat[i, j]  # Scale by 0.5 for proper decomposition
            for sign, pauli_word in jw_pauli_terms(i, j, num_qubits):
                pauli_terms.append((coeff * sign, pauli_word))
    new_pauli_terms = {}
    for sign, pauli_word in pauli_terms:
        new_pauli_terms[pauli_word] = 0
    for sign, pauli_word in pauli_terms:
        new_pauli_terms[pauli_word] += sign
    pauli_terms_string = []
    for pauli_word, coeff in new_pauli_terms.items():
        coeff_imag = np.imag(coeff)
        if np.abs(coeff_imag)>1e-8:
            pauli_terms_string.append(f"{coeff_imag:+.6g}j*{pauli_word}")
    # Combine all terms into a single string
    return " ".join(pauli_terms_string)

def j_matrix_to_pauli_string(J_mat):
    """
    Converts a J matrix to a Pauli string using the Jordan-Wigner transformation.
    Args:
        J_mat (numpy array): Symmetric J matrix of size (n, n).
    Returns:
        str: Pauli string representation of the K matrix.
    """
    num_qubits = J_mat.shape[0]
    pauli_terms = []

    # Process each non-diagonal element of K_mat
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j or abs(K_mat[i, j]) < 1e-12:  # Skip diagonal and insignificant terms
                continue

            coeff = J_mat[i, j]  # Scale by 0.5 for proper decomposition
            for sign, pauli_word in jw_occupancy_operators(i, j, num_qubits):
                pauli_terms.append((coeff * sign, pauli_word))
    new_pauli_terms = {}
    for sign, pauli_word in pauli_terms:
        new_pauli_terms[pauli_word] = 0
    for sign, pauli_word in pauli_terms:
        new_pauli_terms[pauli_word] += sign
    pauli_terms_string = []
    for pauli_word, coeff in new_pauli_terms.items():
        coeff_imag = np.real(coeff)
        if np.abs(coeff_imag)>1e-8:
            pauli_terms_string.append(f"{coeff_imag:+.6g}j*{pauli_word}")
    # Combine all terms into a single string
    return " ".join(pauli_terms_string)    

def parse_hamiltonian(hamiltonian_str, num_qubits=4):
    """
    Converts a Hamiltonian string into a dictionary of Pauli words and coefficients.
    Args:
        hamiltonian_str (str): Hamiltonian in the string format.
    Returns:
        dict: Dictionary with Pauli words as keys and coefficients as values.
    """
    pauli_dict = {}
    
    # Split the string by newlines to get individual terms
    terms = hamiltonian_str.split(' +\n')
    
    for term in terms:
        # Split each term into coefficient and operator
        coefficient, operator = term.split(' [')
        coefficient = float(coefficient.strip())  # Convert coefficient to float
        operator = operator.strip(']')  # Remove closing bracket
        
        # If no operator (empty string), default to 'I' (identity) for all qubits
        if operator == '':
            pauli_word = 'I' * num_qubits 
        else:
            # Convert to the format needed for your simulator
            pauli_word = ['I'] * num_qubits
            print
            for op in operator.split():
                gate = op[0]  # Pauli gate (X, Y, Z)
                qubit_idx = int(op[1:])  # Qubit index
                pauli_word[qubit_idx] = gate
            pauli_word = ''.join(pauli_word)
        
        # Add to dictionary
        pauli_dict[pauli_word[::-1]] = coefficient
    
    return pauli_dict
