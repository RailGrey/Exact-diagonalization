import numpy as np
from scipy.sparse import kron
import matplotlib.pyplot as plt
from itertools import combinations

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_y():
    return np.array([[0, -1j], [1j, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def xy_hamiltonian(N, J=1.0, h=0.0):
    """Construct the XY Hamiltonian for N spins."""
    sx = pauli_x()
    sy = pauli_y()
    sz = pauli_z()
    I = np.eye(2)
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    # XY interaction terms: S_i^x S_{i+1}^x + S_i^y S_{i+1}^y
    for i in range(N):
        # S_i^x S_{i+1}^x term
        op_xx = 1
        for j in range(N):
            if j == i:
                op_xx = kron(op_xx, sx)
            elif j == (i+1)%N:
                op_xx = kron(op_xx, sx)
            else:
                op_xx = kron(op_xx, I)
        H -= J * op_xx
        
        # S_i^y S_{i+1}^y term
        op_yy = 1
        for j in range(N):
            if j == i:
                op_yy = kron(op_yy, sy)
            elif j == (i+1)%N:
                op_yy = kron(op_yy, sy)
            else:
                op_yy = kron(op_yy, I)
        H -= J * op_yy
    
    # Transverse field term: h * S_i^z
    for i in range(N):
        op = 1
        for j in range(N):
            if j == i:
                op = kron(op, sz)
            else:
                op = kron(op, I)
        H -= h * op
    
    return H

def basis_states_fixed_sz(N, sz_value):
    """
    Generate basis states (as integers) with fixed total S_z.
    sz_value: total S_z (e.g., 0 for half up, half down)
    Returns: list of basis state integers
    """
    n_up = int(sz_value + N/2)
    basis = []
    # For each way to choose n_up spins to be up, create the corresponding basis state
    for ups in combinations(range(N), n_up):
        # Start with all spins down (state = 0)
        state = 0
        # Set the bits for the chosen 'up' spins
        for pos in ups:
            state += 2**pos  # Set spin at position 'pos' to up
        basis.append(state)
    return basis

def xy_hamiltonian_fixed_sz(N, J=1.0, h=0.0, sz_value=0):
    """
    Construct the XY Hamiltonian in the subspace with fixed total S_z.
    Returns: Hamiltonian matrix (dense), basis_states (list of int)
    """
    basis = basis_states_fixed_sz(N, sz_value)
    dim = len(basis)
    state_index = {state: i for i, state in enumerate(basis)}
    H = np.zeros((dim, dim), dtype=complex)
    
    for idx, state in enumerate(basis):
        # Diagonal: Transverse field term (S_i^z)
        e = 0
        for i in range(N):
            # (state >> i) shifts the bits of 'state' right by i places,
            # so the least significant bit now represents the spin at site i.
            # '& 1' extracts that bit: 1 means spin up, 0 means spin down.
            si = 1 if (state >> i) & 1 else -1
            e += -h * si
        H[idx, idx] = e
        
        # Off-diagonal: XY interaction terms
        for i in range(N):
            neighbor = (i + 1) % N
            
            # Check if we can flip spins at i and i+1
            # For XX term: flip both spins (if they're different)
            # For YY term: flip both spins with phase factor
            
            # Get current spin values
            si = 1 if (state >> i) & 1 else -1
            sj = 1 if (state >> neighbor) & 1 else -1
            
            # Create flipped state
            flipped = state ^ ((1 << i) | (1 << neighbor))
            
            # Only connect if flipped state is in the subspace
            if flipped in state_index:
                jdx = state_index[flipped]
                
                # XX term: -J * (S_i^x S_{i+1}^x)
                # This connects states where spins at i and i+1 are flipped
                if si != sj:  # Only non-zero when spins are different
                    H[idx, jdx] += -J
                
                # YY term: -J * (S_i^y S_{i+1}^y)
                # This has the same matrix elements as XX but with phase factors
                # For spin-1/2, the YY term is identical to XX term
                if si != sj:
                    H[idx, jdx] += -J
    
    return H, basis

# Parameters
N = 4         # Number of spins
J = 1.0       # Coupling constant
h = 0.0       # Transverse field

# Build full Hamiltonian
H = xy_hamiltonian(N, J, h)

# Diagonalize
eigvals, eigvecs = np.linalg.eigh(H)

print("Full Hilbert space eigenvalues (energies):")
print(eigvals)

# Example usage for fixed S_z sector
sz_value = 0  # total S_z = 0 (2 up, 2 down for N=4)

H_sz, basis_sz = xy_hamiltonian_fixed_sz(N, J, h, sz_value)
eigvals_sz, eigvecs_sz = np.linalg.eigh(H_sz)
print(f"\nEigenvalues (energies) in S_z = {sz_value} sector:")
print(eigvals_sz)

print(f"\nFull Hilbert space dimension: {2**N}")
print(f"Fixed S_z sector dimension: {len(basis_sz)}")
