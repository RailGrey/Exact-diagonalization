import numpy as np
from scipy.sparse import kron
import matplotlib.pyplot as plt
from itertools import combinations

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def ising_hamiltonian(N, J=1.0, h=0.0):
    """Construct the Ising Hamiltonian with transverse field for N spins."""
    sx = pauli_x()
    sz = pauli_z()
    I = np.eye(2)
    H = np.zeros((2**N, 2**N))
    # Interaction term
    for i in range(N):
        op = 1
        for j in range(N):
            if j == i:
                op = kron(op, sz)
            elif j == (i+1)%N:
                op = kron(op, sz)
            else:
                op = kron(op, I)
        H -= J * op
    # Transverse field term
    for i in range(N):
        op = 1
        for j in range(N):
            if j == i:
                op = kron(op, sx)
            else:
                op = kron(op, I)
        H -= h * op
    return H

# --- Ising Hamiltonian in fixed S_z subspace ---

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

def ising_hamiltonian_fixed_sz(N, J=1.0, h=0.0, sz_value=0):
    """
    Construct the Ising Hamiltonian in the subspace with fixed total S_z.
    Returns: Hamiltonian matrix (dense), basis_states (list of int)
    """
    basis = basis_states_fixed_sz(N, sz_value)
    dim = len(basis)
    state_index = {state: i for i, state in enumerate(basis)}
    H = np.zeros((dim, dim))
    for idx, state in enumerate(basis):
        # Diagonal: Ising interaction
        e = 0
        for i in range(N):
            # (state >> i) shifts the bits of 'state' right by i places,
            # so the least significant bit now represents the spin at site i.
            # '& 1' extracts that bit: 1 means spin up, 0 means spin down.
            si = 1 if (state >> i) & 1 else -1
            neighbor = (i + 1) % N
            sj = 1 if (state >> neighbor) & 1 else -1
            print(si, sj)
            e += -J * si * sj
        H[idx, idx] = e
        # Off-diagonal: Transverse field (flips one spin)
        if h != 0:
            for i in range(N):
                # Flip spin at i
                flipped = state ^ (1 << i)
                # Only connect if flipped state is in the subspace
                if flipped in state_index:
                    jdx = state_index[flipped]
                    H[idx, jdx] += -h
    return H, basis