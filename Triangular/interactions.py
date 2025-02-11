import numpy as np
import tensorflow as tf

default_p1 = (1., 0.)
default_p2 = (-1 / 2, np.sqrt(3) / 2)


def coord_to_site_bravais(L, x, y):
    return L * y + x

def site_to_coord_bravais(L, site):
    y = site // L
    x = site - L * y
    return x, y

def generate_triangular(Nx, Ny, p1=default_p1, p2=default_p2):
    triangular_lattice = np.ones((Nx * Ny, 2))

    for i in range(Nx):
        for j in range(Ny):
            originx = i * p1[0] + j * p2[0]
            originy = i * p1[1] + j * p2[1]

            triangular_lattice[j * Nx + i, 0] = originx
            triangular_lattice[j * Nx + i, 1] = originy

    return triangular_lattice


def generate_sublattices_square(Lx, Ly):
    A_coords = []
    B_coords = []
    A_sites = []
    B_sites = []
    coord_fn = lambda x, y: coord_to_site_bravais(Lx, x, y)

    for nx in range(Lx):
        for ny in range(Ly):
            if nx % 2 == 0:
                if ny % 2 == 0:
                    A_coords.append((nx, ny))
                    A_sites.append(coord_fn(nx, ny))
                else:
                    B_coords.append((nx, ny))
                    B_sites.append(coord_fn(nx, ny))
            else:
                if ny % 2 == 0:
                    B_coords.append((nx, ny))
                    B_sites.append(coord_fn(nx, ny))
                else:
                    A_coords.append((nx, ny))
                    A_sites.append(coord_fn(nx, ny))

    return A_coords, B_coords, A_sites, B_sites

def reorder_interaction(_i_,_j_,sublattice_assignments,reorder=False):
    sublattice_i = sublattice_assignments[_i_]
    sublattice_j = sublattice_assignments[_j_]
    if reorder:
        if sublattice_j == (sublattice_i + 1) % 3:
            interaction_i = _i_
            interaction_j = _j_
        elif sublattice_j == sublattice_i:
            interaction_i = _i_
            interaction_j = _j_
        else:
            interaction_i = _j_
            interaction_j = _i_
    else:
        interaction_i = _i_
        interaction_j = _j_

    return interaction_i, interaction_j

def generate_sublattices_triangular(Lx, Ly):
    A_sites = []
    B_sites = []
    C_sites = []
    all_assignments = []
    coord_fn = lambda x, y: coord_to_site_bravais(Lx, x, y)

    for nx in range(Lx):
        for ny in range(Ly):
            if nx % 3 == 0:
                if ny % 3 == 0:
                    sublattice = 0
                    A_sites.append(coord_fn(nx, ny))
                elif ny % 3 == 1:
                    sublattice = 1
                    C_sites.append(coord_fn(nx, ny))
                else:
                    sublattice = 2
                    B_sites.append(coord_fn(nx, ny))
            elif nx % 3 == 1:
                if ny % 3 == 0:
                    sublattice = 1
                    B_sites.append(coord_fn(nx, ny))
                elif ny % 3 == 1:
                    sublattice = 2
                    A_sites.append(coord_fn(nx, ny))
                else:
                    sublattice = 0
                    C_sites.append(coord_fn(nx, ny))
            else:
                if ny % 3 == 0:
                    sublattice = 2
                    C_sites.append(coord_fn(nx, ny))
                elif ny % 3 == 1:
                    sublattice = 0
                    B_sites.append(coord_fn(nx, ny))
                else:
                    sublattice = 1
                    A_sites.append(coord_fn(nx, ny))
            all_assignments.append(sublattice)

    return A_sites, B_sites, C_sites, all_assignments


def buildlattice_triangular(Lx, Ly, bc="open", reorder=False):
    assert Lx == Ly, 'Lx must be equal to Ly'
    assert bc in ['open', 'periodic']
    square_interactions = []
    diagonal_interactions = []
    all_interactions = []
    coord_fn = lambda x, y: coord_to_site_bravais(Lx, x, y)

    _, _, _, sublattices = generate_sublattices_triangular(Lx, Ly)

    for n in range(Lx - 1):
        for n_ in range(Ly):
            # horizontal square lattice interactions (excluding boundary terms)
            site_i = coord_fn(n, n_)
            site_j = coord_fn(n + 1, n_)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            square_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

    for n in range(Lx):
        for n_ in range(Ly - 1):
            # vertical square lattice interactions (excluding boundary terms)
            site_i = coord_fn(n, n_)
            site_j = coord_fn(n, n_ + 1)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            square_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

    # diagonals
    for n in range(Lx - 1):
        for n_ in range(Ly - 1):
            site_i = coord_fn(n, n_)
            site_j = coord_fn(n + 1, n_ + 1)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            diagonal_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

    if bc == "periodic":
        for n in range(Lx):
            site_i = coord_fn(n, Ly - 1)
            site_j = coord_fn(n, 0)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            square_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

        for n_ in range(Ly):
            site_i = coord_fn(Lx - 1, n_)
            site_j = coord_fn(0, n_)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            square_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

        # diag across y boundary
        for n in range(1, Lx):
            site_i = coord_fn(n, 0)
            site_j = coord_fn(n - 1, Ly - 1)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            diagonal_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

        # diag across x boundary
        for n_ in range(Ly - 1):
            site_i = coord_fn(Lx - 1, n_)
            site_j = coord_fn(0, n_ + 1)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            diagonal_interactions.append((int_i, int_j))
            all_interactions.append((int_i, int_j))

        # corner interaction
        site_i = coord_fn(Lx - 1, Ly - 1)
        site_j = coord_fn(0, 0)
        int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
        diagonal_interactions.append((int_i, int_j))
        all_interactions.append((int_i, int_j))

    return square_interactions, diagonal_interactions, all_interactions

def buildlattice_alltoall(L, reorder=False):
    same_sublattice = []
    diff_sublattice = []
    all_interactions = []
    N_spins = L ** 2
    _, _, _, sublattices = generate_sublattices_triangular(L, L)

    for i in range(N_spins):
        for j in range(i, N_spins):
            xi = i % L
            yi = i // L
            xj = j % L
            yj = j // L
            site_i = coord_to_site_bravais(L, xi, yi)
            site_j = coord_to_site_bravais(L, xj, yj)
            sublattice_i = sublattices[site_i]
            sublattice_j = sublattices[site_j]
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            interaction = [int_i,int_j]
            if sublattice_i == sublattice_j:
                same_sublattice.append(interaction)
            else:
                diff_sublattice.append(interaction)
            all_interactions.append(interaction)
    return same_sublattice,diff_sublattice,all_interactions

def get_norm(vec):
    return vec[0]**2 + vec[1]**2

def buildlattice_alltoall_primitive_vector(L: int, p1=default_p1, p2=default_p2, periodic=False, reorder=False):
    interactions = {}
    N_spins = L ** 2
    _, _, _, sublattices = generate_sublattices_triangular(L, L)

    for i in range(N_spins):
        for j in range(i, N_spins):
            xi_square = i % L
            yi_square = i // L

            xj_square = j % L
            yj_square = j // L

            site_i = coord_to_site_bravais(L, xi_square, yi_square)
            site_j = coord_to_site_bravais(L, xj_square, yj_square)
            int_i, int_j = reorder_interaction(site_i, site_j, sublattices,reorder=reorder)
            interaction = (int_i,int_j)

            delta_x_square = xj_square - xi_square
            delta_y_square = yj_square - yi_square

            if periodic:
                delta_y_square_peri = -1 * (L - delta_y_square)
                delta_x_square_peri = -1 * (L - delta_x_square)
                r_no_peri = (delta_x_square * p1[0] + delta_y_square * p2[0],
                             delta_y_square * p2[1])
                r_x_peri = (delta_x_square_peri * p1[0] + delta_y_square * p2[0],
                             delta_y_square * p2[1])
                r_y_peri = (delta_x_square * p1[0] + delta_y_square_peri * p2[0],
                             delta_y_square_peri * p2[1])
                r_both_peri = (delta_x_square_peri * p1[0] + delta_y_square_peri * p2[0],
                             delta_y_square_peri * p2[1])
                r_vecs = np.array([r_no_peri,r_x_peri,r_y_peri,r_both_peri])
                r_norms = np.array([get_norm(r_no_peri),get_norm(r_x_peri),get_norm(r_y_peri),get_norm(r_both_peri)])
                idx = np.where(r_norms==min(r_norms))
                interactions[interaction] = r_vecs[idx][0]
            else:
                delta_x = delta_x_square * p1[0] + delta_y_square * p2[0]
                delta_y = delta_x_square * p1[1] + delta_y_square * p2[1]
                interactions[interaction] = (delta_x, delta_y)

    return interactions

def get_all_longest_r_interactions_triangular(L):
    all_interactions = buildlattice_alltoall_primitive_vector(L, default_p1, default_p2, periodic=True)
    longest_x = (default_p1[0] + default_p2[0]) * int(L/2)
    longest_y = (default_p1[1] + default_p2[1]) * int(L/2)
    longest_r_interactions = []
    for idx, r in all_interactions.items():
        if (abs(r[0]) == longest_x) & (abs(r[1]) == longest_y):
            longest_r_interactions.append(idx)

    return longest_r_interactions


def get_batched_interactions_Jmats(L, interactions, interactions_batch_size, tf_dtype):
    num_batches = len(interactions) // interactions_batch_size
    J_matrix_list = {}
    interactions_list = {}

    for batch in range(num_batches):
        start = batch * interactions_batch_size
        stop = (batch + 1) * interactions_batch_size
        interactions_batch = interactions[start:stop]
        J_mat = np.zeros((len(interactions_batch), L ** 2))
        for n, interaction in enumerate(interactions_batch):
            i, j = interaction
            J_mat[n, i] += 1
            J_mat[n, j] += 1
        J_matrix = tf.constant(J_mat, dtype=tf_dtype)
        J_matrix_list[batch] = J_matrix
        interactions_list[batch] = interactions_batch

    if num_batches * interactions_batch_size != len(interactions):
        start = num_batches * interactions_batch_size
        interactions_batch = interactions[start:]
        J_mat = np.zeros((len(interactions_batch), L ** 2))
        for n, interaction in enumerate(interactions_batch):
            i, j = interaction
            J_mat[n, i] += 1
            J_mat[n, j] += 1
        J_matrix = tf.constant(J_mat, dtype=tf_dtype)
        J_matrix_list[num_batches] = J_matrix
        interactions_list[num_batches] = interactions_batch

    return J_matrix_list, interactions_list
