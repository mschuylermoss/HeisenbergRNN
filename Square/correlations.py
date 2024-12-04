import numpy as np
import tensorflow as tf

from interactions import coord_to_site_bravais, buildlattice_alltoall_primitive_vector, buildlattice_alltoall, generate_sublattices_square

default_p1 = (1,0)
default_p2 = (0,1)
default_kx = np.pi
default_ky = np.pi

def get_longest_r_interactions_square(L):
    all_interactions = buildlattice_alltoall_primitive_vector(L,p1=default_p1,p2=default_p2,periodic=True)
    longest_r_interactions = []
    for idx, r in all_interactions.items():
        if (abs(r[0]) == L/2) & (abs(r[1]) == L/2):
            longest_r_interactions.append(idx)
    return longest_r_interactions


def calculate_all_kspace_correlations(L: int, Sij: np.ndarray, kx_mesh: np.ndarray, ky_mesh: np.ndarray, 
                                      p1=default_p1, p2=default_p2,
                                      snake=False,
                                      periodic=False):
    Sk = np.zeros((len(kx_mesh), len(ky_mesh)), dtype=complex)
    interactions_r = buildlattice_alltoall_primitive_vector(L, p1, p2, snake, periodic)
    for idx, r in interactions_r.items():
        Sk += np.exp(1j * (r[0] * kx_mesh[:, np.newaxis] + r[1] * ky_mesh[np.newaxis, :])) * 0.25 * Sij[idx]
        if idx[0] != idx[1]:
            Sk += np.exp(-1j * (r[0] * kx_mesh[:, np.newaxis] + r[1] * ky_mesh[np.newaxis, :])) * 0.25 * Sij[idx]
    Sk /= L ** 2
    return np.real(Sk)


def calculate_structure_factor(L: int, Sij: np.ndarray, 
                               kx=default_kx, ky=default_ky, 
                               p1=default_p1, p2=default_p2, 
                               var_Sij=None, 
                               snake=False,
                               periodic=False):
    interactions_r = buildlattice_alltoall_primitive_vector(L, p1, p2, snake, periodic)
    Sk = 0
    Sk_var = 0
    for idx, r in interactions_r.items():
        Sk += np.exp(1j * (r[0] * kx + r[1] * ky)) * 0.25 * Sij[idx]
        if idx[0] != idx[1]:
            Sk += np.exp(-1j * (r[0] * kx + r[1] * ky)) * 0.25 * Sij[idx]
        if var_Sij is not None:
            Sk_var += (np.exp(-1j * (r[0] * kx + r[1] * ky)) * 0.25) ** 2 * var_Sij[idx]
            if idx[0] != idx[1]:
                Sk_var += (np.exp(-1j * (r[0] * kx + r[1] * ky)) * 0.25) ** 2 * var_Sij[idx]
    Sk /= L ** 2
    Sk_err = np.sqrt(Sk_var) / L ** 2
    return np.real(Sk), np.real(Sk_err)

def calculate_expectation_ft_Si_square(L:int, 
                                       Sis:np.ndarray, 
                                       snake=False):
    _, _, A_sites, B_sites = generate_sublattices_square(L,L,snake=snake)
    N = L**2
    ft_factors = []
    for n in range(N):
        in_A = (n in A_sites)
        ft_factor = 2*in_A - 1
        ft_factors.append(ft_factor)
    ft_Si = np.sum(0.5 * (Sis * ft_factors), axis=1)
    mean_ft_Si = np.mean(abs(ft_Si)**2)
    var_ft_Si = np.var(abs(ft_Si)**2)
    return mean_ft_Si, var_ft_Si

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

def get_Si(log_fxn,tf_dtype=tf.float32):

    def Sxyi_vectorized(samples,og_amps):
        N = tf.shape(samples)[1]
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], N, axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.eye(N,dtype=tf_dtype)[tf.newaxis, :, :], 2)
        subtract = samples_tiled_not_flipped - samples_tiled_flipped
        signs = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract, axis=1))
        _, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N))) # (Ns*N,N)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, N)) - og_amps[:, tf.newaxis]) # (Ns, N)
        Si_x = amp_ratio
        Si_y = signs * amp_ratio
        local_Sis = Si_x + Si_y

        return local_Sis # (Ns,)

    def Szi_vectorized(samples):

        N = tf.shape(samples)[1]
        Si_z_real = tf.eye(N,dtype=tf_dtype) @ (2 * tf.transpose(samples) - 1)
        Si_z = tf.complex(Si_z_real,tf.cast(0,dtype=tf_dtype))
        local_Sis = tf.transpose(Si_z)
    
        return local_Sis # (Ns,)

    return Sxyi_vectorized,Szi_vectorized

def get_Heisenberg_realspace_Correlation_Vectorized(log_fxn, tf_dtype=tf.float32):
    @tf.function()
    def Heisenberg_realspace_Correlation_Vectorized_xx_yy(samples, og_amps, J_matrix):
        N = tf.shape(samples)[1]
        num_interactions = tf.shape(J_matrix)[0]
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], num_interactions, axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
        samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
        signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        _, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)))
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, num_interactions)) - og_amps[:, tf.newaxis])
        xx = amp_ratio
        yy = -1 * signs * amp_ratio
        return xx + yy

    @tf.function()
    def Heisenberg_realspace_Correlation_Vectorized_zz(samples, J_matrix):
        zz = tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1
        zz = tf.complex(tf.transpose(zz), tf.cast(0.0, dtype=tf_dtype))
        return zz

    return Heisenberg_realspace_Correlation_Vectorized_xx_yy, Heisenberg_realspace_Correlation_Vectorized_zz


def undo_marshall_sign(L):
    N = L**2
    _, _, A_sites, B_sites = generate_sublattices_square(L,L)
    interactions = np.array(buildlattice_alltoall(L))
    minus_signs_matrix = np.ones((N,N))
    for interaction in interactions:
        s1, s2 = interaction
        no_minus = ((s1 in A_sites) ^ (s2 in B_sites)) or ((s1 in B_sites) ^ (s2 in A_sites))
        minus_signs_matrix[s1,s2] = 2 * no_minus - 1
    return minus_signs_matrix