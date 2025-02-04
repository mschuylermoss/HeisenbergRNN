import numpy as np
import tensorflow as tf

from interactions import coord_to_site_bravais
from interactions import buildlattice_alltoall, buildlattice_alltoall_primitive_vector
from interactions import generate_sublattices_square

default_p1 = (1.,0.)
default_p2 = (-1/2,np.sqrt(3)/2)

default_kx = 2*np.pi/3
default_ky = 2*np.pi/np.sqrt(3)


def calculate_all_kspace_correlations(L: int, 
                                  Sij: np.ndarray, 
                                  kx_mesh: np.ndarray, ky_mesh: np.ndarray, 
                                  p1=default_p1, p2=default_p2,
                                  snake=False,
                                  periodic=False):
    Sk = np.zeros((len(kx_mesh), len(ky_mesh)), dtype=complex)
    interactions_r = buildlattice_alltoall_primitive_vector(L, p1, p2, snake, periodic)

    for idx, r in interactions_r.items():
        Sk += np.exp(1j * (r[0] * kx_mesh[:, np.newaxis] + r[1] * ky_mesh[np.newaxis, :])) * Sij[idx]
        if idx[0] != idx[1]:
            Sk += np.exp(-1j * (r[0] * kx_mesh[:, np.newaxis] + r[1] * ky_mesh[np.newaxis, :])) * Sij[idx]
    Sk /= L ** 2
    return np.real(Sk)

def calculate_structure_factor(L: int, 
                               Sij: np.ndarray, 
                               kx=default_kx, ky=default_ky, 
                               p1=default_p1, p2=default_p2, 
                               var_Sij=None, 
                               snake=False,
                               periodic=False):
    interactions_r = buildlattice_alltoall_primitive_vector(L, p1, p2, snake, periodic)

    Sk = 0
    Sk_var = 0
    for idx, r in interactions_r.items():
        Sk += np.exp(1j * (r[0] * kx + r[1] * ky)) * Sij[idx]
        if idx[0] != idx[1]:
            Sk += np.exp(-1j * (r[0] * kx + r[1] * ky)) * Sij[idx]
        if var_Sij is not None:
            Sk_var += np.exp(1j * (r[0] * kx + r[1] * ky))*np.conj(np.exp(1j * (r[0] * kx + r[1] * ky))) * var_Sij[idx]
            if idx[0] != idx[1]:
                Sk_var += np.exp(1j * (r[0] * kx + r[1] * ky))*np.conj(np.exp(1j * (r[0] * kx + r[1] * ky))) * var_Sij[idx]
    Sk /= L ** 2
    Sk_var /= L ** 4
    return np.real(Sk), np.real(Sk_var)


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
    print("getting correlation function!")

    # @tf.function()
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

        # xx = tf.reduce_sum(amp_ratio, axis=0)
        xx = amp_ratio
        # yy = -tf.reduce_sum(signs * amp_ratio, axis=0)
        yy = -1 * signs * amp_ratio
        # zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=1)
        return 0.25 * (xx + yy)

    # @tf.function()
    def Heisenberg_realspace_Correlation_Vectorized_zz(samples, J_matrix):
        # zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=1)
        zz = tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1
        zz = tf.complex(tf.transpose(zz), tf.cast(0.0, dtype=tf_dtype))

        return 0.25 * zz

    return Heisenberg_realspace_Correlation_Vectorized_xx_yy, Heisenberg_realspace_Correlation_Vectorized_zz


def get_Heisenberg_realspace_Correlation_Vectorized_TriMS(log_fxn, tf_dtype=tf.float32):
    print("getting triMS correlation function!")

    xx_yy_coeff = tf.complex(-0.5, tf.cast(0., dtype=tf_dtype))
    yx_xy_coeff = tf.complex(tf.cast(np.sqrt(3) / 2, dtype=tf_dtype), tf.cast(0., dtype=tf_dtype))

    # @tf.function()
    def Heisenberg_Correlation_Vectorized_diag(samples, J_matrix):

        zz = tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1
        zz = tf.complex(tf.transpose(zz), tf.cast(0.0, dtype=tf_dtype))

        return 0.25 * zz
    
    # @tf.function()
    def Heisenberg_Correlation_Vectorized_offdiag_diff(samples, log_amps, J_matrix, J_matrix_is, J_matrix_js):
        N = tf.shape(samples)[1]

        samples_tiled = tf.repeat(samples[:, :, tf.newaxis], len(J_matrix), axis=2)

        samples_tiled_both_flipped = tf.math.mod(samples_tiled + tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
        subtract = samples_tiled_both_flipped - samples_tiled
        signs = tf.complex(tf.math.abs(tf.reduce_sum(subtract, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_both_flipped = tf.transpose(samples_tiled_both_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_both_flipped, (-1, N)))
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(J_matrix))) - log_amps[:, tf.newaxis])
        xx = amp_ratio
        yy = -signs * amp_ratio

        samples_tiled_is_flipped = tf.math.mod(samples_tiled + tf.transpose(J_matrix_is)[tf.newaxis, :, :], 2)
        subtract_is = samples_tiled - samples_tiled_is_flipped
        signs_is = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract_is, axis=1))
        yx = signs_is * amp_ratio

        samples_tiled_js_flipped = tf.math.mod(samples_tiled + tf.transpose(J_matrix_js)[tf.newaxis, :, :], 2)
        subtract_js = samples_tiled - samples_tiled_js_flipped
        signs_js = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract_js, axis=1))
        xy = signs_js * amp_ratio

        xx_yy = xx_yy_coeff * (xx + yy)
        yx_xy = yx_xy_coeff * (yx - xy)
        return 0.25 * (xx_yy + yx_xy)

    # @tf.function()
    def Heisenberg_Correlation_Vectorized_offdiag_same(samples, og_amps, J_matrix):
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

        return 0.25 * (xx + yy)

    return Heisenberg_Correlation_Vectorized_offdiag_same, Heisenberg_Correlation_Vectorized_offdiag_diff, Heisenberg_Correlation_Vectorized_diag


def undo_marshall_sign(L):
    N = L ** 2
    _, _, A_sites, B_sites = generate_sublattices_square(L, L)
    interactions = np.array(buildlattice_alltoall(L))
    minus_signs_matrix = np.ones((N, N))
    for interaction in interactions:
        s1, s2 = interaction
        no_minus = ((s1 in A_sites) ^ (s2 in B_sites)) or ((s1 in B_sites) ^ (s2 in A_sites))
        minus_signs_matrix[s1, s2] = 2 * no_minus - 1
    return minus_signs_matrix
