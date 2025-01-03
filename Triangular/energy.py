import numpy as np
import tensorflow as tf


def get_Heisenberg_Energy_triangular_TriMS(J, interaction_list_is, interaction_list_js, log_fxn,
                                             marshall_sign=False, symmetrize=False, parity=False,
                                             tf_dtype=tf.float32):
    print("getting energy function!")

    N_spins = max(max(interaction_list_is), max(interaction_list_js)) + 1
    J_matrix_is_np = np.zeros((len(interaction_list_is), N_spins))
    J_matrix_js_np = np.zeros((len(interaction_list_is), N_spins))
    J_matrix_np = np.zeros((len(interaction_list_is), N_spins))
    for n, _ in enumerate(interaction_list_is):
        i = interaction_list_is[n]
        J_matrix_is_np[n, i] += 1
        J_matrix_np[n, i] += 1
        j = interaction_list_js[n]
        J_matrix_js_np[n, j] += 1
        J_matrix_np[n, j] += 1

    J_matrix_is = tf.constant(J_matrix_is_np, dtype=tf_dtype)
    J_matrix_js = tf.constant(J_matrix_js_np, dtype=tf_dtype)
    J_matrix = tf.constant(J_matrix_np, dtype=tf_dtype)
    xx_yy = tf.complex(-0.5, tf.cast(0., dtype=tf_dtype))
    yx_xy = tf.complex(tf.cast(np.sqrt(3) / 2, dtype=tf_dtype), tf.cast(0., dtype=tf_dtype))
    J_tf = tf.cast(J, dtype=tf_dtype)

    # @tf.function()
    def Heisenberg_Energy_Vectorized_tf_function_MS(samples, log_amps):
        N = tf.shape(samples)[1]

        Energies_zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
        Energies_zz = tf.complex(Energies_zz, tf.cast(0.0, dtype=tf_dtype))

        samples_tiled = tf.repeat(samples[:, :, tf.newaxis], len(interaction_list_is), axis=2)

        samples_tiled_both_flipped = tf.math.mod(samples_tiled + tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
        subtract = samples_tiled_both_flipped - samples_tiled
        signs = tf.complex(tf.math.abs(tf.reduce_sum(subtract, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_both_flipped = tf.transpose(samples_tiled_both_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_both_flipped, (-1, N)), symmetrize=symmetrize,
                                            parity=parity)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interaction_list_is))) - log_amps[:, tf.newaxis])
        Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
        Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)

        samples_tiled_is_flipped = tf.math.mod(samples_tiled + tf.transpose(J_matrix_is)[tf.newaxis, :, :], 2)
        subtract_is = samples_tiled - samples_tiled_is_flipped
        signs_is = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract_is, axis=1))
        Energies_yx = tf.reduce_sum(signs_is * amp_ratio, axis=1)

        samples_tiled_js_flipped = tf.math.mod(samples_tiled + tf.transpose(J_matrix_js)[tf.newaxis, :, :], 2)
        subtract_js = samples_tiled - samples_tiled_js_flipped
        signs_js = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract_js, axis=1))
        Energies_xy = tf.reduce_sum(signs_js * amp_ratio, axis=1)

        Energies_xx_yy = xx_yy * (Energies_xx + Energies_yy)
        Energies_yx_xy = yx_xy * (Energies_yx - Energies_xy)
        Energies = tf.complex(0.25 * J_tf, tf.cast(0., dtype=tf_dtype)) * (Energies_zz + Energies_xx_yy + Energies_yx_xy)

        return Energies

    # @tf.function()
    def Heisenberg_Energy_Vectorized_tf_function_noMS(samples, log_amps):
        N = tf.shape(samples)[1]

        Energies_zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
        Energies_zz = tf.complex(Energies_zz, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(interaction_list_is), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
        samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
        signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)), symmetrize=symmetrize,
                                            parity=parity)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interaction_list_is))) - log_amps[:, tf.newaxis])
        Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
        Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)
        Energies = tf.complex(0.25 * J_tf, tf.cast(0., dtype=tf_dtype)) * (Energies_xx + Energies_yy + Energies_zz)

        return Energies

    if marshall_sign:
        print("Marshall Sign applied!")
        return Heisenberg_Energy_Vectorized_tf_function_MS
    else:
        print("Marshall Sign not applied!")
        return Heisenberg_Energy_Vectorized_tf_function_noMS


def get_Heisenberg_Energy_triangular_SquareMS(J, interactions_s, interactions_diags, N_spins, log_fxn,
                                                marshall_sign=False, symmetrize=False, parity=False,
                                                tf_dtype=tf.float32):
    print("getting energy function!")
    if marshall_sign:
        ms = -1
        print("Marshall Sign applied!")
    else:
        ms = 1
        print("Marshall Sign not applied!")
    
    J_mat_s = np.zeros((len(interactions_s), N_spins))
    for n, interaction in enumerate(interactions_s):
        i, j = interaction
        J_mat_s[n, i] += -1
        J_mat_s[n, j] += -1

    J_mat_d = np.zeros((len(interactions_diags), N_spins))
    for n, interaction in enumerate(interactions_diags):
        i, j = interaction
        J_mat_d[n, i] += 1
        J_mat_d[n, j] += 1

    J_mat_s_tf = tf.constant(J_mat_s, dtype=tf_dtype)
    J_mat_d_tf = tf.constant(J_mat_d, dtype=tf_dtype)
    J_tf = tf.complex(tf.cast(J, dtype=tf_dtype), tf.cast(0.0, dtype=tf_dtype))

    # @tf.function()
    def Heisenberg_Energy_Vectorized_tf_function_square(samples, og_amps):
        N = tf.shape(samples)[1]
        if J_tf.dtype.name == 'complex64':
            tf_dtype = tf.float32
        else:
            tf_dtype = tf.float64

        Energies_zz = tf.reduce_sum(tf.math.abs(J_mat_s_tf @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
        Energies_zz = tf.complex(Energies_zz, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(interactions_s), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(J_mat_s_tf)[tf.newaxis, :, :], 2)
        samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
        signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)), symmetrize=symmetrize,
                                            parity=parity)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interactions_s))) - og_amps[:, tf.newaxis])
        Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
        Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)
        Energies = 0.25 * J_tf * (ms * Energies_xx + ms * Energies_yy + Energies_zz)

        return Energies

    # @tf.function()
    def Heisenberg_Energy_Vectorized_tf_function_diags(samples, og_amps):
        N = tf.shape(samples)[1]
        if J_tf.dtype.name == 'complex64':
            tf_dtype = tf.float32
        else:
            tf_dtype = tf.float64
        Energies_zz = tf.reduce_sum(tf.math.abs(J_mat_d_tf @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
        Energies_zz = tf.complex(Energies_zz, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(interactions_diags), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(J_mat_d_tf)[tf.newaxis, :, :], 2)
        samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
        signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)), symmetrize=symmetrize,
                                            parity=parity)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interactions_diags))) - og_amps[:, tf.newaxis])
        Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
        Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)
        Energies = 0.25 * J_tf * (Energies_xx + Energies_yy + Energies_zz)

        return Energies

    return Heisenberg_Energy_Vectorized_tf_function_square, Heisenberg_Energy_Vectorized_tf_function_diags


