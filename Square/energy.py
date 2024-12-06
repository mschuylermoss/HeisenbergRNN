import numpy as np
import tensorflow as tf


def get_Heisenberg_Energy_Vectorized_square(J, interactions, log_fxn, marshall_sign=False, symmetrize=False,
                                            parity=False, tf_dtype=tf.float32):
    print("getting energy function!")

    if marshall_sign:
        ms = -1
        print("Marshall Sign applied!")
    else:
        ms = 1
        print("Marshall Sign not applied!")

    np_interactions = np.array(interactions)
    Nspins = max(max(np_interactions[:, 0]), max(np_interactions[:, 1])) + 1
    J_mat_s = np.zeros((len(interactions), Nspins))
    for n, interaction in enumerate(interactions):
        i, j = interaction
        J_mat_s[n, i] += -1
        J_mat_s[n, j] += -1

    J_matrix = tf.constant(J_mat_s, dtype=tf_dtype)
    J_tf = tf.complex(tf.cast(J, dtype=tf_dtype), tf.cast(0.0, dtype=tf_dtype))

    @tf.function()
    def Heisenberg_Energy_Vectorized_tf_function_square(samples, og_amps):
        N = tf.shape(samples)[1]

        Energies_zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
        Energies_zz = tf.complex(Energies_zz, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(interactions), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
        samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
        signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, tf.cast(0.0, dtype=tf_dtype))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)), symmetrize=symmetrize,
                                            parity=parity)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interactions))) - og_amps[:, tf.newaxis])
        Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
        Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)
        Energies = 0.25 * J_tf * (ms * Energies_xx + ms * Energies_yy + Energies_zz)

        return Energies

    return Heisenberg_Energy_Vectorized_tf_function_square
