import numpy as np
import tensorflow as tf


def get_C4v_square(N_spins):
    L = int(np.sqrt(N_spins))

    def apply_symmetries(samples, spin_parity):
        samples = tf.reshape(samples, (samples.shape[0], L, L, 1))
        all_symmetrized_samples = []
        all_symmetrized_samples.append(samples)
        num_symmetries = 8
        num_tot_symmetries = 8
        rot90 = tf.image.rot90(samples, k=1)
        all_symmetrized_samples.append(rot90)
        rot180 = tf.image.rot90(samples, k=2)
        all_symmetrized_samples.append(rot180)
        rot270 = tf.image.rot90(samples, k=3)
        all_symmetrized_samples.append(rot270)
        flip_h = tf.image.flip_left_right(samples)
        all_symmetrized_samples.append(flip_h)
        flip_v = tf.image.flip_up_down(samples)
        all_symmetrized_samples.append(flip_v)
        flip_d = tf.transpose(samples, perm=[0, 2, 1, 3])
        all_symmetrized_samples.append(flip_d)
        flip_offd = tf.transpose(rot180, perm=[0, 2, 1, 3])
        all_symmetrized_samples.append(flip_offd)

        if spin_parity:
            num_tot_symmetries = num_symmetries * 2
            all_symmetrized_samples.append(tf.abs(1 - samples))
            all_symmetrized_samples.append(tf.abs(1 - rot90))
            all_symmetrized_samples.append(tf.abs(1 - rot180))
            all_symmetrized_samples.append(tf.abs(1 - rot270))
            all_symmetrized_samples.append(tf.abs(1 - flip_h))
            all_symmetrized_samples.append(tf.abs(1 - flip_v))
            all_symmetrized_samples.append(tf.abs(1 - flip_d))
            all_symmetrized_samples.append(tf.abs(1 - flip_offd))

        return tf.stack(all_symmetrized_samples, axis=0), num_tot_symmetries

    return apply_symmetries
