import numpy as np
import tensorflow as tf

from interactions import site_to_coord_bravais, coord_to_site_bravais

def get_C6v_square(N_spins): 
    """
    Apply discrete lattice symmetries to a set of samples.
    This function was meant for samples on a square lattice.
    Has not been generalized.

    Args:
        samples: A Tensor of shape (num_samples, Nx * Ny) with elements ranging from 1 to
            the size  of the local_hilbert_space.
        lattice: String specifying the lattice.

    Returns:
        A Tensor of shape (num_symmetries* num_samples, Nx, Ny, 1) and
        an integer specifying the number of symmetries

    """
    L = int(np.sqrt(N_spins))
    indices = [list(site_to_coord_bravais(L, site) for site in range(N_spins))]
    # if L is odd, center at midpoint
    # if L % 2:
    center = (L // 2, L // 2)
    for i in range(5):
        indices.append(index_map_rot_60(indices[i], L, center))
    indices.append([(y, x) for x, y in indices[0]])
    for i in range(5):
        indices.append(index_map_rot_60(indices[6 + i], L, center))
    site_indices = []

    for _ind in indices:
        site_indices.append(tf.constant([coord_to_site_bravais(L, x, y) for (x, y) in _ind]))

    def apply_symmetries(samples, spin_parity: bool):  # correct
        """
        Apply discrete lattice symmetries to a set of samples

        Args:
            samples: A Tensor of shape (num_samples, self.Nx, self.Ny) with elements ranging from 1 to
                the size  of the local_hilbert_space.
            lattice: String specifying the lattice.

        Returns:
            A Tensor of shape (num_symmetries* num_samples, self.Nx, self.Ny, 1) and
            an integer specifying the number of symmetries

        """
        all_symmetrized_samples = []
        all_symmetrized_samples.append(samples)
        num_symmetries = 12
        for _i in range(1, num_symmetries):
            all_symmetrized_samples.append(tf.gather(samples, site_indices[_i], axis=1))
        if spin_parity:
            num_tot_symmetries = num_symmetries * 2
            for _j in range(num_symmetries):
                all_symmetrized_samples.append(tf.abs(1 - all_symmetrized_samples[_j]))
        else:
            num_tot_symmetries = num_symmetries
        return tf.stack(all_symmetrized_samples, axis=0), num_tot_symmetries

    return apply_symmetries


def get_C2v_square(N_spins):
    L = int(np.sqrt(N_spins))

    def apply_symmetries(samples, spin_parity):
        samples = tf.reshape(samples, (samples.shape[0], L, L, 1))
        all_symmetrized_samples = []
        all_symmetrized_samples.append(samples)
        num_symmetries = 4
        num_tot_symmetries = 4
        rot180 = tf.image.rot90(samples, k=-2)
        all_symmetrized_samples.append(rot180)
        flip_d = tf.transpose(samples, perm=[0, 2, 1, 3])
        all_symmetrized_samples.append(flip_d)
        flip_offd = tf.transpose(a=rot180, perm=[0, 2, 1, 3])
        all_symmetrized_samples.append(flip_offd)

        if spin_parity:
            num_tot_symmetries = num_symmetries * 2

            all_symmetrized_samples.append(tf.abs(1 - samples))
            all_symmetrized_samples.append(tf.abs(1 - rot180))
            all_symmetrized_samples.append(tf.abs(1 - flip_d))
            all_symmetrized_samples.append(tf.abs(1 - flip_offd))

        return tf.stack(all_symmetrized_samples, axis=0), num_tot_symmetries

    return apply_symmetries


def index_map_rot_60(indices, L, center=(0, 0)):
    # (i,j) -> (i - j, i)
    new_indices = []
    for (i, j) in indices:
        _i = i - center[0]
        _j = j - center[1]
        new_indices.append((int((_i - _j + center[0]) % L), int((_i + center[1]))))

    return new_indices
