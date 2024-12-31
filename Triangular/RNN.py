import random
from abc import ABC

import numpy as np
import tensorflow as tf

from symmetries import get_C2v_square, get_C6v_square
from interactions import coord_to_site_bravais, generate_sublattices_triangular


def get_rnn_cell(name):
    if name == 'MDGRU':
        return MDGRU
    elif name == 'MDPeriodic':
        return MDPeriodic
    else:
        return NotImplementedError


class MDGRU(tf.keras.layers.SimpleRNNCell):
    """
    An implementation of a 2D tensorized GRU RNN cell
    """

    def __init__(self, num_units=None, local_hilbert_size=None, name=None, dtype=None):
        super(MDGRU, self).__init__(units=num_units, name=name)
        # save class variables
        self._input_size = local_hilbert_size
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = local_hilbert_size
        self._dtype = dtype
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                        mode="fan_avg",
                                                                        distribution="uniform")

    def build(self, input_shape):
        self.W = self.add_weight(name="W_" + self.name, shape=[self.units, 2 * self.units, 2 * self._input_size],
                                 initializer=self.kernel_initializer,
                                 dtype=self.dtype)

        self.b = self.add_weight(name="b_" + self.name, shape=[self.units],
                                 initializer=self.kernel_initializer,
                                 dtype=self.dtype)

        self.Wg = self.add_weight(name="Wg_" + self.name, shape=[self.units, 2 * self.units, 2 * self._input_size],
                                  initializer=self.kernel_initializer,
                                  dtype=self.dtype)

        self.bg = self.add_weight(name="bg_" + self.name, shape=[self.units],
                                  initializer=self.kernel_initializer,
                                  dtype=self.dtype)

        self.Wmerge = self.add_weight(name="Wmerge_" + self.name, shape=[2 * self.units, self.units],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype)

    def call(self, inputs, states, training=False):
        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat((states[0], states[1]), 1),
                                   tf.concat((inputs[0], inputs[1]), 1))

        # prepare input linear combination
        state_mul = tf.einsum('ijk,ajk -> ia', inputstate_mul, self.W)
        state_mulg = tf.einsum('ijk,ajk -> ia', inputstate_mul, self.Wg)

        u = tf.nn.sigmoid(state_mulg + self.bg)
        state_tilda = tf.nn.tanh(state_mul + self.b)  # [batch_sz, num_units] 

        new_state = u * state_tilda + (1. - u) * tf.matmul(tf.concat(states, 1), self.Wmerge)
        output = new_state
        return output, new_state


class MDPeriodic(tf.keras.layers.SimpleRNNCell):
    """
    An implementation of a 2D periodic GRU RNN cell
    """

    def __init__(self, num_units=None, local_hilbert_size=None, name=None, dtype=None):
        super(MDPeriodic, self).__init__(units=num_units, name=name)
        self._input_size = local_hilbert_size
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = local_hilbert_size
        self._dtype = dtype
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                        mode="fan_avg",
                                                                        distribution="uniform")

    def build(self, input_shape):
        self.W = self.add_weight(name="W_" + self.name, shape=[4 * (self.units + self._input_size), self.units],
                                 initializer=self.kernel_initializer,
                                 dtype=self.dtype)

        self.b = self.add_weight(name="b_" + self.name, shape=[self.units],
                                 initializer=self.kernel_initializer,
                                 dtype=self.dtype)

        self.Wg = self.add_weight(name="Wg_" + self.name, shape=[4 * (self.units + self._input_size), self.units],
                                  initializer=self.kernel_initializer,
                                  dtype=self.dtype)

        self.bg = self.add_weight(name="bg_" + self.name, shape=[self.units],
                                  initializer=self.kernel_initializer,
                                  dtype=self.dtype)

        self.Wmerge = self.add_weight(name="Wmerge_" + self.name, shape=[4 * self.units, self.units],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype)

    def call(self, inputs, states, training=False):
        state_mul = tf.matmul(
            tf.concat([inputs[0], inputs[1], inputs[2], inputs[3], states[0], states[1], states[2], states[3]], 1),
            self.W)  # [batch_sz, num_units]

        state_mulg = tf.matmul(
            tf.concat([inputs[0], inputs[1], inputs[2], inputs[3], states[0], states[1], states[2], states[3]], 1),
            self.Wg)  # [batch_sz, num_units]

        state_tilda = tf.nn.tanh(state_mul + self.b)  # [batch_sz, num_units]
        u = tf.nn.sigmoid(state_mulg + self.bg)

        new_state = u * state_tilda + (1. - u) * tf.matmul(tf.concat(states, 1), self.Wmerge)
        output = new_state
        return output, new_state


class RNNWavefunction(ABC):
    def __init__(self, local_hilbert_space: int, num_sites: int, boundary_condition='open',
                 tf_dtype=tf.float32):
        """
        Abstract RNNWavefunction class that contains reusable functions across all RNNs

        Args:
            local_hilbert_space: The local Hilbert space per RNN cell or "super site"
            num_sites: The number of sites in the lattice
            boundary condition: what kind of boundary conditions, determines which RNN subclass to use
            tf_dtype: tensorflow data type to be used in the RNN 

        """

        self.local_hilbert_space = local_hilbert_space
        self.N_spins = num_sites
        self.tf_dtype = tf_dtype
        if self.local_hilbert_space > 2:
            raise NotImplementedError
        assert boundary_condition in ["open",
                                      "periodic"], f'Lattice must be "open" or "periodic" received {boundary_condition}'

        if boundary_condition == "open":
            self.apply_symmetries = get_C2v_square(num_sites)
        else:
            self.apply_symmetries = get_C6v_square(num_sites)

        tf.keras.backend.set_floatx(tf_dtype.name)

    @staticmethod
    def heavyside(inputs):
        """
       Apply the heavyside function elementwise to the input.

       Args:
           inputs: A Tensor of any shape.

       Returns:
           A Tensor with the heavyside applied elementwise.
       """
        sign = tf.sign(tf.sign(inputs) + 0.1)
        return 0.5 * (sign + 1.0)

    @staticmethod
    def l1_normalize(inputs, axis=1):
        """
        Apply L1 normalization with respect over a certain axis.

        Args:
            inputs: A Tensor of any shape.
            axis: Axis over which to apply the normalization. Default axis is 1.

        Returns:
            A Tensor normalized along the specified axis.
        """
        sum_ = tf.reduce_sum(inputs, axis=axis)
        inv_sum = tf.math.divide(1, sum_ + 1e-5)
        norm_input = tf.math.multiply(inv_sum[:, tf.newaxis], inputs)
        return norm_input

    @staticmethod
    def regularized_identity(inputs, epsilon=1e-5):
        """
        Apply the regularized identity to a certain input.

        Args:
            inputs: A Tensor of any shape.
            epsilon: Regularization constant to make sure the square root is not negative. Default is 1e-5.

        Returns:
            A regularized tensor.
        """
        sign = tf.sign(tf.sign(inputs) + 0.1)  # tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
        return tf.stop_gradient(sign) * tf.sqrt(inputs ** 2 + epsilon ** 2)

    def mag_normalization(self, probs, num_up, num_generated_spins):
        """
        Enforce that the probabilities of creating samples with Mz != 0 are zero.

        Args:
            probs: probabilities that the spin is 0 or 1
            num_up: number of spins sampled 1 so far in the autoregressive sequence
            num_generated_spins: number of spens that have been sampled so far in the autoregressive sequence

        Returns:
            Tensor of size (num_samples, local_hilbert_space) with output probabilities reweighted if necessary
            such that Mz=0 is conserved.
        """
        num_samples = probs.shape[0]
        num_down = num_generated_spins - num_up
        activations_up = self.heavyside(((self.N_spins) // 2 - 1) - num_up)
        activations_down = self.heavyside(((self.N_spins) // 2 - 1) - num_down)
        probs = probs * tf.cast(tf.stack([activations_down, activations_up], axis=1), self.tf_dtype)
        probs = probs / (tf.reshape(tf.norm(tensor=probs, axis=1, ord=1), [num_samples, 1]))  # l1 normalizing

        return probs

    def average_symmetries(self, log_probs, total_phases, num_symmetries: int, num_samples: int):
        """
        Average the amplitudes over the different discrete symmetries.

        Args:
            log_probs: A Tensor of shape (num_symmetries*num_samples,) containing the log-probabilities of the samples.
            total_phases: A Tensor of shape (num_symmetries*num_samples,) containing the phases of the samples.
            num_samples: The number of samples drawn.

        Returns:
            A Tensor of shape (num_samples,) containing the averaged log-probs of the samples.
            A Tensor of shape (num_samples,) containing the averaged log-amplitudes of the samples.
        """

        log_probs = tf.reshape(log_probs, (num_symmetries, num_samples))
        log_num_symmetries = tf.math.log(tf.cast(num_symmetries, dtype=log_probs.dtype))
        log_probs = tf.math.reduce_logsumexp(log_probs - log_num_symmetries, axis=0)

        return log_probs, tf.complex(0.5 * log_probs, tf.reshape(total_phases, (num_symmetries, num_samples))[0])

    def softsign(self, inputs):
        """
        Apply the softsign function elementwise to the input.

        Args:
            inputs: A Tensor of any shape.

        Returns:
            A tensor with the softsign applied elementwise.
        """
        return tf.constant(np.pi, dtype=self.tf_dtype) * (tf.keras.activations.softsign(inputs))

    def sample(self, num_samples: int):
        """Should return a tensor of the form (Nsamples, Nspins)"""
        raise NotImplementedError

    def log_probsamps(self, samples, symmetrize: bool):
        """Should return a tensor of the form (Nsamples, )"""
        raise NotImplementedError


class cMDRNNWavefunction(RNNWavefunction):
    def __init__(self,
                 systemsize_x: int, systemsize_y: int,
                 units: int, local_hilbert_space: int = 2,
                 cell=MDGRU,
                 weight_sharing='all',
                 use_complex=False,
                 h_symmetries=True,
                 seed=111,
                 kernel_initializer='glorot_uniform',
                 tf_dtype=tf.float32):

        """
        systemsize_x, systemsize_y:  int
                     number of sites in x, y directions
        num_units:   int
                     number of memory num_units (length of hidden vector)
        local_hilbert_space: int
                     size of the local hilbet space
                     (related to the number of spins per site, which
                     is related to the lattice of interest)
        cell:        a tensorflow RNN cell
        activation:  activation of the RNN cell
        seed:        pseudo-random number generator
        h_symmetries: bool
                     determines whether HAMILTONIAN symmetries are
                     enforced during the sampling step
        l_symmetries: bool
                     determines whether LATTICE symmetries are
                     enforced during the sampling step
        kernel_initializer: str
                     indicates how the model is initialized
        """

        self.Nx = systemsize_x
        self.Ny = systemsize_y

        super().__init__(local_hilbert_space, self.Nx * self.Ny, 'open', tf_dtype)

        self.weight_sharing = weight_sharing
        assert self.weight_sharing in ['all', 'sublattice'], \
            'weight_sharing must be on of `all` or `sublattice`'
        self.use_complex = use_complex
        self.h_symmetries = h_symmetries
        self.log_cutoff = 1e-10

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        tf.random.set_seed(seed)  # tensorflow pseudo-random generator

        if self.weight_sharing == 'sublattice':
            A_lattice, B_lattice, C_lattice, _ = generate_sublattices_triangular(systemsize_x, systemsize_x)
            map_A = dict(zip(A_lattice, [0] * len(A_lattice)))
            map_B = dict(zip(B_lattice, [1] * len(B_lattice)))
            map_C = dict(zip(C_lattice, [2] * len(C_lattice)))

            self.lattice_map = map_A | map_B | map_C
            self.rnn = [
                cell(num_units=units, local_hilbert_size=local_hilbert_space, name=f"RNN_{0}_{i}",
                     dtype=self.tf_dtype) for i in range(3)]
            self.dense = [tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense_{i}', 
                                                dtype=self.tf_dtype, kernel_initializer=kernel_initializer) 
                                                for i in range(3)]
            if self.use_complex:
                self.dense_phase = [
                    tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense_phase_{i}', 
                                          dtype=self.tf_dtype, kernel_initializer=kernel_initializer)
                                          for i in range(3)]
        else:
            self.rnn = cell(num_units=units, local_hilbert_size=local_hilbert_space, name=f"RNN", dtype=self.tf_dtype)
            self.dense = tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense', dtype=self.tf_dtype,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=kernel_initializer)
            if self.use_complex:
                self.dense_phase = tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense_phase',
                                                        dtype=self.tf_dtype,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=kernel_initializer)

    def sample(self, num_samples):

        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            num_samples:     int
                             number of samples to be produced

            ------------------------------------------------------------------------
            Returns:        samples
                            tf.Tensor of shape (num_samples, Nx * Ny)
        """

        samples = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        rnn_states = {}
        inputs = {}
        for ny in range(-2, self.Ny):  # Loop over the number of sites
            for nx in range(-2, self.Nx + 2):
                inputs[f"{nx}{ny}"] = tf.zeros((num_samples, self.local_hilbert_space),
                                               dtype=self.tf_dtype)
                if self.weight_sharing=='sublattice':
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(num_samples)[0]
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(num_samples)[0]

                # Making a loop over the sites with the 2DRNN
        num_up = tf.zeros(num_samples, dtype=self.tf_dtype)
        num_generated_spins = 0
        for ny in range(self.Ny):

            if ny % 2 == 0:
                for nx in range(self.Nx):  # left to right

                    neighbor_inputs = [inputs[f"{nx - 1}{ny}"], inputs[f"{nx}{ny - 1}"]]
                    neighbor_states = [rnn_states[f"{nx - 1}{ny}"], rnn_states[f"{nx}{ny - 1}"]]

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(output_ampl)

                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    output_ampl_softmax = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)

                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output_ampl_softmax), num_samples=1),
                                             [-1, ])

                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.local_hilbert_space, dtype=self.tf_dtype)
                    samples[nx][ny] = sample_temp

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))

            if ny % 2 == 1:
                for nx in range(self.Nx - 1, -1, -1):  # right to left

                    neighbor_inputs = (inputs[f"{nx + 1}{ny}"], inputs[f"{nx}{ny - 1}"])
                    neighbor_states = (rnn_states[f"{nx + 1}{ny}"], rnn_states[f"{nx}{ny - 1}"])

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(output_ampl)

                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    output_ampl_softmax = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)
                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output_ampl_softmax), num_samples=1),
                                             [-1, ])

                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.local_hilbert_space, dtype=self.tf_dtype)
                    samples[nx][ny] = sample_temp

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))

        samples = tf.transpose(tf.stack(values=samples, axis=0), perm=[2, 0, 1])  # [num_samples, Ny, Nx]

        full_samples = tf.reshape(samples, (-1, self.Nx * self.Ny))
        full_samples_long = tf.cast(full_samples, dtype=self.tf_dtype)

        return full_samples_long

    def log_probsamps(self, samples, symmetrize: bool, parity: bool):
        """
        calculate the log-probabilities of ```samples``
        ------------------------------------------------------------------------
        Parameters:

        samples:         tf.Tensor
                         a tf.placeholder of shape (number of samples, Nx * Ny)
                         containing the input samples in integer encoding
        symmetrize:      bool
                         a boolean that tells us whether to symmetrize the input samples or not

        ------------------------------------------------------------------------
        Returns:
        log-probs        tf.Tensor of shape (number of samples,)
                         the log-probability of each sample
        log-amps         tf.Tensor of shape (number of samples,)
                         the complex log-amplitudes of each sample
        """

        num_samples = samples.shape[0]
        samples = tf.cast(samples, dtype=tf.int32)
        if symmetrize:
            samples, num_tot_symmetries = self.apply_symmetries(samples, parity)
            samples = tf.reshape(samples, (num_tot_symmetries * num_samples, self.Ny, self.Nx))
        else:
            samples = tf.reshape(samples, (num_samples, self.Ny, self.Nx))
            num_tot_symmetries = 1

        samples_transpose = tf.transpose(samples, perm=[1, 2, 0])
        rnn_states = {}
        inputs = {}
        for ny in range(-2, self.Ny):  # Loop over the number of sites
            for nx in range(-2, self.Nx + 2):
                inputs[f"{nx}{ny}"] = tf.zeros((num_tot_symmetries * num_samples, self.local_hilbert_space),
                                               dtype=self.tf_dtype)
                if self.weight_sharing=='sublattice':
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(num_tot_symmetries * num_samples)[0]
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(num_tot_symmetries * num_samples)[0]

        probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        phases = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

        # Making a loop over the sites with the 2DRNN
        num_up = tf.zeros(num_samples * num_tot_symmetries, dtype=self.tf_dtype)
        num_generated_spins = 0
        for ny in range(self.Ny):

            if ny % 2 == 0:

                for nx in range(self.Nx):  # left to right
                    neighbor_inputs = (inputs[f"{nx - 1}{ny}"], inputs[f"{nx}{ny - 1}"])
                    neighbor_states = (rnn_states[f"{nx - 1}{ny}"], rnn_states[f"{nx}{ny - 1}"])

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(output_ampl)
                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    sample_temp = samples_transpose[nx, ny]
                    inputs[f"{nx}{ny}"] = tf.cast(tf.one_hot(
                        sample_temp, self.local_hilbert_space, axis=-1), dtype=self.tf_dtype)

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))

                    probs[nx][ny] = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)
                    if self.use_complex:
                        output_phase_ss = self.softsign(output_phase)
                        phases[nx][ny] = output_phase_ss

            if ny % 2 == 1:

                for nx in range(self.Nx - 1, -1, -1):  # right to left

                    neighbor_inputs = (inputs[f"{nx + 1}{ny}"], inputs[f"{nx}{ny - 1}"])
                    neighbor_states = (rnn_states[f"{nx + 1}{ny}"], rnn_states[f"{nx}{ny - 1}"])

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(output_ampl)
                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    sample_temp = samples_transpose[nx, ny]
                    inputs[f"{nx}{ny}"] = tf.cast(tf.one_hot(
                        sample_temp, self.local_hilbert_space, axis=-1), dtype=self.tf_dtype)

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))

                    probs[nx][ny] = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)
                    if self.use_complex:
                        output_phase_ss = self.softsign(output_phase)
                        phases[nx][ny] = output_phase_ss

        one_hot_samples = tf.transpose(tf.one_hot(tf.cast(
            samples_transpose, dtype=tf.int32), depth=self.local_hilbert_space, dtype=self.tf_dtype), perm=[2, 0, 1, 3])
        probs = tf.transpose(tf.stack(values=probs, axis=0), perm=[2, 0, 1, 3])
        log_probs = tf.reduce_sum(tf.reduce_sum(tf.math.log(tf.reduce_sum(
            tf.multiply(probs, one_hot_samples), axis=3)), axis=2), axis=1)

        if self.use_complex:
            phases = tf.transpose(tf.stack(values=phases, axis=0), perm=[2, 0, 1, 3])
            total_phases = tf.cast(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(
                tf.multiply(phases, one_hot_samples), axis=3), axis=2), axis=1), dtype=self.tf_dtype)
        else:
            total_phases = tf.zeros_like(log_probs)
        log_amplitudes = tf.complex(0.5 * log_probs, total_phases)

        if symmetrize:
            log_probs, log_amplitudes = self.average_symmetries(log_probs, total_phases, num_tot_symmetries,
                                                                num_samples)

        return log_probs, log_amplitudes


class periodic_cMDRNNWavefunction(RNNWavefunction):
    def __init__(self,
                 systemsize_x: int, systemsize_y: int,
                 units: int, local_hilbert_space: int = 2,
                 cell=MDPeriodic,
                 weight_sharing='all',
                 use_complex=False,
                 h_symmetries=True,
                 seed=111,
                 kernel_initializer='glorot_uniform',
                 tf_dtype=tf.float32):

        """
        systemsize_x, systemsize_y:  integers
                     number of sites in x, y directions of SQUARE lattice
        num_units:   integer
                     length of hidden vector, controls number of parameters
        local_hilbert_space: integer
                     size of the local hilbet space of sites on SQUARE lattice
                     (related to the number of spins per RNN cell, which
                     is related to the true lattice of interest)
        cell:        a tensorflow RNN cell
        seed:        pseudo-random number generator
        h_symmetries: bool
                     determines whether HAMILTONIAN symmetries are
                     enforced during the sampling step
                     strictly enforced by enforcing zero magnetization
        l_symmetries: bool
                     determines whether LATTICE symmetries are applied
                     enforced in a data-augmentation way
        lattice: string
                     the lattice of the underlying system
                     determines which lattice symmetries to apply
        kernel_initializer: string
                     indicates how the model's weights are initialized
        tf_dtype: tensorflow data type class
        """

        self.Nx = systemsize_x
        self.Ny = systemsize_y

        super().__init__(local_hilbert_space, self.Nx * self.Ny, 'periodic', tf_dtype)

        self.weight_sharing = weight_sharing
        assert self.weight_sharing in ['all', 'sublattice'], \
            'weight_sharing must be on of `all` or `sublattice`'
        self.use_complex = use_complex
        self.h_symmetries = h_symmetries
        self.log_cutoff = 1e-10

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        tf.random.set_seed(seed)  # tensorflow pseudo-random generator

        if self.weight_sharing == 'sublattice':
            A_lattice, B_lattice, C_lattice, _ = generate_sublattices_triangular(systemsize_x, systemsize_x)
            map_A = dict(zip(A_lattice, [0] * len(A_lattice)))
            map_B = dict(zip(B_lattice, [1] * len(B_lattice)))
            map_C = dict(zip(C_lattice, [2] * len(C_lattice)))

            self.lattice_map = map_A | map_B | map_C
            self.rnn = [
                cell(num_units=units, local_hilbert_size=local_hilbert_space, name=f"RNN_{0}_{i}",
                     dtype=self.tf_dtype) for i in range(3)]
            self.dense = [tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense_{i}', 
                                                dtype=self.tf_dtype, kernel_initializer=kernel_initializer) 
                                                for i in range(3)]
            if self.use_complex:
                self.dense_phase = [
                    tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense_phase_{i}', 
                                          dtype=self.tf_dtype, kernel_initializer=kernel_initializer)
                                          for i in range(3)]
        else:
            self.rnn = cell(num_units=units, local_hilbert_size=local_hilbert_space, name=f"RNN", dtype=self.tf_dtype)
            self.dense = tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense', dtype=self.tf_dtype,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=kernel_initializer)
            if self.use_complex:
                self.dense_phase = tf.keras.layers.Dense(local_hilbert_space, name=f'RNNWF_dense_phase',
                                                        dtype=self.tf_dtype,
                                                        kernel_initializer=kernel_initializer,
                                                        bias_initializer=kernel_initializer)

    def sample(self, num_samples):

        """
            generate samples from the probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            num_samples:     int
                             number of samples to be produced

            ------------------------------------------------------------------------
            Returns:        samples
                            tf.Tensor of shape (num_samples, Nx * Ny * num_spins_per_cell)
        """

        samples = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

        rnn_states = {}
        inputs = {}
        for ny in range(-2, self.Ny):
            for nx in range(-2, self.Nx + 2):
                inputs[f"{nx}{ny}"] = tf.zeros((num_samples, self.local_hilbert_space),
                                               dtype=self.tf_dtype)
                if self.weight_sharing=='sublattice':
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(num_samples)[0]
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(num_samples)[0]

        # Making a loop over the sites with the 2DRNN
        num_up = tf.zeros(num_samples, dtype=self.tf_dtype)
        num_generated_spins = 0
        for ny in range(self.Ny):
            if ny % 2 == 0:
                for nx in range(self.Nx):  # left to right
                    neighbor_inputs = (inputs[f"{nx - 1}{ny}"], inputs[f"{nx}{ny - 1}"],
                                       inputs[f"{(nx + 1) % self.Nx}{ny}"], inputs[f"{nx}{(ny + 1) % self.Ny}"])
                    neighbor_states = (rnn_states[f"{nx - 1}{ny}"], rnn_states[f"{nx}{ny - 1}"],
                                       rnn_states[f"{(nx + 1) % self.Nx}{ny}"], rnn_states[f"{nx}{(ny + 1) % self.Ny}"])

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(
                        output_ampl)  # (num_samples, local_Hilbert_space)

                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    output_ampl_softmax = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)
                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output_ampl_softmax), num_samples=1),
                                             [-1, ])  # (num_samples, )
                    sample_temp_one_hot = tf.one_hot(sample_temp, depth=self.local_hilbert_space,
                                                     dtype=self.tf_dtype)  # (num_samples, local_hilbert_space)
                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))
                    sample_temp = tf.expand_dims(sample_temp, axis=-1)

                    inputs[f"{nx}{ny}"] = sample_temp_one_hot  # (num_samples, local_hilbert_space)
                    samples[nx][ny] = sample_temp  # (num_samples, num_spins_per_cell)

            if ny % 2 == 1:
                for nx in range(self.Nx - 1, -1, -1):  # right to left
                    neighbor_inputs = [inputs[f"{nx + 1}{ny}"], inputs[f"{nx}{ny - 1}"],
                                       inputs[f"{(nx + 1) % self.Nx}{ny}"], inputs[f"{nx}{(ny + 1) % self.Ny}"]]
                    neighbor_states = [rnn_states[f"{nx + 1}{ny}"], rnn_states[f"{nx}{ny - 1}"],
                                       rnn_states[f"{(nx - 1) % self.Nx}{ny}"], rnn_states[f"{nx}{(ny - 1) % self.Ny}"]]

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(
                        output_ampl)  # (num_samples, local_Hilbert_space)

                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    output_ampl_softmax = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)

                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output_ampl_softmax), num_samples=1),
                                             [-1, ])  # (num_samples, )
                    sample_temp_one_hot = tf.one_hot(sample_temp, depth=self.local_hilbert_space,
                                                     dtype=self.tf_dtype)  # (num_samples, local_hilbert_space)

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))
                    sample_temp = tf.expand_dims(sample_temp, axis=-1)

                    inputs[f"{nx}{ny}"] = sample_temp_one_hot  # (num_samples, local_hilbert_space)
                    samples[nx][ny] = sample_temp  # (num_samples, num_spins_per_cell)

        samples = tf.transpose(tf.stack(values=samples, axis=0), perm=[2, 0, 1, 3])

        flat_samples = tf.cast(tf.reshape(samples, (-1, self.Nx * self.Ny)), dtype=self.tf_dtype)

        return flat_samples

    def log_probsamps(self, samples, symmetrize: bool, parity: bool):
        """
        calculate the log-probabilities of input samples
        ------------------------------------------------------------------------
        Parameters:

        samples:         tf.tensor
                         A tf.Tensor of shape (num_samples , Nx, Ny, 1)
                         containing the input samples in integer encoding of the sample
        symmetrize:      bool
                         a boolean that tells us whether to apply lattice symmetries on
                         the input samples or not

        ------------------------------------------------------------------------
        Returns:
        log-probs        tf.Tensor of shape (number of samples,)
                         the log-probability of each sample
        log-amps         tf.Tensor of shape (number of samples,)
                         the complex log-amplitudes of each sample
        """
        num_samples = samples.shape[0]
        samples = tf.cast(samples, dtype=tf.int32)
        if symmetrize:
            samples, num_tot_symmetries = self.apply_symmetries(samples, parity)
            samples = tf.reshape(samples, (num_tot_symmetries * num_samples, self.Ny, self.Nx))
        else:
            samples = tf.reshape(samples, (num_samples, self.Ny, self.Nx))
            num_tot_symmetries = 1
        samples_transpose = tf.transpose(samples, perm=[1, 2, 0])
        rnn_states = {}
        inputs = {}
        for ny in range(-2, self.Ny):
            for nx in range(-2, self.Nx + 2):
                inputs[f"{nx}{ny}"] = tf.zeros((num_tot_symmetries * num_samples, self.local_hilbert_space),
                                dtype=self.tf_dtype)
                if self.weight_sharing=='sublattice':
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(num_tot_symmetries * num_samples)[0]
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(num_tot_symmetries * num_samples)[0]

        probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        phases = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

        # Making a loop over the sites with the 2DRNN
        num_up = tf.zeros(num_samples * num_tot_symmetries, dtype=self.tf_dtype)
        num_generated_spins = 0
        for ny in range(self.Ny):
            if ny % 2 == 0:
                for nx in range(self.Nx):  # left to right

                    neighbor_inputs = [inputs[f"{nx - 1}{ny}"], inputs[f"{nx}{ny - 1}"],
                                       inputs[f"{(nx + 1) % self.Nx}{ny}"], inputs[f"{nx}{(ny + 1) % self.Ny}"]]
                    neighbor_states = [rnn_states[f"{nx - 1}{ny}"], rnn_states[f"{nx}{ny - 1}"],
                                       rnn_states[f"{(nx + 1) % self.Nx}{ny}"], rnn_states[f"{nx}{(ny + 1) % self.Ny}"]]

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(output_ampl)
                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    sample_temp = samples_transpose[nx, ny]
                    inputs[f"{nx}{ny}"] = tf.cast(tf.one_hot(
                        sample_temp, self.local_hilbert_space, axis=-1), dtype=self.tf_dtype)

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))

                    probs[nx][ny] = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)
                    if self.use_complex:
                        output_phase_ss = self.softsign(output_phase)
                        phases[nx][ny] = output_phase_ss

            if ny % 2 == 1:
                for nx in range(self.Nx - 1, -1, -1):  # right to left

                    neighbor_inputs = [inputs[f"{nx + 1}{ny}"], inputs[f"{nx}{ny - 1}"],
                                       inputs[f"{(nx + 1) % self.Nx}{ny}"], inputs[f"{nx}{(ny + 1) % self.Ny}"]]
                    neighbor_states = [rnn_states[f"{nx + 1}{ny}"], rnn_states[f"{nx}{ny - 1}"],
                                       rnn_states[f"{(nx - 1) % self.Nx}{ny}"], rnn_states[f"{nx}{(ny - 1) % self.Ny}"]]

                    if self.weight_sharing=='sublattice':
                        site = coord_to_site_bravais(self.Nx,nx,ny)
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[self.lattice_map[site]](
                            neighbor_inputs, neighbor_states)
                        output_ampl = self.dense[self.lattice_map[site]](rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase[self.lattice_map[site]](rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn(neighbor_inputs, neighbor_states)
                        output_ampl = self.dense(rnn_output)
                        if self.use_complex:
                            output_phase = self.dense_phase(rnn_output)

                    output_ampl_softmax = tf.keras.activations.softmax(output_ampl)
                    if self.h_symmetries:
                        output_ampl_softmax = self.mag_normalization(output_ampl_softmax, num_up, num_generated_spins)

                    sample_temp = samples_transpose[nx, ny]
                    inputs[f"{nx}{ny}"] = tf.cast(tf.one_hot(
                        sample_temp, self.local_hilbert_space, axis=-1), dtype=self.tf_dtype)

                    num_generated_spins += 1
                    num_up = tf.add(num_up, tf.cast(sample_temp, self.tf_dtype))

                    probs[nx][ny] = tf.clip_by_value(output_ampl_softmax, self.log_cutoff, 1.0)
                    if self.use_complex:
                        output_phase_ss = self.softsign(output_phase)
                        phases[nx][ny] = output_phase_ss

        one_hot_samples = tf.transpose(tf.one_hot(tf.cast(
            samples_transpose, dtype=tf.int32), depth=self.local_hilbert_space, dtype=self.tf_dtype), perm=[2, 0, 1, 3])

        probs = tf.transpose(tf.stack(values=probs, axis=0), perm=[2, 0, 1, 3])
        log_probs = tf.reduce_sum(tf.reduce_sum(tf.math.log(tf.reduce_sum(
            tf.multiply(probs, one_hot_samples), axis=3)), axis=2), axis=1)
        if self.use_complex:
            phases = tf.transpose(tf.stack(values=phases, axis=0), perm=[2, 0, 1, 3])
            total_phases = tf.cast(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(
                tf.multiply(phases, one_hot_samples), axis=3), axis=2), axis=1), dtype=self.tf_dtype)
        else:
            total_phases = tf.zeros_like(log_probs)
        log_amplitudes = tf.complex(0.5 * log_probs, total_phases)

        if symmetrize:
            log_probs, log_amplitudes = self.average_symmetries(log_probs, total_phases, num_tot_symmetries,
                                                                num_samples)
        return log_probs, log_amplitudes


if __name__ == '__main__':

    print("Test normalization")

    def tf_integer_to_binary(integer, bitsize: int, axis=1):
        return tf.reverse(tf.math.mod(tf.bitwise.right_shift(tf.expand_dims(integer, 1), tf.range(bitsize)), 2), [axis])

    Lx = 4
    Ly = 4
    Nspins = Lx * Ly
    tf_dtype = tf.float32
    indices = tf.range(0, 2 ** Nspins)
    all_states = tf.cast(tf_integer_to_binary(indices, Nspins), dtype=tf_dtype)
    nh = 10
    bc = 'open'
    weight_sharing='sublattice'

    if bc == 'periodic':
        cell = get_rnn_cell('MDPeriodic')
        rnn = periodic_cMDRNNWavefunction(Lx, Ly, nh, weight_sharing=weight_sharing)
    else:  # 'open'
        cell = get_rnn_cell('MDGRU')
        rnn = cMDRNNWavefunction(Lx, Ly, nh, weight_sharing=weight_sharing)

    log_p, log_amp = rnn.log_probsamps(all_states, symmetrize=True, parity=False)
    print(tf.reduce_sum(tf.math.exp(log_p)))
    print(tf.norm(tf.math.exp(log_amp)))
