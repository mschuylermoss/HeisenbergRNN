import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
import numpy as np

from train import train_
from utils import get_train_method, data_saver

from interactions import buildlattice_square, buildlattice_alltoall
from correlations import get_batched_interactions_Jmats
from correlations import get_Heisenberg_realspace_Correlation_Vectorized, get_Si, calculate_expectation_ft_Si_square
from correlations import undo_marshall_sign, calculate_structure_factor


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def estimate_energy(config, save_path, energy_fxn, sample_fxn, log_fxn, strategy):
    PRINT = config['PRINT']
    WRITE = config['WRITE']
    num_samples_final_energy_estimate = config.get('num_samples_final_energy_estimate', None)
    batch_size = config.get('num_samples', 1024)
    Nx = config['Nx']
    Ny = config['Ny']
    N_spins = Nx * Ny

    num_batches = num_samples_final_energy_estimate // batch_size
    if num_samples_final_energy_estimate % batch_size != 0:
        num_batches += 1
        num_samples_final_energy_estimate = num_batches * batch_size

    num_samples_per_device = int(np.ceil(batch_size / strategy.num_replicas_in_sync))
    if batch_size % strategy.num_replicas_in_sync != 0:
        batch_size = num_samples_per_device * strategy.num_replicas_in_sync
        num_samples_final_energy_estimate = batch_size * num_batches

    if PRINT:
        print(f"\nCalculating the final energy with {num_samples_final_energy_estimate} samples")
        print(f"Batch size = {batch_size}")
        print(f"Number of samples per device: {num_samples_per_device}\n")

    ens_final_directory = f'/final_energies/ns{num_samples_final_energy_estimate}'
    if not os.path.exists(save_path + ens_final_directory):
        os.makedirs(save_path + ens_final_directory)

    if num_samples_final_energy_estimate is not None:

        @tf.function()
        def final_energy_single():
            print(f"Tracing single energy calculation")

            samples_tf = sample_fxn(num_samples_per_device)

            log_probs_tf, log_amps_tf = log_fxn(samples_tf)
            local_energies_tf = energy_fxn(samples_tf, log_amps_tf)
            return tf.stack([tf.math.real(local_energies_tf),
                             tf.math.imag(local_energies_tf)], axis=-1)

        @tf.function
        def distributed_final_energy():
            print("Tracing distributed energy calculation")
            local_energies_tf_per_rep = strategy.run(final_energy_single)
            local_energies_tf_per_rep_reduced = strategy.gather(local_energies_tf_per_rep,
                                                                axis=0)
            return tf.complex(local_energies_tf_per_rep_reduced[:, 0],
                              local_energies_tf_per_rep_reduced[:, 1])

        if not os.path.exists(save_path + ens_final_directory + f'/final_energies_mean_N{N_spins}.npy'):

            if PRINT:
                print("Calculating final energy from RNNWF")
            energies_total = []
            for batch in range(batch_size, num_samples_final_energy_estimate + 1, batch_size):
                energies_chunk = distributed_final_energy()
                energies_total.append(energies_chunk)
                print(f"Calculating batch {batch}/{num_samples_final_energy_estimate}")
                print(f"The mean of this chunk is : ", np.mean(energies_chunk), np.var(energies_chunk))

            energies_total = np.concatenate(energies_total).flatten()
            final_energies_mean = np.mean(energies_total)
            final_energies_var = np.var(energies_total)

            if WRITE:
                np.save(save_path + ens_final_directory + f'/final_energies_mean_N{N_spins}.npy',
                        final_energies_mean)
                np.save(save_path + ens_final_directory + f'/final_energies_var_N{N_spins}.npy', final_energies_var)

        else:
            final_energies_mean = np.load(save_path + ens_final_directory + f'/final_energies_mean_N{N_spins}.npy')
            final_energies_var = np.load(save_path + ens_final_directory + f'/final_energies_var_N{N_spins}.npy')

        if PRINT:
            print(f"\nFinal training quantities (calculated with {num_samples_final_energy_estimate} samples):")
            print(f"meanE = {final_energies_mean}")
            print(f"varE = {final_energies_var}")


def estimate_correlations_distributed(config, save_path, sample_fxn, log_fxn, strategy):
    PRINT = config['PRINT']
    task_id = config.get('task_id', 0)

    number_of_replicas = strategy.num_replicas_in_sync
    num_samples_final_correlations_estimate = config.get('num_samples_final_correlations_estimate', None)
    batch_size_samples = config.get('num_samples', 1024)
    batch_size_per_device = int(np.ceil(batch_size_samples / number_of_replicas))
    batch_size_samples = batch_size_per_device * number_of_replicas
    num_sample_batches = num_samples_final_correlations_estimate // batch_size_samples
    if num_samples_final_correlations_estimate % batch_size_samples != 0:
        num_sample_batches += 1
        num_samples_final_correlations_estimate = num_sample_batches * batch_size_samples

    Nx = config['Nx']
    Ny = config['Ny']
    tf_dtype = config.get('tf_dtype', tf.float32)
    N_spins = Nx * Ny
    if config['boundary_condition'] == 'periodic':
        periodic = True
    else:
        periodic = False

    correlation_mode = config['correlation_mode']
    assert correlation_mode in ['Sxyz',
                                'Sz'], f"`correlation` mode must be `Sxyz` or `Sz`, received {correlation_mode}"

    if PRINT:
        print(f"\nCalculating final correlations with {num_samples_final_correlations_estimate} samples")
        print(f"Batch size = {batch_size_samples} ({num_sample_batches} batches)")
        print(f"{number_of_replicas} devices, so batch size per device: {batch_size_per_device}")
        print(f"Mode = {correlation_mode}")

    corr_final_directory = f'/final_corrs/ns_{num_samples_final_correlations_estimate}'
    if not os.path.exists(save_path + corr_final_directory) and task_id == 0:
        os.makedirs(save_path + corr_final_directory)

    interactions = buildlattice_alltoall(Nx)
    interactions_batch_size = len(buildlattice_square(Nx, Ny))  # number of first order
    J_matrix_list, interactions_batched = get_batched_interactions_Jmats(Nx, interactions,
                                                                         interactions_batch_size, tf_dtype)
    num_interaction_batches = len(J_matrix_list.keys())
    path_exists_Sxy = os.path.exists(save_path + corr_final_directory + f'/Sxy_matrix.npy')
    path_exists_Sz = os.path.exists(save_path + corr_final_directory + f'/Sz_matrix.npy')

    if correlation_mode == 'Sxyz':  # depending on the mode we check if both exist
        path_exists = (path_exists_Sxy and path_exists_Sz)
    else:
        path_exists = path_exists_Sz

    if not path_exists:

        if PRINT:
            print("Calculating final real space correlations from RNNWF")
            print("First getting all samples...")

        @tf.function()
        def distributed_get_samples():
            samples_dist = strategy.run(sample_fxn, args=(batch_size_per_device,))
            return strategy.gather(samples_dist, axis=0)

        @tf.function()
        def distributed_get_samples_logpsis():
            samples_dist = strategy.run(sample_fxn, args=(batch_size_per_device,))
            _, logpsis = strategy.run(log_fxn, args=(samples_dist,))
            logpsis_stacked = strategy.run(lambda _x: tf.stack([tf.math.real(_x),
                                                                tf.math.imag(_x)], axis=-1), args=(logpsis,))
            logpsis_stacked_gathered = strategy.gather(logpsis_stacked, axis=0)

            return strategy.gather(samples_dist, axis=0), tf.complex(logpsis_stacked_gathered[:, 0],
                                                                     logpsis_stacked_gathered[:, 1])

        all_samples = {}
        all_log_amps = {}
        for sampling_batch in range(num_sample_batches):
            if correlation_mode == 'Sxyz':
                samples_batch, log_amps_batch = distributed_get_samples_logpsis()
                all_samples[sampling_batch] = samples_batch
                all_log_amps[sampling_batch] = log_amps_batch
            else:
                all_samples[sampling_batch] = distributed_get_samples()
            print(f"Getting samples for batch {sampling_batch + 1}/{num_sample_batches}")

        if task_id == 0:
            save_dict(all_samples,
                      save_path + corr_final_directory + f'/final_rcorrelations_samples.pkl')
            if correlation_mode == 'Sxyz':
                save_dict(all_log_amps,
                          save_path + corr_final_directory + f'/final_rcorrelations_logamps.pkl')

        sz_matrix = np.full((N_spins, N_spins), np.nan)
        var_sz_matrix = np.full((N_spins, N_spins), np.nan)
        sxy_matrix = np.full((N_spins, N_spins), np.nan)
        var_sxy_matrix = np.full((N_spins, N_spins), np.nan)
        if task_id == 0:
            np.save(save_path + corr_final_directory + f'/Sz_matrix',
                    sz_matrix)
            np.save(save_path + corr_final_directory + f'/var_Sz_matrix',
                    var_sz_matrix)
            if correlation_mode == 'Sxyz':
                np.save(save_path + corr_final_directory + f'/Sxy_matrix',
                        sxy_matrix)
                np.save(save_path + corr_final_directory + f'/var_Sxy_matrix',
                        var_sxy_matrix)

    else:  # path does exist continue calculating
        if PRINT:
            print("Continue calculating final real space correlations from RNNWF")
            print("First loading all samples...")

        all_samples = load_dict(save_path + corr_final_directory + '/final_rcorrelations_samples.pkl')
        if correlation_mode == 'Sxyz':
            all_log_amps = load_dict(save_path + corr_final_directory + '/final_rcorrelations_logamps.pkl')

        sz_matrix = np.load(save_path + corr_final_directory + f'/Sz_matrix.npy')
        var_sz_matrix = np.load(save_path + corr_final_directory + f'/var_Sz_matrix.npy')
        if correlation_mode == 'Sxyz':
            sxy_matrix = np.load(save_path + corr_final_directory + f'/Sxy_matrix.npy')
            var_sxy_matrix = np.load(save_path + corr_final_directory + f'/var_Sxy_matrix.npy')

    rsp_corr_fxn_xx_yy, rsp_corr_fxn_zz = get_Heisenberg_realspace_Correlation_Vectorized(
        log_fxn,
        tf_dtype=tf_dtype)

    @tf.function()
    def distributed_rsp_corr_fxn_zz_xx_yy(samples, logamps, j_matrix):
        print("Tracing distributed xx_yy_zz")

        @tf.function()
        def value_fn(ctx):
            samples_chunked = tf.reshape(samples, (number_of_replicas, batch_size_per_device, N_spins))
            logamps_chunked = tf.reshape(logamps, (number_of_replicas, batch_size_per_device))
            return samples_chunked[ctx.replica_id_in_sync_group], logamps_chunked[ctx.replica_id_in_sync_group]

        samples_distributed, logpsis_distributed = strategy.experimental_distribute_values_from_function(value_fn)
        zz = strategy.run(lambda _x, _J: tf.math.real(rsp_corr_fxn_zz(_x, _J)), args=(samples_distributed, j_matrix,))
        xx_yy = strategy.run(lambda _x, _y, _J: tf.math.real(rsp_corr_fxn_xx_yy(_x, _y, _J)),
                             args=(samples_distributed, logpsis_distributed, j_matrix,))
        return strategy.gather(zz, axis=0), strategy.gather(xx_yy, axis=0)

    @tf.function()
    def distributed_rsp_corr_fxn_zz(samples, j_matrix):
        print("Tracing distributed zz")

        def value_fn(ctx):
            samples_chunked = tf.reshape(samples, (number_of_replicas, batch_size_per_device, N_spins))
            return samples_chunked[ctx.replica_id_in_sync_group]

        samples_distributed = strategy.experimental_distribute_values_from_function(value_fn)
        zz = strategy.run(lambda _x, _J: tf.math.real(rsp_corr_fxn_zz(_x, _J)), args=(samples_distributed, j_matrix,))
        return strategy.gather(zz, axis=0)

    for batch_i in range(num_interaction_batches):
        timestart = time.time()
        print(f"Calculating correlations for interaction batch {batch_i + 1}/{num_interaction_batches}")
        J_mat_batch = J_matrix_list[batch_i]
        interactions_batch = np.array(interactions_batched[batch_i])
        if not np.isnan(sz_matrix[interactions_batch[0, 0], interactions_batch[0, 1]]):
            print(f"Correlations for interaction batch {batch_i + 1} already calculated!")
            continue
        batch_means_sxy = np.zeros((num_sample_batches, len(interactions_batch)))
        batch_means_sz = np.zeros((num_sample_batches, len(interactions_batch)))
        batch_vars_sxy = np.zeros((num_sample_batches, len(interactions_batch)))
        batch_vars_sz = np.zeros((num_sample_batches, len(interactions_batch)))
        for batch_s in range(num_sample_batches):
            print(f"sample batch {batch_s}/{num_sample_batches}")
            samples_batch = all_samples[batch_s]
            if correlation_mode == 'Sxyz':
                log_amps_batch = all_log_amps[batch_s]
                sziszj, sxyisxyj = distributed_rsp_corr_fxn_zz_xx_yy(samples_batch, log_amps_batch, J_mat_batch)
            else:
                sziszj = distributed_rsp_corr_fxn_zz(samples_batch, J_mat_batch)
            batch_means_sz[batch_s, :] = np.mean(sziszj.numpy(), axis=0)
            batch_vars_sz[batch_s, :] = np.var(sziszj.numpy(), axis=0)
            if correlation_mode == 'Sxyz':
                batch_means_sxy[batch_s, :] = np.mean(sxyisxyj.numpy(), axis=0)
                batch_vars_sxy[batch_s, :] = np.var(sxyisxyj.numpy(), axis=0)

        sxy_allsamples = np.mean(batch_means_sxy, axis=0)
        sz_allsamples = np.mean(batch_means_sz, axis=0)
        if correlation_mode == 'Sxyz':
            var_sxy_allsamples = np.mean(batch_vars_sxy, axis=0) + np.var(batch_means_sxy, axis=0)
        var_sz_allsamples = np.mean(batch_vars_sz, axis=0) + np.var(batch_means_sz, axis=0)

        spin_is = interactions_batch[:, 0]
        spin_js = interactions_batch[:, 1]
        sz_matrix[spin_is, spin_js] = sz_allsamples
        var_sz_matrix[spin_is, spin_js] = var_sz_allsamples
        if correlation_mode == 'Sxyz':
            sxy_matrix[spin_is, spin_js] = sxy_allsamples
            var_sxy_matrix[spin_is, spin_js] = var_sxy_allsamples

        if task_id == 0:
            np.save(save_path + corr_final_directory + f'/Sz_matrix',
                    sz_matrix)
            np.save(save_path + corr_final_directory + f'/var_Sz_matrix',
                    var_sz_matrix)
            if correlation_mode == 'Sxyz':
                np.save(save_path + corr_final_directory + f'/Sxy_matrix',
                        sxy_matrix)
                np.save(save_path + corr_final_directory + f'/var_Sxy_matrix',
                        var_sxy_matrix)
        print(f"Time per interaction batch: {time.time() - timestart}")

    if not np.isnan(sz_matrix[-1, -1]):  # done calculating matrix
        if correlation_mode == 'Sxyz':
            undo_marshall_sign_minus_signs = undo_marshall_sign(Nx)
            SiSj = sz_matrix + sxy_matrix * undo_marshall_sign_minus_signs
            var_SiSj = var_sz_matrix + var_sxy_matrix
            Sk, err_Sk = calculate_structure_factor(Nx, SiSj, var_Sij=var_SiSj, periodic=periodic)
            np.save(save_path + corr_final_directory + f'Sk_from_SiSj', Sk)
            np.save(save_path + corr_final_directory + f'err_Sk_from_SiSj', err_Sk)
            print(f"Sk (from <SiSj>) = {Sk}")

        else:
            SiSj = 3 * sz_matrix
            var_SiSj = 3 * var_sz_matrix
            Sk, err_Sk = calculate_structure_factor(Nx, SiSj, var_Sij=var_SiSj, periodic=periodic)
            np.save(save_path + corr_final_directory + f'Sk_from_SziSzj', Sk)
            np.save(save_path + corr_final_directory + f'err_Sk_from_SziSzj', err_Sk)
            print(f"Sk (from <SziSzj>) = {Sk}")

    if PRINT:
        print(f"\n Done calculating correlation matrices... "
              f"(calculated with {num_samples_final_correlations_estimate} samples)")


def estimate_Sk_from_Si_distributed(config, save_path, sample_fxn, log_fxn, strategy):
    PRINT = config['PRINT']
    task_id = config.get('task_id', 0)

    number_of_replicas = strategy.num_replicas_in_sync
    num_samples_final_correlations_estimate = config.get('num_samples_final_correlations_estimate', None)
    batch_size_samples = config.get('num_samples', 1024)
    batch_size_per_device = int(np.ceil(batch_size_samples / number_of_replicas))
    batch_size_samples = batch_size_per_device * number_of_replicas
    num_sample_batches = num_samples_final_correlations_estimate // batch_size_samples
    if num_samples_final_correlations_estimate % batch_size_samples != 0:
        num_sample_batches += 1
        num_samples_final_correlations_estimate = num_sample_batches * batch_size_samples

    Nx = config['Nx']
    Ny = config['Ny']
    tf_dtype = config.get('tf_dtype', tf.float32)
    N_spins = Nx * Ny

    correlation_mode = config['correlation_mode']
    assert correlation_mode in ['Sxyz',
                                'Sz'], f"`correlation` mode must be `Sxyz` or `Sz`, received {correlation_mode}"

    if PRINT:
        print(f"\nCalculating Sk from <Si>'s with {num_samples_final_correlations_estimate} samples")
        print(f"Batch size = {batch_size_samples} ({num_sample_batches} batches)")
        print(f"{number_of_replicas} devices, so batch size per device: {batch_size_per_device}")
        print(f"Mode = {correlation_mode}")

    corr_final_directory = f'/final_corrs/ns_{num_samples_final_correlations_estimate}'
    if not os.path.exists(save_path + corr_final_directory) and task_id == 0:
        os.makedirs(save_path + corr_final_directory)

    path_exists_Si = os.path.exists(save_path + corr_final_directory + f'/Sk_from_Si.npy')
    path_exists_Szi = os.path.exists(save_path + corr_final_directory + f'/Sk_from_Szi.npy')

    if correlation_mode == 'Sxyz':  # depending on the mode we check if both exist
        path_exists = path_exists_Si
    else:
        path_exists = path_exists_Szi

    if not path_exists:

        if PRINT:
            print("Calculating Sk from <Si>'s from RNNWF")
            print("First getting all samples...")

        @tf.function()
        def distributed_get_samples():
            samples_dist = strategy.run(sample_fxn, args=(batch_size_per_device,))
            return strategy.gather(samples_dist, axis=0)

        @tf.function()
        def distributed_get_samples_logpsis():
            samples_dist = strategy.run(sample_fxn, args=(batch_size_per_device,))
            _, logpsis = strategy.run(log_fxn, args=(samples_dist,))
            logpsis_stacked = strategy.run(lambda _x: tf.stack([tf.math.real(_x),
                                                                tf.math.imag(_x)], axis=-1), args=(logpsis,))
            logpsis_stacked_gathered = strategy.gather(logpsis_stacked, axis=0)

            return strategy.gather(samples_dist, axis=0), tf.complex(logpsis_stacked_gathered[:, 0],
                                                                     logpsis_stacked_gathered[:, 1])

        all_samples = {}
        all_log_amps = {}
        for sampling_batch in range(num_sample_batches):
            if correlation_mode == 'Sxyz':
                samples_batch, log_amps_batch = distributed_get_samples_logpsis()
                all_samples[sampling_batch] = samples_batch
                all_log_amps[sampling_batch] = log_amps_batch
            else:
                all_samples[sampling_batch] = distributed_get_samples()
            print(f"Getting samples for batch {sampling_batch + 1}/{num_sample_batches}")

        Si_xx_yy, Si_zz = get_Si(log_fxn, tf_dtype=tf_dtype)

        @tf.function()
        def distributed_Si_fxn_zz_xx_yy(samples, logamps):
            print("Tracing distributed xx_yy_zz")

            @tf.function()
            def value_fn(ctx):
                samples_chunked = tf.reshape(samples, (number_of_replicas, batch_size_per_device, N_spins))
                logamps_chunked = tf.reshape(logamps, (number_of_replicas, batch_size_per_device))
                return samples_chunked[ctx.replica_id_in_sync_group], logamps_chunked[ctx.replica_id_in_sync_group]

            samples_distributed, logpsis_distributed = strategy.experimental_distribute_values_from_function(value_fn)
            zz = strategy.run(lambda _x: tf.math.real(Si_zz(_x)), args=(samples_distributed,))
            xx_yy = strategy.run(lambda _x, _y: tf.math.real(Si_xx_yy(_x, _y)),
                                 args=(samples_distributed, logpsis_distributed,))
            return strategy.gather(zz, axis=0), strategy.gather(xx_yy, axis=0)

        @tf.function()
        def distributed_Si_fxn_zz(samples):
            print("Tracing distributed zz")

            def value_fn(ctx):
                samples_chunked = tf.reshape(samples, (number_of_replicas, batch_size_per_device, N_spins))
                return samples_chunked[ctx.replica_id_in_sync_group]

            samples_distributed = strategy.experimental_distribute_values_from_function(value_fn)
            zz = strategy.run(lambda _x: tf.math.real(Si_zz(_x)), args=(samples_distributed,))
            return strategy.gather(zz, axis=0)

        timestart = time.time()
        batch_means_ft_si = np.zeros((num_sample_batches))
        batch_vars_ft_si = np.zeros((num_sample_batches))
        for batch_s in range(num_sample_batches):
            print(f"sample batch {batch_s}/{num_sample_batches}")
            samples_batch = all_samples[batch_s]
            if correlation_mode == 'Sxyz':
                log_amps_batch = all_log_amps[batch_s]
                szi, sxyi = distributed_Si_fxn_zz_xx_yy(samples_batch, log_amps_batch)
                Sis_batch = szi + sxyi
            else:
                szi = distributed_Si_fxn_zz(samples_batch)
                Sis_batch = szi

            mean_ft_Si, var_ft_Si = calculate_expectation_ft_Si_square(Nx, Sis_batch)
            batch_means_ft_si[batch_s] = mean_ft_Si
            batch_vars_ft_si[batch_s] = var_ft_Si

        Si_allsamples = np.mean(batch_means_ft_si, axis=0)
        var_ft_Si_allsamples = np.mean(batch_vars_ft_si, axis=0) + np.var(batch_means_ft_si, axis=0)
        Sk = 1 / (N_spins) * 3 * Si_allsamples

        if task_id == 0:
            if correlation_mode == 'Sxyz':
                np.save(save_path + corr_final_directory + f'/Sk_from_Si',
                        Sk)
                np.save(save_path + corr_final_directory + f'/var_ft_Si',
                        var_ft_Si_allsamples)
            else:
                np.save(save_path + corr_final_directory + f'/Sk_from_Szi',
                        Sk)
                np.save(save_path + corr_final_directory + f'/var_ft_Szi',
                        var_ft_Si_allsamples)
        print(f"Time for calculating Sk from <Si>'s: {time.time() - timestart}")

        if PRINT:
            print(f"\n Done calculating Sk from <Si>'s... "
                  f"(calculated with {num_samples_final_correlations_estimate} samples)")

    else:
        print(f"\n Already calculated Sk from <Si>'s... "
              f"(calculated with {num_samples_final_correlations_estimate} samples)")

        if correlation_mode == 'Sxyz':
            Sk = np.load(save_path + corr_final_directory + f'/Sk_from_Si.npy')
        else:
            Sk = np.load(save_path + corr_final_directory + f'/Sk_from_Szi.npy')
        print(f"Sk (from <Si>) = {Sk}")


def estimate_(config: dict):
    print("\nEstimating...")

    lattice = config.get('Lattice')

    RNNWF, energy_function, strategy = train_(config)
    Nx = config['Nx']
    Ny = config['Ny']
    N_spins = Nx * Ny

    # Get quantities to estimate
    ENERGY = config.get('ENERGY', False)
    CORRELATIONS_MATRIX = config.get('CORRELATIONS_MATRIX', False)
    Sk_from_Si = config.get('Sk_from_Si', False)

    # Get save path
    data_path_prepend = config.get('data_path_prepend', './data/')
    task_id = config.get('task_id', 0)
    T0 = config['Tmax']
    h_symmetries = config.get('h_symmetries', False)
    l_symmetries = config.get('l_symmetries', False)
    spin_parity = config.get('spin_parity', False)
    train_method = get_train_method(T0, h_symmetries, l_symmetries)
    save_path = data_saver(config, train_method, N_spins, data_path_prepend=data_path_prepend, task_id=task_id)
    print(save_path)

    # Distributed stuff
    print(f"Training strategy obtained from train function")
    print(f"Found stategy: {strategy}")
    print(f"Task ID: {task_id}")
    print(f"Total number of GPUs: {strategy.num_replicas_in_sync}\n")

    # 9. Final Energy Estimate: 
    # ----------------------------------------------------------------------------------------------
    if ENERGY:
        estimate_energy(config, save_path, energy_function, RNNWF.sample,
                        lambda x: RNNWF.log_probsamps(x, symmetrize=l_symmetries, parity=spin_parity), strategy)

    # 10. Final Correlations Matrix: 
    # ----------------------------------------------------------------------------------------------
    if CORRELATIONS_MATRIX:
        estimate_correlations_distributed(config, save_path, RNNWF.sample,
                                          lambda x: RNNWF.log_probsamps(x, symmetrize=l_symmetries, parity=spin_parity),
                                          strategy)

    # 11. structure factor from <Si> 
    # ----------------------------------------------------------------------------------------------
    if Sk_from_Si:
        estimate_Sk_from_Si_distributed(config, save_path, RNNWF.sample,
                                        lambda x: RNNWF.log_probsamps(x, symmetrize=l_symmetries, parity=spin_parity),
                                        strategy)
    return RNNWF


if __name__ == "__main__":
    devices = tf.config.list_logical_devices("CPU")
    print(devices)

    config_test = {
        # Seeding for reproducibility purposes
        'seed': 0,
        'experiment_name': 'testing',

        #### System
        'Hamiltonian': 'AFHeisenberg',
        'Lattice': 'Square',
        'boundary_condition': 'periodic',
        'Apply_MS': True,
        'Nx': 4,  # number of sites in x-direction
        'Ny': 4,  # number of sites in the y-direction

        #### RNN
        'RNN_Type': 'TwoD',
        'units': 16,  # number of memory/hidden units
        'use_complex': False,  # weights shared between RNN cells or not
        'num_samples': 100,  # Batch size
        'lr': 5e-4,  # learning rate
        'gradient_clip': True,
        'tf_dtype': tf.float32,

        #### Annealing
        'scale': 1.,
        'rate': 0.25,
        'Tmax': 0,
        'num_warmup_steps': 1000,  # number of warmup steps 1000 = default (also shouldn't be relevant if Tmax = 0)
        'num_annealing_steps': 1000,  # number of annealing steps
        'num_equilibrium_steps': 5,  # number of gradient steps at each temperature value
        'num_training_steps': 0,  # number of training steps

        #### Symmetries
        'h_symmetries': True,
        'l_symmetries': False,
        'spin_parity': False,

        #### Other
        'CKPT': True,  # whether to save the model during training
        'WRITE': True,  # whether to save training data to file
        'PRINT': True,  # whether to print progress throughout training
        'TRAIN': True,  # whether to train the model

        'strategy': tf.distribute.OneDeviceStrategy(devices[0].name),

        'ENERGY': True,
        'num_samples_final_energy_estimate': 1000,
        'CORRELATIONS_MATRIX': True,
        'Sk_from_Si': True,
        'correlation_mode': 'Sxyz',
        'num_samples_final_correlations_estimate': 10000,
    }

    estimate_(config_test)
