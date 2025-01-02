import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

import numpy as np

from utils import data_saver, sync_function, get_train_method
from RNN import get_rnn_cell, cMDRNNWavefunction, periodic_cMDRNNWavefunction
from interactions import buildlattice_square
from energy import get_Heisenberg_Energy_Vectorized_square
from logger import Logger

import errno
import shutil
import os
import time
import datetime
import signal
import sys


def train_(config: dict):
    #### Other ####
    CKPT = config['CKPT']
    WRITE = config['WRITE']
    PRINT = config['PRINT']
    TRAIN = config['TRAIN']
    GET_PATH = config.get('GET_PATH', False)

    # PATHS
    data_path_prepend = config.get('data_path_prepend', './data/')
    strategy = config.get('strategy', None)
    if not GET_PATH and strategy is None:
        raise ValueError("Must pass a tensorflow strategy to train.py")
    task_id = config.get('task_id', 0)
    gpu_devices = tf.config.list_logical_devices("GPU")
    cpu_devices = tf.config.list_logical_devices("CPU")
    number_of_available_gpus = len(gpu_devices)
    number_of_available_cpus = len(cpu_devices)
    XLA = True if os.environ.get("XLA", "0") == "1" else False
    if XLA:
        xla_flags = os.environ.get("XLA_FLAGS", None)
        assert xla_flags is not None, 'XLA_FLAGS environment variable not set, use \n' \
                                      '`export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda`\n' \
                                      'before launching program to enable XLA.'

    if PRINT:
        print("_" * 25)
        if number_of_available_gpus > 0:
            if PRINT:
                print("Using GPU(s)")
            devices = gpu_devices
        else:
            if PRINT:
                print("Using CPU(s)")
            devices = cpu_devices
        print(f"{number_of_available_cpus} CPUs found")
        print(f"{number_of_available_gpus} GPUs found")
        print(f"XLA enabled?:  {XLA}")

    # 1. Set all parameters:
    # ----------------------------------------------------------------------------------------------
    # Seeding for reproducibility purposes
    seed = config['seed'] + task_id * 10000
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.random.set_seed(seed)  # tensorflow pseudo-random generator

    #### System Parameters ####
    Hamiltonian = config['Hamiltonian']
    boundary_condition = config.get('boundary_condition', 'open')
    Apply_MS = config['Apply_MS']
    Nx = config['Nx']
    Ny = config['Ny']
    N_sites = Nx * Ny
    N_spins = N_sites

    if (Hamiltonian == 'AFHeisenberg'):
        J = +1
    elif (Hamiltonian == 'FMHeisenberg'):
        J = -1
    else:
        raise ValueError(f'{Hamiltonian} not available.')

    #### RNN Parameters ####
    RNN_cell = config.get('RNN_cell', 'MDGRU' if boundary_condition == 'open' else 'MDPeriodic')
    available_RNN_cells = ['MDPeriodic', 'MDGRU']
    assert RNN_cell in available_RNN_cells, f'{RNN_cell} is not a valid RNN_cell, choose one of {available_RNN_cells}'
    units = config['units']
    use_complex = config.get('use_complex', False)
    num_samples = config['num_samples']
    lr = config.get('lr', 5e-4)
    gradient_clip = config.get('gradient_clip', False)
    tf_dtype = config.get('tf_dtype', tf.float32)
    assert isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule), \
        '`lr` to be an instance of `tf.keras.optimizers.schedules.LearningRateSchedule`, ' \
        f'received {lr}'
    kernel_initializer = config.get('kernel_initializer', 'glorot_uniform')

    #### Annealing ####
    T0 = config['Tmax']
    num_warmup_steps = config['num_warmup_steps']
    num_annealing_steps = config['num_annealing_steps']
    num_equilibrium_steps = config['num_equilibrium_steps']
    num_training_steps = config['num_training_steps']
    num_steps = num_annealing_steps * num_equilibrium_steps + num_warmup_steps + num_training_steps

    #### Symmetries ####
    h_symmetries = config['h_symmetries']
    l_symmetries = config['l_symmetries']
    spin_parity = config['spin_parity']

    # 2. Set data path:
    # ----------------------------------------------------------------------------------------------
    train_method = get_train_method(T0, h_symmetries, l_symmetries)

    if GET_PATH:
        save_path = data_saver(config, train_method, N_spins, data_path_prepend=data_path_prepend, task_id=-1)
        return save_path
    else:
        save_path = data_saver(config, train_method, N_spins, data_path_prepend=data_path_prepend, task_id=task_id)

    sync = int(sum(sync_function(strategy)().numpy().tolist()))
    print(f"\nSync {sync} devices")

    previous_config = config.get('previous_config', None)
    if previous_config is not None:
        scale_ = previous_config.get('scale', None)
        rate_ = previous_config.get('rate', None)
        if scale_ == None:
            previous_config['scale'] = config['scale']
        if rate_ == None:
            previous_config['rate'] = config['rate']

        previous_N_sites = previous_config['Nx'] * previous_config['Ny']
        previous_N_spins = previous_N_sites
        previous_train_method = get_train_method(previous_config['Tmax'],
                                                 previous_config['h_symmetries'],
                                                 previous_config['l_symmetries'])
        old_save_path = data_saver(previous_config, previous_train_method, previous_N_spins, data_path_prepend)
        files = os.listdir(old_save_path)
        files_to_copy = filter(lambda x: (not os.path.isdir(old_save_path + f'/{x}')) and
                                         (x not in ['config.txt', 'DONE.txt']), files)
        old_checkpoint_time = os.path.getmtime(old_save_path + '/checkpoint')

        if os.path.exists(save_path + '/checkpoint'):
            new_checkpoint_time = os.path.getmtime(save_path + '/checkpoint')
            print(f"Checkpoint found! Times:\n"
                  f"Old ckpt: {time.ctime(old_checkpoint_time)}\n"
                  f"New ckpt: {time.ctime(new_checkpoint_time)}")
        else:
            new_checkpoint_time = -1
        if old_checkpoint_time > new_checkpoint_time and (task_id == 0):
            if task_id == 0:
                for file in files_to_copy:
                    shutil.copy(old_save_path + '/' + file, save_path + '/' + file)
                print(f"Copied old checkpoint data from {old_save_path} to {save_path}")
        else:
            print("Continuing from newest checkpoint!")

    ckpt_path = save_path
    sync = int(sum(sync_function(strategy)().numpy().tolist()))
    print(f"Sync {sync} devices\n")

    if task_id > 0:  # if we have multiple workers, create temporary checkpoint paths
        files = os.listdir(save_path)
        files_to_copy = filter(lambda x: (not os.path.isdir(save_path + f'/{x}')) and
                                         (x not in ['config.txt', 'DONE.txt']), files)
        for file in files_to_copy:
            if file == 'checkpoint':
                shutil.copy(save_path + '/' + file, save_path + f'/tmp_{task_id}/' + file)
            elif 'ckpt' in file:
                shutil.copy(save_path + '/' + file, save_path + f'/tmp_{task_id}/' + file)
        ckpt_path = save_path + f'/tmp_{task_id}/'

    sync = int(sum(sync_function(strategy)().numpy().tolist()))
    print(f"Sync {sync} devices")

    # 3. Print relevant information:
    # ----------------------------------------------------------------------------------------------
    print(f"Tensorflow version {tf.__version__}")
    print(f"Boundary condition = {boundary_condition}")
    print("Number of spins =", N_spins)
    print(f"Using complex?? {use_complex}")
    print(f"Number of hidden num_units = {units}")
    print("Initial_temperature =", T0)
    if np.isclose(T0, 0.0):
        print("No annealing")
    else:
        print("Annealing")
        print(f"Number of warmup steps = {num_warmup_steps}")
        print(f"Number of annealing steps = {num_annealing_steps}")

    if h_symmetries:
        print("Exploiting Hamiltonian Symmetries")
    elif l_symmetries:
        print("Exploiting Lattice Symmetries")
    else:
        print("Symmetries not enforced during training")

    print(f"Total number of training steps = {num_steps}")
    print('Seed = ', seed)

    # 4. Initialize RNN wave function:
    # ----------------------------------------------------------------------------------------------

    local_hilbert_space_size = int(2)
    print(f"Size of the local hilbert space of each RNN cell = {local_hilbert_space_size}")

    RNN_cell_fun = get_rnn_cell(RNN_cell)
    print(f'RNN_cell = {RNN_cell_fun}\n')
    number_of_nodes = strategy.num_replicas_in_sync // len(devices)
    number_of_replicas = strategy.num_replicas_in_sync
    print(f"Number of nodes: {number_of_nodes}")
    print(f"Number of GPUs per node: {number_of_available_gpus}")
    print(f"Total number of GPUs: {number_of_replicas}")
    num_samples_per_device = int(np.ceil(num_samples / number_of_replicas))
    print(f"Number of samples per device: {num_samples_per_device}\n")

    RNN_x = Nx
    RNN_y = Ny

    # The model has to be created within the strategy scope
    with strategy.scope():
        if boundary_condition == 'open':
            RNNWF_fun = cMDRNNWavefunction
        else:
            RNNWF_fun = periodic_cMDRNNWavefunction
        RNNWF = RNNWF_fun(cell=RNN_cell_fun,
                          systemsize_x=RNN_x, systemsize_y=RNN_y, units=units,
                          local_hilbert_space=local_hilbert_space_size, seed=seed,
                          use_complex=use_complex,
                          h_symmetries=h_symmetries,
                          kernel_initializer=kernel_initializer,
                          tf_dtype=tf_dtype)

        test_sample = RNNWF.sample(2)
        test_logpsi = RNNWF.log_probsamps(test_sample, symmetrize=False, parity=False)

    trainable_variables = []
    trainable_variables.extend(RNNWF.rnn.trainable_variables)
    trainable_variables.extend(RNNWF.dense.trainable_variables)
    if use_complex:
        trainable_variables.extend(RNNWF.dense_phase.trainable_variables)
    print("Weights are shared")

    variables_names = [v.name for v in trainable_variables]
    variable_sum = 0
    for k, v in zip(variables_names, trainable_variables):
        v1 = tf.reshape(v, [-1])
        print(k, v1.shape)
        variable_sum += v1.shape[0]
    print('The sum of params is {0}'.format(variable_sum))

    # Creating the optimizer
    # The optimizer has to be created within the strategy scope
    with strategy.scope():
        print(f'Learning rate schedule {lr}')
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, )

    if gradient_clip:
        print('Gradient clipping is enabled.')
    print(f'Datatype for RNN {tf_dtype}')

    # Creating training variables
    # Variables have to be created within the strategy scope
    with strategy.scope():
        T = tf.Variable(T0, name="temperature", dtype=tf_dtype)  # initializing temperature
    global_step = tf.Variable(1, name="global_step")

    # 5. Generate list of lattice interactions:
    # ----------------------------------------------------------------------------------------------
    interactions_square = buildlattice_square(Nx, Ny, bc=boundary_condition)

    # 6. Define the training step:
    # ----------------------------------------------------------------------------------------------
    Heisenberg_Energy = get_Heisenberg_Energy_Vectorized_square(J, interactions_square, RNNWF.log_probsamps,
                                                                marshall_sign=Apply_MS,
                                                                symmetrize=l_symmetries, parity=spin_parity,
                                                                tf_dtype=tf_dtype)

    if TRAIN:
        @tf.function()
        def single_train_step_vmc():
            print(f"Tracing single train step")

            samples_tf = RNNWF.sample(num_samples_per_device)
            with tf.GradientTape() as tape:
                log_probs_tf, log_amps_tf = RNNWF.log_probsamps(samples_tf, symmetrize=l_symmetries,
                                                                parity=spin_parity)
                local_energies_tf = tf.stop_gradient(Heisenberg_Energy(samples_tf, log_amps_tf))
                cost_term_1 = tf.reduce_mean(
                    tf.multiply(tf.math.conj(log_amps_tf), local_energies_tf))
                cost_term_2 = tf.reduce_mean(tf.math.conj(log_amps_tf)) * tf.reduce_mean(
                    local_energies_tf)
                cost = 2 * tf.math.real(cost_term_1 - cost_term_2)

            # Crucial to divide by the number of replicas since apply_gradients sums gradients over replicas
            gradients = tape.gradient(cost, trainable_variables)
            if gradient_clip:
                gradients = [tf.clip_by_value(g, -10., 10.) for g in gradients]
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            local_energies_tf = tf.stack([tf.math.real(local_energies_tf),
                                          tf.math.imag(local_energies_tf)], axis=-1)

            return cost, local_energies_tf, log_probs_tf

        @tf.function()
        def distributed_train_step_vmc():
            print("Tracing distributed train step")
            cost_tf_per_rep, local_energies_tf_per_rep, log_probs_tf_per_rep = strategy.run(single_train_step_vmc)
            global_step.assign_add(1)

            local_energies_tf_per_rep_reduced = strategy.gather(local_energies_tf_per_rep,
                                                                axis=0)
            local_energies_tf_per_rep_total = tf.complex(local_energies_tf_per_rep_reduced[:, 0],
                                                         local_energies_tf_per_rep_reduced[:, 1])
            return strategy.reduce(tf.distribute.ReduceOp.SUM, cost_tf_per_rep, axis=None), \
                local_energies_tf_per_rep_total, \
                strategy.gather(log_probs_tf_per_rep, axis=0)

    # 7. Start from checkpoint or from scratch:
    # ----------------------------------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, variables=trainable_variables,
                               temperature=T)

    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
    logger = Logger(save_path)

    def sigterm_handler(signum, frame):
        print("train_distributed.py: Received SIGUSR2. Exiting gracefully...")
        print(f"train_distributed.py: Saving checkpoint...")
        # if task_id == 0:
        #     logger.save()
        #     manager.save()
        print(f"train_distributed.py: checkpoint successful")
        time.sleep(1)
        print(f"train_distributed.py: Sending SIGTERM...")
        signal.raise_signal(signal.SIGTERM)
        sys.exit(0)

    signal.signal(signal.SIGUSR2, sigterm_handler)

    if CKPT:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
            print(f"Restored learning rate is {optimizer.learning_rate}.")
            print(f"Continuing at step {ckpt.step.numpy()} with temperature T = {ckpt.temperature.numpy()}")
            assert logger.restore(), 'Failed to load data...'
            time.sleep(1)

        else:
            print("Ckpt ON. Initializing from scratch.")

    else:
        print("Ckpt OFF. Initializing from scratch.")

    # 8. Train model:
    # ----------------------------------------------------------------------------------------------
    if TRAIN:
        # Training loop
        while True:
            it = global_step.numpy()
            if (it - 1) >= num_steps:
                if not os.path.exists(save_path + '/DONE.txt') and (task_id == 0):
                    with open(save_path + '/DONE.txt', 'w') as file:
                        file.write(f'Completed on {datetime.datetime.today()}')
                if task_id > 0:  # if we have multiple workers, clean up temporary paths
                    for attempt in range(1, 4):
                        try:
                            if os.path.exists(ckpt_path):
                                shutil.rmtree(ckpt_path)
                                print(f"Deleted temporary path {ckpt_path}")
                                break
                        except OSError as e:
                            if e.errno == errno.EBUSY:
                                print(
                                    f"Attempt {attempt}/{3}: Cannot delete temporary path {ckpt_path}. Trying again after 1 sec.")
                                time.sleep(0.5)
                            else:
                                raise
                break

            if it <= num_warmup_steps:
                if PRINT and int(it) % 100 == 0:
                    print(f"\nTraining step: {it}/{num_steps}")
                    print(f"Warming up at T={T.numpy()}")
            elif num_warmup_steps < it <= (num_annealing_steps * num_equilibrium_steps + num_warmup_steps):
                if PRINT and int(it) % 100 == 0:
                    print(f"\nTraining step: {it}/{num_steps}")
                    print(f"Annealing at T={T.numpy()}")
                if it % num_equilibrium_steps == 0:
                    annealing_step = (it - num_warmup_steps) / num_equilibrium_steps
                    T.assign(T0 * (1 - annealing_step / num_annealing_steps))
            else:
                if PRINT and int(it) % 100 == 0:
                    print(f"\nTraining step: {it}/{num_steps}")
                    print("Standard training")

            start = time.time()
            if it==1:
                trace_time = time.time()
            cost, local_energies, log_probs = distributed_train_step_vmc()
            if it==1:
                print("Trace time:", time.time() - trace_time)
            print(f"Step {it}")
            time_per_step = time.time() - start
            cost_np = cost.numpy()
            local_energies_np = local_energies.numpy()
            log_probs_np = log_probs.numpy().astype(np.complex64)
            T_np = T.numpy()

            if WRITE:
                logger(T_np, local_energies_np, log_probs_np, cost_np,
                       time_per_step=np.array([time_per_step, number_of_replicas]))

            if PRINT and (int(it) % 100 == 0):
                print(f'mean(E): {logger.data["meanEnergy"][-1]}')
                print(f'var(RE(E)): {logger.data["RE_varEnergy"][-1]}, '
                      f'var(IM(E)): {logger.data["IM_varEnergy"][-1]}')
                print(f'mean(F): {logger.data["meanFreeEnergy"][-1]}')
                print(f'var(RE(F)): {logger.data["RE_varFreeEnergy"][-1]}, '
                      f'var(IM(F)): {logger.data["IM_varFreeEnergy"][-1]}')
                print(f'#samples {num_samples}, #Training step {it}')
                print("Temperature: ", T_np)
                print(f"Time per step: {time_per_step}")

            if CKPT and (int(it) % 100 == 0):
                if np.isnan(logger.data["meanEnergy"][-1]):
                    print("NaN detected, not saving checkpoint!")
                    sys.exit(1)
                manager.save()
                print(f"Saved checkpoint for step {int(it)}: {ckpt_path}")
                if task_id == 0:
                    logger.save()

    return RNNWF, Heisenberg_Energy, strategy

