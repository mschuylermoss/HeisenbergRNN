import os

import numpy as np
import tensorflow as tf


class LRSchedule_constant(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, dtype=tf.float32):
        self.initial_learning_rate = tf.constant(initial_learning_rate, dtype=dtype)

    def __call__(self, step):
        return self.initial_learning_rate


class LRSchedule_decay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, step_start=0, halfrate=5000, dtype=tf.float32):
        self.initial_learning_rate = tf.constant(initial_learning_rate, dtype=dtype)
        self.halfrate = tf.cast(halfrate, dtype=dtype)
        self.step_start = tf.cast(step_start, dtype=dtype)

    def __call__(self, step):
        global_step = tf.cast(step, dtype=self.initial_learning_rate.dtype)
        return self.initial_learning_rate * (1. / (1. + (global_step - self.step_start) / self.halfrate))


def get_train_method(T0: float, h_symmetries: bool, l_symmetries: bool) -> str:
    if np.isclose(T0, 0.0):
        train_method = 'No_Annealing'
    else:
        train_method = 'Annealing'

    if h_symmetries or l_symmetries:
        if h_symmetries:
            train_method = train_method + '_h'
        if l_symmetries:
            train_method = train_method + '_l'
        train_method = train_method + '_Symmetries'
    else:
        train_method = train_method + '_NoSymmetries'
    return train_method


def data_saver(config, train_method: str, N: int, data_path_prepend, task_id=0):
    hamiltonian = config['Hamiltonian']
    lattice = f'Square{"_no_MS" if not config["Apply_MS"] else ""}' \
              + f'{"/periodic" if config.get("boundary_condition", "open") == "periodic" else "/open"}'
    datapath = f'{data_path_prepend}/{hamiltonian}/{lattice}/'
    experiment_name = config['experiment_name']
    size = f'/N_{N}'
    train_method = f'/{train_method}'
    testing_params = f"/nh{config['units']}_scale{config['scale']}_rate{config['rate']}"
    save_path = datapath + experiment_name + testing_params + size + train_method + f'/seed_{config["seed"]}'

    # If multiple workers are active, only make directories in node 0
    if task_id == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + '/config.txt'):
            with open(save_path + '/config.txt', 'w') as file:
                for k, v in config.items():
                    file.write(k + f'={v}\n')
    elif task_id > 0:
        if not os.path.exists(save_path + f'/tmp_{task_id}/'):
            os.makedirs(save_path + f'/tmp_{task_id}/')
    return save_path


def sync_function(strategy):
    @tf.function()
    def sync():
        @tf.function()
        def tf_sync_function():
            return tf.constant([1., ], tf.float32)

        values = strategy.run(tf_sync_function)
        return strategy.gather(values, axis=0)

    return sync


def step_schedule_exp_decay(L, scale=1., rate=0.5):
    return int((1000 + scale * 100000) * np.exp(-(L - 6) * rate) + (scale * 2000))
