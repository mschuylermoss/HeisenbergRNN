import tensorflow as tf
import numpy as np
import os


class LRSchedule_constant(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.learning_rate


class LRSchedule_decay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, step_start=0, halfrate=5000):
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.halfrate = halfrate
        self.step_start = step_start

    def __call__(self, step):
        self.learning_rate = self.initial_learning_rate * (1. / (1. + (step - self.step_start) / self.halfrate))
        return self.learning_rate


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
    hamiltonian = config['Hamiltonian'] + \
                  f'{"_periodic/" if config.get("boundary_condition", "open") == "periodic" else ""}' \
                  + f'/{config["Lattice"]}{"_no_MS" if not config["Apply_MS"] else ""}{"_triMS" if config.get("tri_MS",False) else ""}/'
    datapath = f'{data_path_prepend}/{hamiltonian}'
    experiment_name = config['experiment_name']
    size = f'/N_{N}'
    train_method = f'/{train_method}'
    weight_sharing = config.get("weight_sharing", 'all')
    weight_sharing_method = f'_{weight_sharing}' if not weight_sharing == 'all' else ''
    testing_params = f"/nh{config['units']}_scale{config['scale']}_rate{config['rate']}{weight_sharing_method}"
    # testing_params = f"/nh{config['units']}_na{config['num_annealing_steps']}{weight_sharing_method}"
    save_path = datapath + experiment_name + testing_params + size + train_method + f'/seed_{config["seed"]}'

    # If multiple workers are active, only make directories in node 0
    if task_id == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + '/config.txt', 'w') as file:
            for k, v in config.items():
                file.write(k + f'={v}\n')
    elif task_id > 0:
        if not os.path.exists(save_path + f'/tmp_{task_id}/'):
            os.makedirs(save_path + f'/tmp_{task_id}/')
    return save_path

def optimizer_initializer(optimizer, strategy):
    with strategy.scope():
        fake_var = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        fake_loss = tf.reduce_sum(fake_var ** 2)
    grads = tape.gradient(fake_loss, [fake_var])
    # Ask the optimizer to apply the processed gradients.
    optimizer.apply_gradients(zip(grads, [fake_var]))

def sync_function(strategy):
    @tf.function()
    def sync():
        @tf.function()
        def tf_sync_function():
            return tf.constant([1.,], tf.float32)
        values = strategy.run(tf_sync_function)
        return strategy.gather(values, axis=0)
    return sync


def step_schedule_exp_decay(L, scale=1., rate=0.5):
    return int((1000 + scale * 100000) * np.exp(-(L - 6) * rate) + (scale * 2000))