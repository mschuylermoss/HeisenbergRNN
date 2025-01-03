import argparse
import json
import os
import signal
import sys

sys.path.append('..')
import time

# parse arguments
parser = argparse.ArgumentParser(description='Parser for starting multiple runs HPC')
parser.add_argument('--task_id', default=0)
parser.add_argument('--world_size', default=1)
parser.add_argument('--path', default=False, type=bool)

parser.add_argument('--ms', default="1", type=str)
parser.add_argument('--units', default=256, type=int)
parser.add_argument('--experiment_name', default='Nov29', type=str)
parser.add_argument('--bc', default='open', type=str)
parser.add_argument('--num_anneal', default=10000, type=int)
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--rate', default=0.475, type=float)
parser.add_argument('--T0', default=0., type=float)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--use_complex', default="0", type=str)
parser.add_argument('--lsym', default="1", type=str)
parser.add_argument('--schedule', default="rate", type=str)
args = parser.parse_args()

path = bool(args.path)
task_id = int(args.task_id)
world_size = int(args.world_size)

if not path:
    ### DISTRIBUTED ###
    if world_size > 1:
        print(f"Distributed training with multiple nodes")
        # Wait until all jobs are launched
        # Initialize an empty list to store the processed arguments
        processed_args = []
        # Iterate over the arguments and combine flags with their values if necessary
        i = 1  # Start from 1 to skip the script name (sys.argv[0])
        while i < len(sys.argv):
            arg = sys.argv[i]
            # Check if the argument is a flag (starts with "--") and the next one is not another flag
            if arg.startswith("--") and (i + 1 < len(sys.argv)) and not sys.argv[i + 1].startswith("--"):
                # Combine the flag with its value
                if arg not in ['--task_id', '--world_size', '--path']:
                    processed_args.append(f"{arg}={sys.argv[i + 1]}")
                i += 2  # Skip the next argument since it's been combined
            else:
                # Just append the argument as-is
                if not any(x in arg for x in ['--task_id', '--world_size', '--path']):
                    processed_args.append(arg)
                i += 1
        processed_args_str = " ".join(processed_args)
        time.sleep(5)
        # Get all the addresses
        coordinator_addresses = []
        for idx in range(world_size):
            with open(f'./outputs/srun{processed_args_str}_{idx}.out', 'r') as file:
                line_0 = file.readlines()[0].strip()
                coordinator_addresses.append(line_0)
        print(coordinator_addresses)
        print(f"Task ID: {task_id}")

        # Set TF_CONFIG variable
        tf_config = {
            'cluster': {
                'worker': coordinator_addresses,
            },
            'task': {'type': 'worker', 'index': task_id}
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)

        # Now we import tensorflow and initialize the cluster
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print('Waiting for initialization')
        time.sleep(10)
    else:
        import tensorflow as tf

        for dev in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(
                dev, enable=True
            )

        if sys.platform == 'darwin':
            print("Detected MacOS, defaulting to CPU")
            tf.config.set_visible_devices([], "GPU")
        gpu_devices = tf.config.list_logical_devices("GPU")
        cpu_devices = tf.config.list_logical_devices("CPU")
        number_of_available_gpus = len(gpu_devices)
        number_of_available_cpus = len(cpu_devices)
        if number_of_available_gpus > 0:
            print("Using GPU(s)")
            devices = gpu_devices
        else:
            print("Using CPU(s)")
            devices = cpu_devices
        if number_of_available_gpus > 1:
            print(f"Distributed training with {number_of_available_gpus} devices")
            strategy = tf.distribute.MirroredStrategy()
        else:
            print(f"Training with a single device (default)")
            strategy = tf.distribute.OneDeviceStrategy(devices[0].name)

else:
    import tensorflow as tf

    strategy = None

from train import train_
from estimate import estimate_

from utils import LRSchedule_decay, LRSchedule_constant, step_schedule_exp_decay

if __name__ == '__main__':

    experiment_name = args.experiment_name
    num_samples = 100
    number_of_annealing_step = int(args.num_anneal)
    scale = args.scale
    rate = args.rate
    lr_decay_rate = 5000
    units = int(args.units)
    tf_dtype = tf.float32
    bc = args.bc
    MS = (args.ms == "1")
    use_complex = 0 
    data_path_prepend = '/mnt/ceph/users/smoss/HeisenbergRNN'
    l_symmetries = (args.lsym == "1")
    schedule = args.schedule

    config_step1 = {
        # Seeding for reproducibility purposes
        'seed': args.seed,

        #### System
        'Hamiltonian': 'AFHeisenberg',
        'boundary_condition': bc,
        'Apply_MS': MS,
        'Nx': 6,  # number of sites in x-direction
        'Ny': 6,  # number of sites in the y-direction

        #### RNN
        'units': units,  # number of memory/hidden units
        'use_complex': use_complex,  # weights shared between RNN cells or not
        'num_samples': num_samples,  # Batch size
        'lr': LRSchedule_constant(5e-4),  # learning rate
        'gradient_clip': True,
        'tf_dtype': tf_dtype,

        #### Annealing
        'Tmax': args.T0,
        # Highest temperature, if Tmax=0 then its VMC, ***add if statement to skip annealing if Tmax = 0
        'num_warmup_steps': 1000,  # number of warmup steps 1000 = default (also shouldn't be relevant if Tmax = 0)
        'num_annealing_steps': int(scale * number_of_annealing_step),  # number of annealing steps
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
        'experiment_name': experiment_name,  # unique name
        'data_path_prepend': data_path_prepend,
    }

    config_step2 = {
        # Seeding for reproducibility purposes
        'seed': args.seed,
        'previous_config': config_step1,

        #### System
        'Hamiltonian': 'AFHeisenberg',
        'boundary_condition': bc,
        'Apply_MS': MS,
        'Nx': 6,  # number of sites in x-direction
        'Ny': 6,  # number of sites in the y-direction

        #### RNN
        'units': units,  # number of memory/hidden units
        'use_complex': use_complex,  # weights shared between RNN cells or not
        'num_samples': num_samples,  # Batch size
        'lr': LRSchedule_decay(5e-4, 1000 + (5 * int(scale * number_of_annealing_step)),
                                        int(scale * lr_decay_rate)),
        'gradient_clip': True,
        'tf_dtype': tf_dtype,

        #### Annealing
        'Tmax': 0,  # Highest temperature, if Tmax=0 then its VMC, ***add if statement to skip annealing if Tmax = 0
        'num_warmup_steps': 1000,  # number of warmup steps 1000 = default (also shouldn't be relevant if Tmax = 0)
        'num_annealing_steps': int(scale * number_of_annealing_step),  # number of annealing steps
        'num_equilibrium_steps': 5,  # number of gradient steps at each temperature value
        'num_training_steps': int(scale * 25000),  # number of training steps

        #### Symmetries
        'h_symmetries': True,
        'l_symmetries': False,
        'spin_parity': False,

        #### Other
        'CKPT': True,  # whether to save the model during training
        'WRITE': True,  # whether to save training data to file
        'PRINT': True,  # whether to print progress throughout training
        'TRAIN': True,  # whether to train the model
        'experiment_name': experiment_name,  # unique name
        'data_path_prepend': data_path_prepend,
    }

    config_step3 = {
        # Seeding for reproducibility purposes
        'seed': args.seed,
        'previous_config': config_step2,

        #### System
        'Hamiltonian': 'AFHeisenberg',
        'boundary_condition': bc,
        'Apply_MS': MS,
        'Nx': 6,  # number of sites in x-direction
        'Ny': 6,  # number of sites in the y-direction

        #### RNN
        'units': units,  # number of memory/hidden units
        'use_complex': use_complex,  # weights shared between RNN cells or not
        'num_samples': num_samples,  # Batch size
        'lr': LRSchedule_decay(5e-4, 1000 + (5 * int(scale * number_of_annealing_step)),
                                        int(scale * lr_decay_rate)),
        'gradient_clip': True,
        'tf_dtype': tf_dtype,

        #### Annealing
        'Tmax': 0,  # Highest temperature, if Tmax=0 then its VMC, ***add if statement to skip annealing if Tmax = 0
        'num_warmup_steps': 1000,  # number of warmup steps 1000 = default (also shouldn't be relevant if Tmax = 0)
        'num_annealing_steps': int(scale * number_of_annealing_step),  # number of annealing steps
        'num_equilibrium_steps': 5,  # number of gradient steps at each temperature value
        'num_training_steps': int(scale * 50000),  # number of training steps

        #### Symmetries
        'h_symmetries': True,
        'l_symmetries': l_symmetries,
        'spin_parity': False,

        #### Other
        'CKPT': True,  # whether to save the model during training
        'WRITE': True,  # whether to save training data to file
        'PRINT': True,  # whether to print progress throughout training
        'TRAIN': True,  # whether to train the model
        'experiment_name': experiment_name,  # unique name
        'data_path_prepend': data_path_prepend,

        #### Final Estimates
        'ENERGY': True,
        'num_samples_final_energy_estimate': 10000,
        'CORRELATIONS_MATRIX': True,
        'correlation_mode': 'Sxyz',
        'num_samples_final_correlations_estimate': 10000,
    }

    if schedule == 'old':
        configs = [config_step1.copy(), config_step2.copy(), config_step3.copy(), ]
        number_of_steps_list = [90000, 110000, 120000, 125000, 127000, 129000, 131000]
        L_list = [8, 10, 12, 14, 16, 18, 20]
        for _ns, _L in zip(number_of_steps_list, L_list):
            new_conf = configs[-1].copy()
            new_conf["previous_config"] = new_conf.copy()
            new_conf["Nx"] = _L
            new_conf["Ny"] = _L
            new_conf["lr"] = LRSchedule_constant(1e-5)
            new_conf["num_training_steps"] = int(scale * _ns)
            configs.append(new_conf.copy())

    if schedule == 'rate':
        configs = [config_step1.copy(), config_step2.copy(), config_step3.copy(), ]

        L_list = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36]
        for _L in L_list:
            new_conf = configs[-1].copy()
            new_conf["previous_config"] = new_conf.copy()
            new_conf["Nx"] = _L
            new_conf["Ny"] = _L
            new_conf["lr"] = LRSchedule_constant(1e-5 )
            new_conf["num_training_steps"] += step_schedule_exp_decay(_L, scale=scale, rate=rate)
            configs.append(new_conf.copy())
            if _L >= 20: 
                new_conf['CORRELATIONS_MATRIX'] = False

    if path:
        # Only print the path of the last config.
        conf_copy = configs[-1].copy()
        conf_copy['GET_PATH'] = True
        path = train_(conf_copy)
        done_file_path = path + '/DONE.txt'
        print(done_file_path)
        sys.exit(0)
    else:
        def sigterm_handler(signum, frame):
            print("enlarge_triangular.py: Received SIGTERM. Exiting gracefully...")
            # Add any cleanup or finalization code here
            sys.exit(0)


        signal.signal(signal.SIGTERM, sigterm_handler)

        for conf in configs:
            conf['strategy'] = strategy
            conf['task_id'] = task_id
            conf['scale'] = scale
            conf['rate'] = rate
            train_(conf)
            estimate_(conf)
            tf.keras.backend.clear_session()

        print("Python: main(config) is done")
        sys.exit(0)
