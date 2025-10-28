

def get_Si_triangular(L, log_fxn, A_sites, B_sites, C_sites, tf_dtype=tf.float32):

    A_matrix = np.zeros((len(A_sites), L**2))
    B_matrix = np.zeros((len(B_sites), L**2))
    C_matrix = np.zeros((len(C_sites), L**2))
    for n, site in enumerate(A_sites):
        A_matrix[n, site] += 1
    for n, site in enumerate(B_sites):
        B_matrix[n, site] += 1
    for n, site in enumerate(C_sites):
        C_matrix[n, site] += 1
    A_mat_tf = tf.constant(A_matrix,dtype=tf_dtype)
    B_mat_tf = tf.constant(B_matrix,dtype=tf_dtype)
    C_mat_tf = tf.constant(C_matrix,dtype=tf_dtype)
    const_plus = tf.complex(tf.constant((np.sqrt(3) + 1)/2,dtype=tf_dtype),0.0)
    const_minus = tf.complex(tf.constant((np.sqrt(3) - 1)/2,dtype=tf_dtype),0.0)

    def Sxyi_vectorized_A(samples,og_amps):
        N = tf.shape(samples)[1]
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(A_sites), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(A_mat_tf)[tf.newaxis, :, :], 2)
        subtract = samples_tiled_not_flipped - samples_tiled_flipped
        signs = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract, axis=1))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        _, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N))) # (Ns*N,N)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(A_sites))) - og_amps[:, tf.newaxis]) # (Ns, N)
        Si_x = amp_ratio
        Si_y = signs * amp_ratio
        local_Sis = Si_x + Si_y

        return local_Sis # (Ns,)

    def Sxyi_vectorized_B(samples,og_amps):
        N = tf.shape(samples)[1]
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(B_sites), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(B_mat_tf)[tf.newaxis, :, :], 2)
        subtract = samples_tiled_not_flipped - samples_tiled_flipped
        signs = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract, axis=1))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        _, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N))) # (Ns*N,N)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(B_sites))) - og_amps[:, tf.newaxis]) # (Ns, N)
        Si_x = amp_ratio
        Si_y = signs * amp_ratio
        local_Sis = const_minus*Si_x - const_plus*Si_y

        return local_Sis # (Ns,)

    def Sxyi_vectorized_C(samples,og_amps):
        N = tf.shape(samples)[1]
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(C_sites), axis=2)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(C_mat_tf)[tf.newaxis, :, :], 2)
        subtract = samples_tiled_not_flipped - samples_tiled_flipped
        signs = tf.complex(tf.cast(0.0, dtype=tf_dtype), tf.reduce_sum(subtract, axis=1))
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
        _, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N))) # (Ns*N,N)
        amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(C_sites))) - og_amps[:, tf.newaxis]) # (Ns, N)
        Si_x = amp_ratio
        Si_y = signs * amp_ratio
        local_Sis = -1*const_plus*Si_x + const_minus*Si_y

        return local_Sis # (Ns,)

    def Szi_vectorized_A(samples):

        N = tf.shape(samples)[1]
        Si_z_real = A_mat_tf @ (2 * tf.transpose(samples) - 1)
        Si_z = tf.complex(Si_z_real,tf.cast(0,dtype=tf_dtype))
        local_Sis = tf.transpose(Si_z)
    
        return local_Sis # (Ns,)

    def Szi_vectorized_B(samples):

        N = tf.shape(samples)[1]
        Si_z_real = B_mat_tf @ (2 * tf.transpose(samples) - 1)
        Si_z = tf.complex(Si_z_real,tf.cast(0,dtype=tf_dtype))
        local_Sis = tf.transpose(Si_z)
    
        return local_Sis # (Ns,)

    def Szi_vectorized_C(samples):

        N = tf.shape(samples)[1]
        Si_z_real = C_mat_tf @ (2 * tf.transpose(samples) - 1)
        Si_z = tf.complex(Si_z_real,tf.cast(0,dtype=tf_dtype))
        local_Sis = tf.transpose(Si_z)
    
        return local_Sis # (Ns,)

    return Sxyi_vectorized_A,Sxyi_vectorized_B,Sxyi_vectorized_C,Szi_vectorized_A,Szi_vectorized_B,Szi_vectorized_C

def estimate_total_spin(config, save_path, sample_fxn, log_fxn, strategy):
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
    bc = config['boundary_condition']
    if bc == 'periodic':
        periodic = True
    else:
        periodic = False

    if PRINT:
        print(f"\nCalculating <S> with {num_samples_final_correlations_estimate} samples")
        print(f"Batch size = {batch_size_samples} ({num_sample_batches} batches)")
        print(f"{number_of_replicas} devices, so batch size per device: {batch_size_per_device}")

    corr_final_directory = f'/final_correlations/ns_{num_samples_final_correlations_estimate}'
    if not os.path.exists(save_path + corr_final_directory) and task_id == 0:
        os.makedirs(save_path + corr_final_directory)

    A_sites,B_sites,C_sites,_ = generate_sublattices_triangular(Nx,Ny)
    Sxy_A, Sxy_B, Sxy_C, Sz_A, Sz_B, Sz_C = get_Si_triangular(Nx, log_fxn, A_sites, B_sites, C_sites, tf_dtype=tf.float32)

    all_samples = load_dict(save_path + corr_final_directory + '/final_rcorrelations_samples.pkl')
    all_log_amps = load_dict(save_path + corr_final_directory + '/final_rcorrelations_logamps.pkl')

    @tf.function()
    def distributed_total_spin_fxn(samples, logamps, Sz_fxn, Sxy_fxn):
        print("Tracing distributed Si function")

        @tf.function()
        def value_fn(ctx):
            samples_chunked = tf.reshape(samples, (number_of_replicas, batch_size_per_device, N_spins))
            logamps_chunked = tf.reshape(logamps, (number_of_replicas, batch_size_per_device))
            return samples_chunked[ctx.replica_id_in_sync_group], logamps_chunked[ctx.replica_id_in_sync_group]

        samples_distributed, logpsis_distributed = strategy.experimental_distribute_values_from_function(value_fn)
        zz = strategy.run(lambda _x: tf.math.real(Sz_fxn(_x)), args=(samples_distributed,))
        xx_yy = strategy.run(lambda _x, _y: tf.math.real(Sxy_fxn(_x, _y)),
                             args=(samples_distributed, logpsis_distributed,))
        return strategy.gather(zz, axis=0), strategy.gather(xx_yy, axis=0)    

    total_spins = np.zeros((num_samples_final_correlations_estimate,N_spins))
    for batch_s in range(num_sample_batches):
        print(f"sample batch {batch_s}/{num_sample_batches}")
        samples_batch = all_samples[batch_s]
        log_amps_batch = all_log_amps[batch_s]
        Sz_a_batch, Sxy_a_batch = distributed_total_spin_fxn(samples_batch, log_amps_batch, Sz_A, Sxy_A)
        Sz_b_batch, Sxy_b_batch = distributed_total_spin_fxn(samples_batch, log_amps_batch, Sz_B, Sxy_B)
        Sz_c_batch, Sxy_c_batch = distributed_total_spin_fxn(samples_batch, log_amps_batch, Sz_C, Sxy_C)
        total_spins[batch_s*100:(batch_s+1)*100,A_sites] = Sz_a_batch+Sxy_a_batch
        total_spins[batch_s*100:(batch_s+1)*100,B_sites] = Sz_b_batch+Sxy_b_batch
        total_spins[batch_s*100:(batch_s+1)*100,C_sites] = Sz_c_batch+Sxy_c_batch
        # total_spins.append((Sz_a_batch+Sxy_a_batch))
        # total_spins.append((Sz_b_batch+Sxy_b_batch))
        # total_spins.append((Sz_c_batch+Sxy_c_batch))
    
    avg_total_spin = np.mean(np.sum(total_spins,axis=1)**2)
    var_total_spin = np.var(np.sum(total_spins,axis=1)**2)

    if task_id == 0:
        np.save(save_path + corr_final_directory + f'/avg_total_spin',
                avg_total_spin)
        np.save(save_path + corr_final_directory + f'/var_total_spin',
                var_total_spin)

    if PRINT:
        print(f"\n Total average spin <S^2> = {avg_total_spin}")
