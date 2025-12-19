import sys

import tensorflow as tf

import horovod
import horovod.tensorflow.keras as hvd

def main():
    # Horovod: initialize Horovod.
    hvd.init()

    tf.config.optimizer.set_jit(False) # disable XLA jit compiler

    # 配置 Grappler 優化器（禁用有問題的優化）
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': False,          # 解決 NHWC/NCHW 轉換問題
        'constant_folding': True,           # 保留有用的優化
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': False,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': False,
    })
    
    # 減少日誌輸出
    tf.get_logger().setLevel('ERROR')

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(10000).batch(128)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = 0.001 * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#    if hvd.rank() == 0:
#        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    mnist_model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        # run training through horovod.run
        np = int(sys.argv[1])
        hosts = sys.argv[2]
        comm = sys.argv[3]
        print('Running training through horovod.run')
        horovod.run(main, np=np, hosts=hosts, use_gloo=comm == 'gloo', use_mpi=comm == 'mpi')
    else:
        # this is running via horovodrun
        main()
