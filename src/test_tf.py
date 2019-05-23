import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer


def get_times(maximum_time):

    matrix_sizes = []
    device_times = {
        "/cpu:0":[],
        "/gpu:0":[]
    }
    data_type = tf.float32

    for size in range(1000, 50000, 100):
        values = np.random.randn(size, size).astype('float32')
        shape = (size, size)

        stop = False
        for device_name in device_times.keys():
            print("####### Calculating on the " + device_name + " #######")

            with tf.device(device_name):
                r1 = tf.random_uniform(shape=shape, dtype=data_type)
                r2 = tf.random_uniform(shape=shape, dtype=data_type)
                r3 = tf.placeholder(shape=shape, dtype=data_type)
                mm = tf.matmul(r2, r1)
                mm = tf.matmul(mm, r3)
                o = tf.reduce_mean(mm)

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
                start_time = timer()
                result = session.run(o, feed_dict={ r3: values })
                time_taken = timer() - start_time
                print('Result:', result)
                device_times[device_name].append(time_taken)
                if time_taken > maximum_time:
                    stop = True

        matrix_sizes.append(size)
        if stop:
            return device_times, matrix_sizes


def main():

    device_times, matrix_sizes = get_times(1)
    gpu_times = device_times["/gpu:0"]
    cpu_times = device_times["/cpu:0"]

    print(device_times)
    plt.plot(matrix_sizes[:len(gpu_times)], gpu_times, 'o-', label='GPU')
    plt.plot(matrix_sizes[:len(cpu_times)], cpu_times, 'o-', label='CPU')
    plt.ylabel('Time')
    plt.xlabel('Matrix size')
    plt.legend()
    plt.show()