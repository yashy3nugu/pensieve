import threading
from time import gmtime, sleep, strftime, time
import load_trace
from a3c_gs import ActorNetwork, CriticNetwork, compute_gradients, compute_entropy
from run_tests_vmaf import run_tests
import env
import tensorflow as tf
import os
import logging
import numpy as np
import multiprocessing as mp
from Queue import Queue
import pickle
import gc
# from rl_test_gs import run_tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 7
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [20000, 40000, 60000, 80000, 110000, 160000]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 160  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results_gs'
LOG_FILE = './results_gs/log'
TEST_LOG_FOLDER = './test_results_gs/'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None
GLOBAL_WORKERS = 4
NUM_WORKERS = 2
EPOCHS = 100000
ENTROPY_WEIGHT = 5
VMAF = './envivo/vmaf/video'


class Worker():
    def __init__(self, global_assignment, name, all_cooked_time, all_cooked_bw, saver_thread, queue, global_actor):
        self.name = str(name)+"_worker_" + str(global_assignment)
        self.global_assignment = self.name[-1]  # get global assignment index
        self.local_actor = ActorNetwork(state_dim=[
                                        S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, global_workers=GLOBAL_WORKERS, scope="actor_" + self.name, entropy_weight=ENTROPY_WEIGHT)
        self.local_critic = CriticNetwork(state_dim=[
                                          S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE, global_workers=GLOBAL_WORKERS, scope="critic_" + self.name)
        self.env = env.Environment(
            all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)
        self.saver_thread = saver_thread

        self.global_actor = global_actor

        if self.saver_thread:
            self.queue = queue

    def train(self, sess, actor_gradient, critic_gradient):
        self.local_actor.apply_gradients(sess, actor_gradient)
        self.local_critic.apply_gradients(sess, critic_gradient)

        self.other_ids = list(range(GLOBAL_WORKERS))
        del self.other_ids[int(self.global_assignment)]

        self.actor_block_stats = self.local_actor.get_block_vars(sess)
        self.valid_actor_updates = list(np.array(self.actor_block_stats))

        # loop until all locks are de-activated
        while not all([v == True for v in self.valid_actor_updates]):
            self.actor_block_stats = self.local_actor.get_block_vars(sess)
            self.valid_actor_updates = list(np.array(self.actor_block_stats))

        feed_dict_actor = {k: v for (k, v) in zip(
            self.local_actor.feed_gradients, actor_gradient)}
        feed_dict_actor[self.local_actor.lr_placeholder] = self.local_actor.lr_rate
        feed_dict_actor[self.local_actor.entropy_weight_placeholder] = self.local_actor.entropy_weight

        feed_dict_critic = {k: v for (k, v) in zip(
            self.local_critic.feed_gradients, critic_gradient)}
        feed_dict_critic[self.local_critic.lr_placeholder] = self.local_critic.lr_rate

        for i in range(GLOBAL_WORKERS):
            # activate all locks
            sess.run(self.local_actor.block_global[int(i)])
        for i in range(len(self.other_ids)):
            sess.run(self.local_actor.apply_other_grads[int(
                i)], feed_dict=feed_dict_actor)
            sess.run(self.local_critic.apply_other_grads[int(
                i)], feed_dict=feed_dict_critic)
        for i in range(GLOBAL_WORKERS):
            # de-activate all locks
            sess.run(self.local_actor.unblock_global[int(i)])

    def set_entropy(self, epoch):
        if epoch == 10000:
            self.local_actor.entropy_weight = 4

        if epoch == 20000:
            self.local_actor.entropy_weight = 2
        if epoch == 30000:
            self.local_actor.entropy_weight = 1

        if epoch == 40000:
            self.local_actor.entropy_weight = 0.5
        if epoch == 50000:
            self.local_actor.entropy_weight = 0.3

        if epoch == 60000:
            self.local_actor.entropy_weight = 0.05

        if epoch == 70000:
            self.local_actor.entropy_weight = 0.01

    def work(self, sess):
        print('started worker ' + str(self.global_assignment))

        start = time()

        test_log_file_path = LOG_FILE + '_test_' + str(self.global_assignment)

        with sess.as_default(), sess.graph.as_default(), open(LOG_FILE + self.name, 'wb') as log_file, open(test_log_file_path, 'wb') as test_log_file:
            vmaf_size = {}
            for bitrate in range(A_DIM):
                
                vmaf_size[bitrate] = []
                
                with open(VMAF + str(A_DIM - bitrate)) as f:
                    for line in f:
                        vmaf_size[bitrate].append(float(line))
            self.local_actor.transfer_global_params(sess)
            self.local_critic.transfer_global_params(sess)

            epoch = 0
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            last_chunk_vmaf = None

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [action_vec]
            r_batch = []
            entropy_record = []

            time_stamp = 0
            while epoch < EPOCHS:  # experience video streaming forever

                if epoch % 1000 == 0:
                    print(self.name + " in epoch " + str(epoch))

                self.set_entropy(epoch)

                # the action is from the last decision
                # this is to make the framework similar to the real
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain, video_chunk_vmaf = \
                    self.env.get_video_chunk(bit_rate)
                
                next_video_chunk_vmaf = []
                for i in range(A_DIM):
                    next_video_chunk_vmaf.append(
                        vmaf_size[i][self.env.video_chunk_counter])

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                if last_chunk_vmaf is None:
                    last_chunk_vmaf = video_chunk_vmaf

                # -- linear reward --
                # reward is video quality - rebuffer penalty - smoothness
                # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                #     - REBUF_PENALTY * rebuf \
                #     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                #                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                reward = 0.8469011 * video_chunk_vmaf - 28.79591348 * rebuf + 0.29797156 * \
                    np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - 1.06099887 * \
                    np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - \
                    2.661618558192494
                
                positive_smoothness = 0.29797156 * np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.))
                negative_smoothness = 1.06099887 * np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.))
                # -- log scale reward --
                # log_bit_rate = np.log(
                #     VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
                # log_last_bit_rate = np.log(
                #     VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

                # reward = log_bit_rate \
                #     - REBUF_PENALTY * rebuf \
                #     - SMOOTH_PENALTY * \
                #     np.abs(log_bit_rate - log_last_bit_rate)

                # -- HD reward --
                # reward = HD_REWARD[bit_rate] \
                #     - REBUF_PENALTY * rebuf \
                #     - SMOOTH_PENALTY * \
                #     np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

                r_batch.append(reward)

                last_bit_rate = bit_rate
                last_chunk_vmaf = video_chunk_vmaf

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = video_chunk_vmaf / \
                    100  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / \
                    float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / \
                    BUFFER_NORM_FACTOR  # 10 sec
                state[4, :A_DIM] = np.array(
                    next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, :A_DIM] = np.array(
                    next_video_chunk_vmaf) / 100.  # mega byte
                state[6, -1] = np.minimum(video_chunk_remain,
                                          CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                # compute action probability vector
                action_prob = self.local_actor.predict(
                    sess, np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(
                    1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                entropy_record.append(compute_entropy(action_prob[0]))

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(entropy_record[-1]) + '\t' +
                               str(reward) + '\t' + 
                               str(positive_smoothness) + '\t' +
                               str(negative_smoothness) + '\n')
                log_file.flush()

                epoch += 1

                # report experience to the coordinator
                if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:

                    actor_gradient, critic_gradient, td_batch = \
                        compute_gradients(
                            sess=sess,
                            s_batch=np.stack(s_batch, axis=0),
                            a_batch=np.vstack(a_batch),
                            r_batch=np.vstack(r_batch),
                            terminal=end_of_video, actor=self.local_actor, critic=self.local_critic)

                    self.train(sess, actor_gradient, critic_gradient)

                    self.local_actor.update_local_params(sess)
                    self.local_critic.update_local_params(sess)

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]
                    del entropy_record[:]

                    # so that in the log we know where video ends
                    log_file.write('\n')

                # store the state and action into batches
                if end_of_video:
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here
                    last_chunk_vmaf = None

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)

                else:
                    s_batch.append(state)

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1
                    a_batch.append(action_vec)

                if epoch % MODEL_SAVE_INTERVAL == 0 and self.saver_thread and epoch >= 70000:

                    save_path = SUMMARY_DIR + '/global_' + self.global_assignment + \
                        '/nn_model_' + str(epoch) + '.pickle'
                    f = open(save_path, 'wb')
                    params = self.global_actor.get_network_params(sess)
                    pickle.dump(params, f)
                    f.close()

                    self.queue.put(
                        {'epoch': epoch, 'params': params})
                    # del params
                    # gc.collect()

            if self.saver_thread:
                self.queue.put({'epoch': 'finished'})
            end = time()
            elapsed = strftime("%Hh%Mm%Ss", gmtime(end - start))
            # write elapsed time for testing
            with open(LOG_FILE + '_time_training_' + str(self.name), 'w') as f:
                f.write(elapsed)
                f.close()


def main():

    os.system('rm -r ' + SUMMARY_DIR)
    os.system('mkdir ' + SUMMARY_DIR)
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)

    global_actors = []
    global_critics = []
    testing_actors = []

    workers = []
    queues = []
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

    for i in range(GLOBAL_WORKERS):
        queues.append(Queue())
        agent_name = 'global_'+str(i)
        os.system('rm -r ' + SUMMARY_DIR + '/' + agent_name)
        os.system('mkdir ' + SUMMARY_DIR + '/' + agent_name)

        global_actors.append(ActorNetwork(state_dim=[
                             S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, global_workers=None, scope='actor_'+agent_name, entropy_weight=None))
        global_critics.append(CriticNetwork(state_dim=[
                              S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE, global_workers=GLOBAL_WORKERS, scope='critic_'+agent_name))
        testing_actors.append(ActorNetwork(state_dim=[
            S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, global_workers=None, scope='testing_actor_'+agent_name, entropy_weight=None))

    for j in range(GLOBAL_WORKERS):
        for i in range(NUM_WORKERS):
            workers.append(Worker(global_assignment=j, name=i, all_cooked_time=all_cooked_time,
                           all_cooked_bw=all_cooked_bw, saver_thread=i == 0, queue=queues[j] if i == 0 else None, global_actor=global_actors[j]))  # create workers for each global parameter set

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        testing_threads = []
        for i in range(GLOBAL_WORKERS):
            def tester_work(): return run_tests(
                sess, queues[i], testing_actors[i])
            t = threading.Thread(target=(tester_work))
            t.start()
            testing_threads.append(t)
            sleep(0.1)

        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(sess)
            # threading operator to run multiple workers
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.1)
            worker_threads.append(t)

        for t in worker_threads:
            t.join()
        for t in testing_threads:
            t.join()

        sess.close()


if __name__ == '__main__':
    main()
