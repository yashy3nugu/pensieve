import threading
from time import sleep, time
import load_trace
from a3c_gs import ActorNetwork, CriticNetwork, compute_gradients
from run_tests import run_tests
import env
import tensorflow as tf
import os
import logging
import numpy as np
import multiprocessing as mp
from Queue import Queue
# from rl_test_gs import run_tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''

#lr = 0.0003
# entropy = 0.3
# 3 global 2 workers

# lr: actor: 0.0001 critic: 0.001
# 4 global 2 workers
# entropy : 6,4,2,1,0.5,0.3,0.05,0.01 every 10k
# started testing after 1 lakh
# save interval 2000
# highest av is 35-36
# total 1.3 lakh epochs
# final layer 128

# lr: actor: 0.0001 critic: 0.001
# 4 global 2 workers
# entropy : 5,4,2,1,0.5,0.3,0.05,0.01 every 10k
# started testing after 1 lakh
# save interval 2000
# highest av is 37-39
# total 1.3 lakh epochs
# final layer 128,128,64

# same hyper params
# NN net has 3 128 layers
# highest is 39.8

# same hyper params
# NN net has 3 256 layers
# After 1 lakh epochs reduce entropy to 0.005


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
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
EPOCHS = 200000
ENTROPY_WEIGHT = 5


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

        # self.block_global = [tf.assign(self.block_vars[i],[False]) for i in range(global_workers)] #handle to activate lock
        # self.unblock_global = [tf.assign(self.block_vars[i],[True]) for i in range(global_workers)] #handle to deactivate lock

        self.actor_block_stats = self.local_actor.get_block_vars(sess)
        # self.critic_block_stats = self.local_critic.get_block_vars(sess)
        self.valid_actor_updates = list(np.array(self.actor_block_stats))
        # self.valid_critic_updates = list(np.array(self.critic_block_stats))

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

        #
        # for i in range(len(self.other_ids)): sess.run(self.local_AC.apply_other_grads[int(i)], feed_dict=feed_dict) #apply grads to all other global parameters
        # for i in range(GLOBAL_WORKERS): sess.run(self.local_critic.unblock_global[int(i)]) #de-activate all locks

    def work(self, sess):
        print('started worker ' + str(self.global_assignment))

        test_log_file_path = LOG_FILE + '_test_' + str(self.global_assignment)

        with sess.as_default(), sess.graph.as_default(), open(LOG_FILE + self.name, 'wb') as log_file, open(test_log_file_path, 'wb') as test_log_file:

            self.local_actor.transfer_global_params(sess)
            self.local_critic.transfer_global_params(sess)

            # if self.saver_thread:
            #     self.saver = tf.train.Saver(var_list=[x for x in tf.get_collection(
            #         tf.GraphKeys.TRAINABLE_VARIABLES) if 'global_' + str(self.global_assignment) in x.name])

            epoch = 0
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

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

                if epoch == 100000:
                    self.local_actor.entropy_weight = 0.005

                if epoch == 130000:
                    self.local_actor.entropy_weight = 0.001

                if epoch == 170000:
                    self.local_actor.entropy_weight = 0.0005

                # if epoch == 8000:
                #     print('changed lr in epoch ' + str(epoch))
                #     self.local_actor.set_learning_rate(0.000003)
                #     self.local_critic.set_learning_rate(0.000003)

                # the action is from the last decision
                # this is to make the framework similar to the real
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain = \
                    self.env.get_video_chunk(bit_rate)

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                # -- linear reward --
                # reward is video quality - rebuffer penalty - smoothness
                reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                              VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                # -- log scale reward --
                # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
                # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

                # reward = log_bit_rate \
                #          - REBUF_PENALTY * rebuf \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # -- HD reward --
                # reward = HD_REWARD[bit_rate] \
                #          - REBUF_PENALTY * rebuf \
                #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

                r_batch.append(reward)

                last_bit_rate = bit_rate

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                    float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / \
                    float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / \
                    BUFFER_NORM_FACTOR  # 10 sec
                state[4, :A_DIM] = np.array(
                    next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain,
                                          CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                # compute action probability vector
                action_prob = self.local_actor.predict(
                    sess, np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(
                    1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                # entropy_record.append(a3c.compute_entropy(action_prob[0]))

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n')
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

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)

                else:
                    s_batch.append(state)

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1
                    a_batch.append(action_vec)

                if epoch % MODEL_SAVE_INTERVAL == 0 and self.saver_thread and epoch >= 100000:
                    # Save the neural net parameters to disk.
                    # save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                    #                     str(epoch) + ".ckpt")
                    # self.local_actor.update_local_params(sess)
                    # self.local_critic.update_local_params(sess)
                    # print('saving epoch ', epoch,
                    #       ' for global var ', self.global_assignment)

                    # self.testing(sess, epoch, test_log_file)
                    # print('saved epoch ', epoch, ' for global var ',
                    #       self.global_assignment)
                    params = self.global_actor.get_network_params(sess)
                    self.queue.put({'epoch': epoch, 'params': params})

            if self.saver_thread:
                self.queue.put({'epoch': 'finished'})

    # def testing(self, sess, epoch, log_file):
    #     # clean up the test results folder
    #     thread_test_folder = TEST_LOG_FOLDER + self.global_assignment + '/'
    #     os.system('rm -r ' + thread_test_folder)
    #     os.system('mkdir ' + thread_test_folder)

    #     # run test script
    #     # os.system('python rl_test.py ' + nn_model)
    #     # run_tests(sess, self.local_actor, self.global_assignment)

    #     # append test performance to the log
    #     rewards = []
    #     test_log_files = os.listdir(thread_test_folder)
    #     for test_log_file in test_log_files:
    #         reward = []
    #         with open(thread_test_folder + test_log_file, 'rb') as f:
    #             for line in f:
    #                 parse = line.split()
    #                 try:
    #                     reward.append(float(parse[-1]))
    #                 except IndexError:
    #                     break
    #         rewards.append(np.sum(reward[1:]))

    #     rewards = np.array(rewards)

    #     rewards_min = np.min(rewards)
    #     rewards_5per = np.percentile(rewards, 5)
    #     rewards_mean = np.mean(rewards)
    #     rewards_median = np.percentile(rewards, 50)
    #     rewards_95per = np.percentile(rewards, 95)
    #     rewards_max = np.max(rewards)

    #     log_file.write(str(epoch) + '\t' +
    #                    str(rewards_min) + '\t' +
    #                    str(rewards_5per) + '\t' +
    #                    str(rewards_mean) + '\t' +
    #                    str(rewards_median) + '\t' +
    #                    str(rewards_95per) + '\t' +
    #                    str(rewards_max) + '\n')
    #     log_file.flush()


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


if __name__ == '__main__':
    main()
