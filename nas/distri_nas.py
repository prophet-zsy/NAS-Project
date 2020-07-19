import random
import time
import os
import copy
import socket
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager

from .enumerater import Enumerater
from .evaluator import Evaluator
# from .optimizer import Optimizer
# from .sampler import Sampler

NETWORK_POOL_TEMPLATE = []
NETWORK_POOL = []
gpu_list = multiprocessing.Queue()


def call_eva(graph, cell, nn_preblock, round, pos, finetune_signal, pool_len, eva, ngpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu)
    with open("memory/evaluating_log_with_gpu{}.txt".format(ngpu), "a") as f:
        f.write("\nblock_num:{} round:{} network_index:{}/{}".format(len(nn_preblock)+1, round, pos, pool_len))
        start_time = time.time()
        while True:
            try:
                score = eva.evaluate(graph, cell, nn_preblock, False, finetune_signal, f)
                break
            except:
                print("\nevaluating failed and we will try again...\n")
                f.write("\nevaluating failed and we will try again...\n")

        end_time = time.time()
    time_cost = end_time - start_time
    gpu_list.put(ngpu)
    return score, time_cost, pos


class QueueManager(BaseManager):
    pass


class Communication:
    def __init__(self, role, ps_host):
        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        flag = multiprocessing.Queue()
        data_queue = multiprocessing.Queue()

        QueueManager.register('get_task_queue', callable=lambda: task_queue)
        QueueManager.register('get_result_queue', callable=lambda: result_queue)
        QueueManager.register('get_flag', callable=lambda: flag)
        QueueManager.register('get_data_sync', callable=lambda: data_queue)

        # TODO there might be other ways to get the IP address
        server_addr = socket.gethostbyname(ps_host.split(":")[0])
        self.manager = QueueManager(address=(server_addr, int(ps_host.split(":")[1])), authkey=b'abc')

        if role == "ps":
            self.manager.start()
        else:
            while True:
                try:
                    self.manager.connect()
                    break
                except:
                    time.sleep(20)
                    print("waiting for connecting ...")

        self.task = self.manager.get_task_queue()
        self.result = self.manager.get_result_queue()
        self.end_flag = self.manager.get_flag()  # flag for whether the whole process is over
        self.data_sync = self.manager.get_data_sync()  # flag for sync of adding data and sync of round
        self.data_count = 0  # mark how many times to add data locally


class Nas:
    def __init__(self, ps_host='', worker_host='', job_name='ps', task_index=-1, m_best=1, opt_best_k=5, randseed=-1, depth=6,
                 width=3, max_branch_depth=6, num_gpu=1, pattern="Global", block_num=1):
        self.__m_best = m_best
        self.__m_pool = []
        self.__opt_best_k = opt_best_k
        self.__depth = depth
        self.__width = width
        self.__max_bdepth = max_branch_depth
        self.num_gpu = num_gpu
        self.ps_host = ps_host
        self.worker_host = worker_host
        self.job_name = job_name
        self.task_index = task_index
        self.__pattern = pattern
        self.__block_num = block_num
        self.__finetune_threshold = 5

        if randseed is not -1:
            random.seed(randseed)
            tf.set_random_seed(randseed)

        return

    def __list_swap(self, ls, i, j):
        cpy = ls[i]
        ls[i] = ls[j]
        ls[j] = cpy

    def __eliminate(self, network_pool=None, scores=[], round=0):
        """
        Eliminates the worst 50% networks in network_pool depending on scores.
        """
        scores_cpy = scores.copy()
        scores_cpy.sort()
        original_num = len(scores)
        mid_index = original_num // 2
        mid_val = scores_cpy[mid_index]

        original_index = [i for i in range(len(scores))]  # record the

        i = 0
        while i < len(network_pool):
            if scores[i] < mid_val:
                # del network_pool[i]   # TOO SLOW !!
                # del scores[i]
                self.__list_swap(network_pool, i, len(network_pool) - 1)
                self.__list_swap(scores, i, len(scores) - 1)
                self.__list_swap(original_index, i, len(original_index) - 1)
                self.save_info("memory/network_info.txt", network_pool.pop(), round, original_index.pop(), original_num)
                scores.pop()
            else:
                i += 1
        print("NAS: eliminating {}, remaining {}...".format(original_num - len(scores), len(scores)))
        return mid_val

    def __datasize_ctrl(self, type="", eva=None):
        """
        Increase the dataset's size in different way
        """
        # TODO Where is Class Dataset?
        cur_train_size = eva.get_train_size()
        if type.lower() == 'same':
            nxt_size = cur_train_size * 2
        else:
            raise Exception("NAS: Invalid datasize ctrl type")

        eva.set_train_size(nxt_size)
        return

    def __save_log(self, path="",
                   optimizer=None,
                   sampler=None,
                   enumerater=None,
                   evaluater=None):
        with open(path, 'w') as file:
            file.write("-------Optimizer-------")
            file.write(optimizer.log)
            file.write("-------Sampler-------")
            file.write(sampler.log)
            file.write("-------Enumerater-------")
            file.write(enumerater.log)
            file.write("-------Evaluater-------")
            file.write(evaluater.log)
        return

    def save_info(self, path, network, round, original_index, network_num):
        with open(path, 'a') as f:
            f.write("block_num: {} round: {} network_index: {}/{} number of scheme: {}\n".format(len(network.pre_block)+1, round, original_index, network_num, len(network.score_list)))
            f.write("graph_part:")
            self.wirte_list(f, network.graph_part)
            for item in zip(network.graph_full_list, network.cell_list, network.score_list):
                f.write("    graph_full:")
                self.wirte_list(f, item[0])
                f.write("    cell_list:")
                self.wirte_list(f, item[1])
                f.write("    score:")
                f.write(str(item[2]) + "\n")

    def wirte_list(self, f, graph):
        f.write("[")
        for node in graph:
            f.write("[")
            for ajaceny in node:
                f.write(str(ajaceny) + ",")
            f.write("],")
        f.write("]" + "\n")

    def __game(self, eva, NETWORK_POOL, scores, com, round):
        pool_len = len(NETWORK_POOL)
        print("NAS: Now we have {0} networks. Start game!".format(pool_len))
        network_index = 0
        # put all the network in this round into the task queue
        if pool_len < self.__finetune_threshold:
            finetune_signal = True
        else:
            finetune_signal = False
        for nn, score in zip(NETWORK_POOL, scores):
            if round == 1:
                cell, graph = nn.cell_list[-1], nn.graph_full_list[-1]
            else:
                nn.opt.update_model(nn.table, score)
                nn.table = nn.opt.sample()
                nn.spl.renewp(nn.table)
                cell, graph = nn.spl.sample()
                nn.graph_full_list.append(graph)
                nn.cell_list.append(cell)
            com.task.put([graph, cell, nn.pre_block, network_index, round, finetune_signal, pool_len])
            network_index += 1

        # TODO data size control
        com.data_sync.put(com.data_count)  # for multi host
        eva.add_data(1600)

        pool = Pool(processes=self.num_gpu)
        # as long as there's still task available
        eva_result_list = []
        while not com.task.empty():
            gpu = gpu_list.get()  # get a gpu resource
            try:
                graph, cell, nn_pre_block, network_index, round, finetune_signal, pool_len = com.task.get(timeout=1)
            except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
                gpu_list.put(gpu)
                break
            print("round:{} network_index:{}".format(round, network_index))
            print("graph:", graph)
            print("cell:", cell)
            eva_result = pool.apply_async(call_eva, args=(graph, cell, nn_pre_block, round, network_index, finetune_signal, pool_len, eva, gpu))
            eva_result_list.append(eva_result)

        pool.close()
        pool.join()

        for eva_result in eva_result_list:
            score, time_cost, network_index = eva_result.get()
            print("network_index:{} score:{} time_cost:{} ".format(network_index, score, time_cost))
            com.result.put([score, network_index, time_cost])

        while com.result.qsize() != len(NETWORK_POOL):  # waiting for the other workers
            print("we have gotten {} scores , but there are {} networks, waiting for the other workers...".format(com.result.qsize(), len(NETWORK_POOL)))
            time.sleep(20)
        # fill the score list
        print("score  network_index  time_cost")
        while not com.result.empty():
            tmp = com.result.get()
            print(tmp)
            scores[tmp[1]] = tmp[0]

        print("scores:", scores)
        # record the score of every network
        for nn, score in zip(NETWORK_POOL, scores):
            nn.score_list.append(score)

        return scores

    def train_winner_subprocess(self, eva, NETWORK_POOL):
        best_nn = NETWORK_POOL[0]
        best_opt_score = 0
        best_cell_i = 0
        eva.add_data(-1)  # -1 represent that we add all data for training
        print("NAS: Configuring ops and skipping for the best structure and training them...")
        for i in range(self.__opt_best_k):
            best_nn.table = best_nn.opt.sample()
            best_nn.spl.renewp(best_nn.table)
            cell, graph = best_nn.spl.sample()
            best_nn.graph_full_list.append(graph)
            best_nn.cell_list.append(cell)
            with open("memory/train_winner_log.txt", "a") as f:
                f.write("\nblock_num:{} sample_count:{}/{}".format(len(best_nn.pre_block) + 1, i, self.__opt_best_k))
                opt_score = eva.evaluate(graph, cell, best_nn.pre_block, True, True, f)
            best_nn.score_list.append(opt_score)
            if opt_score > best_opt_score:
                best_opt_score = opt_score
                best_cell_i = i
        print("NAS: We have got the best network and its score is {}".format(best_opt_score))
        best_index = best_cell_i - self.__opt_best_k
        return best_nn, best_index

    def __train_winner(self, NETWORK_POOL, round):
        eva_winner = Evaluator()
        with Pool(1) as p:
            best_nn, best_index = p.apply(self.train_winner_subprocess, args=(eva_winner, NETWORK_POOL,))
        self.save_info("memory/network_info.txt", best_nn, round, 0, 1)
        return best_nn, best_index

    def initialize_ops_subprocess(self, NETWORK_POOL):
        from .predictor import Predictor
        pred = Predictor()
        for network in NETWORK_POOL:  # initialize the full network by adding the skipping and ops to graph_part
            network.table = network.opt.sample()
            network.spl.renewp(network.table)
            cell, graph = network.spl.sample()
            # network.graph_full_list.append(graph)
            blocks = []
            for block in network.pre_block:  # get the graph_full adjacency list in the previous blocks
                blocks.append(block[0])  # only get the graph_full in the pre_bock
            pred_ops = pred.predictor(blocks, graph)
            table = network.spl.init_p(pred_ops)  # spl refer to the pred_ops
            network.spl.renewp(table)
            cell, graph = network.spl.sample()  # sample again after renew the table
            network.graph_full_list.append(graph)  # graph from first sample and second sample are the same, so that we don't have to assign network.graph_full at first time
            network.cell_list.append(cell)
        return NETWORK_POOL

    def initialize_ops(self, NETWORK_POOL):
        with Pool(1) as p:
            NETWORK_POOL = p.apply(self.initialize_ops_subprocess, args=(NETWORK_POOL,))
        scores = np.zeros(len(NETWORK_POOL))
        scores = scores.tolist()
        return scores, NETWORK_POOL

    def algorithm_ps(self, block_num, eva, com, NETWORK_POOL_TEMPLATE):
        # Different code is ran on different machine depends on whether it's a ps host or a worker host.
        # PS host is used for generating all the networks and collect the final result of evaluation.
        # And PS host together with worker hosts(as long as they all have GPUs) train as well as evaluate all the networks asynchronously inside one round and synchronously between rounds

        # implement the copy when searching for every block
        NETWORK_POOL = copy.deepcopy(NETWORK_POOL_TEMPLATE)
        for network in NETWORK_POOL:  # initialize the sample module
            network.init_sample(self.__pattern, block_num)

        # Step 2: Search best structure
        print("NAS: Configuring the networks in the first round...")
        scores, NETWORK_POOL = self.initialize_ops(NETWORK_POOL)
        round = 0
        while (len(NETWORK_POOL) > 1):
            # Step 3: Sample, train and evaluate every network
            com.data_count += 1
            round += 1
            scores = self.__game(eva, NETWORK_POOL, scores, com, round)
            # Step 4: Eliminate half structures and increase dataset size
            self.__eliminate(NETWORK_POOL, scores, round)
            # self.__datasize_ctrl("same", epic)
        print("NAS: We got a WINNER!")
        # Step 5: Global optimize the best network
        best_nn, best_index = self.__train_winner(NETWORK_POOL, round+1)
        # self.__save_log("", opt, spl, enu, eva)
        return best_nn, best_index

    def algorithm_worker(self, eva, com):
        for gpu in range(self.num_gpu):
            gpu_list.put(gpu)
        pool = Pool(processes=self.num_gpu)
        while com.end_flag.empty():
            # as long as there's still task available
            # Data control for new round
            while com.data_sync.empty():
                print("waiting for assignment of next round...")
                time.sleep(20)
            com.data_count += 1
            data_count_ps = com.data_sync.get(timeout=1)
            assert com.data_count <= data_count_ps, "add data sync failed..."
            eva.add_data(1600*(data_count_ps-com.data_count+1))
            eva_result_list = []
            while not com.task.empty():
                gpu = gpu_list.get()  # get a gpu resource
                try:
                    graph, cell, nn_pre_block, network_index, round, finetune_signal, pool_len = com.task.get(timeout=1)
                except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
                    gpu_list.put(gpu)
                    break
                print("round:{} network_index:{}".format(round, network_index))
                print("graph:", graph)
                print("cell:", cell)
                eva_result = pool.apply_async(call_eva, args=(graph, cell, nn_pre_block, round, network_index, finetune_signal, pool_len, eva, gpu))
                eva_result_list.append(eva_result)
            for eva_result in eva_result_list:
                score, time_cost, network_index = eva_result.get()
                print("network_index:{} score:{} time_cost:{} ".format(network_index, score, time_cost))
                com.result.put([score, network_index, time_cost])
            while com.task.empty():
                print("waiting for assignment of next round...")
                time.sleep(20)
        pool.close()
        pool.join()
        return "I am a worker..."

    def run(self):
        if self.job_name == "ps":
            print("NAS: Initializing enu and eva...")
            enu = Enumerater(
                depth=self.__depth,
                width=self.__width,
                max_branch_depth=self.__max_bdepth)
            eva = Evaluator()
            print("NAS: Enumerating all possible networks!")
            NETWORK_POOL_TEMPLATE = enu.enumerate()
            com = Communication("ps", self.ps_host)
            for gpu in range(self.num_gpu):
                gpu_list.put(gpu)
            for i in range(self.__block_num):
                print("NAS: Searching for block {}/{}...".format(i + 1, self.__block_num))
                block, best_index = self.algorithm_ps(i, eva, com, NETWORK_POOL_TEMPLATE)
                block.pre_block.append([block.graph_part, block.graph_full_list[best_index], block.cell_list[best_index]])  # or NetworkUnit.pre_block.append()
            com.end_flag.put(1)
            com.manager.shutdown()
            return block.pre_block  # or NetworkUnit.pre_block
        else:
            eva = Evaluator()
            com = Communication("worker", self.ps_host)
            self.algorithm_worker(eva, com)
            return "all of the blocks have been evaluated, please go to the ps manager to view the result..."


if __name__ == '__main__':
    nas = Nas()
    print(nas.run())
