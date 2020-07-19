import random
import time
import os
import copy
import socket
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Pool
import queue

from .enumerater import Enumerater
from .evaluator import Evaluator
# from .optimizer import Optimizer
# from .sampler import Sampler

NETWORK_POOL_TEMPLATE = []
NETWORK_POOL = []
gpu_list = multiprocessing.Queue()


def call_eva(graph, cell, nn_preblock, round, pos, spl_index, finetune_signal, pool_len, eva, ngpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu)
    with open("memory/evaluating_log_with_gpu{}.txt".format(ngpu), "a") as f:
        f.write("\nblock_num:{} round:{} network_index:{}/{}".format(len(nn_preblock)+1, round, pos, pool_len))
        start_time = time.time()
        # while True:
        #     try:
        score = eva.evaluate(graph, cell, nn_preblock, False, finetune_signal, f)
        #     break
        # except:
        #     print("\nevaluating failed and we will try again...\n")
        #     f.write("\nevaluating failed and we will try again...\n")

        end_time = time.time()
    time_cost = end_time - start_time
    gpu_list.put(ngpu)
    return score, time_cost, pos, spl_index


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
        self.__num_spl_every_game = 8

        if randseed is not -1:
            random.seed(randseed)
            tf.set_random_seed(randseed)

        return

    def __list_swap(self, ls, i, j):
        cpy = ls[i]
        ls[i] = ls[j]
        ls[j] = cpy

    def __eliminate(self, network_pool=None, round=0):
        """
        Eliminates the worst 50% networks in network_pool depending on scores.
        """
        # eliminate by the best score it has ever met
        scores = [network_pool[nn_index].best_score for nn_index in range(len(network_pool))]
        scores.sort()
        original_num = len(scores)
        mid_index = original_num // 2
        mid_val = scores[mid_index]
        original_index = [i for i in range(len(scores))]  # record the number of the removed network in the pool

        i = 0
        while i < len(network_pool):
            if network_pool[i].best_score < mid_val:
                self.__list_swap(network_pool, i, len(network_pool) - 1)
                self.__list_swap(original_index, i, len(original_index) - 1)
                self.save_info("memory/network_info.txt", network_pool.pop(), round, original_index.pop(), original_num)
            else:
                i += 1
        print("NAS: eliminating {}, remaining {}...".format(original_num - len(network_pool), len(network_pool)))
        return mid_val

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

    def __game(self, eva, NETWORK_POOL, round):
        task = queue.Queue()
        result = queue.Queue()
        pool_len = len(NETWORK_POOL)
        print("NAS: Now we have {0} networks. Start game!".format(pool_len))
        network_index = 0
        # put all the network in this round into the task queue
        if pool_len < self.__finetune_threshold:
            finetune_signal = True
        else:
            finetune_signal = False
        for nn in NETWORK_POOL:
            if round == 1:
                cell, graph = nn.cell_list[-1], nn.graph_full_list[-1]
                task.put([graph, cell, nn.pre_block, network_index, 0, round, finetune_signal, pool_len])
            else:
                nn.opt.update_model(nn.table, nn.score_list[-1])
                for spl_index in range(self.__num_spl_every_game):
                    nn.table = nn.opt.sample()
                    nn.spl.renewp(nn.table)
                    cell, graph = nn.spl.sample()
                    nn.graph_full_list.append(graph)
                    nn.cell_list.append(cell)
                    task.put([graph, cell, nn.pre_block, network_index, spl_index, round, finetune_signal, pool_len])
            network_index += 1

        # TODO data size control
        eva.add_data(1600)

        pool = Pool(processes=self.num_gpu)
        # as long as there's still task available
        eva_result_list = []
        while not task.empty():
            gpu = gpu_list.get()  # get a gpu resource
            try:
                graph, cell, nn_pre_block, network_index, spl_index, round, finetune_signal, pool_len = task.get(timeout=1)
            except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
                gpu_list.put(gpu)
                break
            print("round:{} network_index:{} spl_index:{}".format(round, network_index, spl_index))
            print("graph:", graph)
            print("cell:", cell)
            eva_result = pool.apply_async(call_eva, args=(graph, cell, nn_pre_block, round, network_index, spl_index, finetune_signal, pool_len, eva, gpu))
            eva_result_list.append(eva_result)

        pool.close()
        pool.join()

        # get the result
        for eva_result in eva_result_list:
            score, time_cost, network_index, spl_index = eva_result.get()
            print("network_index:{} spl_index:{} score:{} time_cost:{} ".format(network_index, spl_index, score, time_cost))
            result.put([score, network_index, spl_index, time_cost])

        # fill the score list
        print("score  network_index spl_index time_cost")
        tmp = [[0 for _ in range(self.__num_spl_every_game)] for _ in range(pool_len)]  # store all of the network's score_list in current game
        while not result.empty():
            score, time_cost, network_index, spl_index = result.get()
            print([score, time_cost, network_index, spl_index])
            tmp[network_index][spl_index] = score
        print(tmp)

        for nn_index in range(pool_len):  # fill the score_list of every network and find the best score
            for sp_index in range(self.__num_spl_every_game):
                score = tmp[nn_index][sp_index]
                NETWORK_POOL[nn_index].score_list.append(score)
                if score > NETWORK_POOL[nn_index].best_score:
                    NETWORK_POOL[nn_index].best_score = score
                    NETWORK_POOL[nn_index].best_index = len(NETWORK_POOL[nn_index].cell_list) - self.__num_spl_every_game + sp_index

    def train_winner_subprocess(self, eva, NETWORK_POOL):
        best_nn = NETWORK_POOL[0]
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
            if opt_score > best_nn.best_score:
                best_nn.best_score = opt_score
                best_nn.best_index = len(best_nn.cell_list) - 1
        print("NAS: We have got the best network and its score is {}".format(best_nn.best_score))
        return best_nn, best_nn.best_index

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
        return NETWORK_POOL

    def algorithm(self, block_id, eva, NETWORK_POOL_TEMPLATE):

        # implement the copy when searching for every block
        NETWORK_POOL = copy.deepcopy(NETWORK_POOL_TEMPLATE)
        for network in NETWORK_POOL:  # initialize the sample module
            network.init_sample(self.__pattern, block_id)

        print("NAS: Configuring the networks in the first round...")
        NETWORK_POOL = self.initialize_ops(NETWORK_POOL)
        round = 0
        while (len(NETWORK_POOL) > 1):
            # Sample, train and evaluate every network
            round += 1
            self.__game(eva, NETWORK_POOL, round)
            # Eliminate half structures and increase dataset size
            self.__eliminate(NETWORK_POOL, round)
            # self.__datasize_ctrl("same", epic)
        print("NAS: We got a WINNER!")
        # Global optimize the best network
        best_nn, best_index = self.__train_winner(NETWORK_POOL, round)
        # self.__save_log("", opt, spl, enu, eva)
        return best_nn, best_index

    def run(self):
        print("NAS: Initializing enu and eva...")
        enu = Enumerater(
            depth=self.__depth,
            width=self.__width,
            max_branch_depth=self.__max_bdepth)
        eva = Evaluator()
        print("NAS: Enumerating all possible networks!")
        NETWORK_POOL_TEMPLATE = enu.enumerate()
        for gpu in range(self.num_gpu):
            gpu_list.put(gpu)
        for block_id in range(self.__block_num):
            print("NAS: Searching for block {}/{}...".format(block_id + 1, self.__block_num))
            block, best_index = self.algorithm(block_id, eva, NETWORK_POOL_TEMPLATE)
            block.pre_block.append([block.graph_part, block.graph_full_list[best_index], block.cell_list[best_index]])  # or NetworkUnit.pre_block.append()
        return block.pre_block  # or NetworkUnit.pre_block



if __name__ == '__main__':
    nas = Nas()
    print(nas.run())
