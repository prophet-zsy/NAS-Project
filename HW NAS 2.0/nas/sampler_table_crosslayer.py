import random
from sampling.load_configuration import load_conf
from optimizer import Dimension
import pickle
from optimizer import Optimizer
from optimizer import RacosOptimization
import numpy as np
from evaluater import Evaluater
from multiprocessing import Process,Pool
import multiprocessing
from base import NetworkUnit
import time
import pickle
import copy
from queue import Queue

# # # # #
class Vertex:
    #顶点类
    def __init__(self,vid,linked):
        self.id = vid#出边
        self.linked = linked #出边指向的顶点id的列表，也可以理解为邻接表
        self.unknow = 1 #是否搜索过
        self.dist = 99 #s到该点的距离,默认为无穷大

def create_class(graph):
    G = []
    for i in range(len(graph)):
        G.append(Vertex(i,graph[i]))
    return G

def bfs(start_node, graph, max_dist):
    start_node.dist = 0
    q = Queue()
    q.put(start_node)
    connection = []
    while(not q.empty()):
        v = q.get()#返回并删除队列头部元素
        for node in v.linked:
            if(graph[node].dist == 99):
                graph[node].dist = v.dist + 1
                if graph[node].dist > max_dist:
                    return connection
                if graph[node].dist > 1:
                    connection.append(graph[node].id)
                if graph[node].unknow: #一个点的邻点只能被加入队列一次
                    q.put(graph[node])
                    graph[node].unknow = 0
    return connection


def connect(graph_part, max_dist):
    connection = []
    for node in range(0,len(graph_part)):
        graph = create_class(graph_part)
        connection.append(bfs(graph[node], graph, max_dist))
    return connection

# # # # #

class Sampler:

    def __init__(self, graph_part, crosslayer_dis):

        '''
        :param nn: NetworkUnit
        '''

        self.graph_part = graph_part
        self.crosslayer_dis = crosslayer_dis
        # 设置结构大小
        self.node_number = len(self.graph_part)


        # 基于节点设置其可能的跨层连接
        self.crosslayer = connect(self.graph_part, self.crosslayer_dis)
        # self.crosslayer = self.get_crosslayer()

        # 读取配置表得到操作的对应映射
        self.setting, self.pros, self.parameters_subscript_node, = load_conf()
        #
        del self.setting['dense']

        self.dic_index = self._init_dict()

        self.p = []

        # 设置优化Dimension
        # 设置优化的参数
        self.__region, self.__type = self.opt_parameters()
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.__region))
        self.dim.set_regions(self.__region, self.__type)
        self.parameters_subscript = []  #

    # 更新p
    def renewp(self, newp):
        self.p = newp

    def get_crosslayer(self):
        cl = []
        for i in range(len(self.graph_part)):
            cl.append(self.bfs(i))
        return cl

    def bfs(self, i):
        res_list = []
        q = Queue()
        q.put([i, 0])

        v = [0 for ii in range(len(self.graph_part) + 1)]

        while q.empty() is False:
            f = q.get()

            if f[1] >= 2:
                if f[1] <= self.crosslayer_dis:
                    res_list.append(f[0])
                else:
                    continue

            for j in self.graph_part[f[0]]:
                if v[j] == 0:
                    q.put([j, f[1] + 1])
                    v[j] = 1

        return res_list
    #
    def region_cross_type(self, __region_tmp, __type_tmp, i):
        region_tmp = copy.copy(__region_tmp)
        type_tmp = copy.copy(__type_tmp)
        for j in self.crosslayer[i]:
            region_tmp.append([0, 1])
            type_tmp.append(2)

        return region_tmp, type_tmp

    # 基于操作的对应映射得到优化参数
    def opt_parameters(self):
        __type_tmp = []
        __region_tmp = [[0, 0] for _ in range(len(self.dic_index))]
        for key in self.dic_index:
            tmp = int(self.dic_index[key][1] - self.dic_index[key][0])
            __region_tmp[self.dic_index[key][-1]] = [0, tmp]
            __type_tmp.append(2)

        __region = []
        __type = []
        for i in range(self.node_number):

            #
            __region_cross, __type_cross = self.region_cross_type(__region_tmp, __type_tmp, i)

            __region = __region + __region_cross
            __type.extend(__type_cross)
        return __region, __type

    def sample(self):
        res = []
        # 基于节点的搜索结构参数
        l = 0
        r = 0
        graph_part_sample = copy.deepcopy(self.graph_part)

        for num in range(self.node_number):
            # 取一个节点大小的概率
            l = r
            r = l + len(self.dic_index) + len(self.crosslayer[num])
            p_node = self.p[l:r]

            #
            # print(p_node)
            # print(p_node[len(self.dic_index):])
            for i in range(len(p_node[len(self.dic_index):])):
                if p_node[len(self.dic_index):][i] == 1:
                    graph_part_sample[num].append(self.crosslayer[num][i])
            #

            first = p_node[self.dic_index['conv'][-1]]
            # first = p_node[0]
            tmp = ()
            # 首位置确定 conv 还是 pooling
            if first == 0:
                # 搜索conv下的操作
                # 基于操作的对应映射取配置所在的地址，进行取值
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            else:
                # 搜索pooling下的操作
                # 基于操作的对应映射取配置所在的地址，进行取值
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            res.append(tmp)

        return res, graph_part_sample

    # 基于opt.sample()所得结果，基于位置得到操作
    def _init_dict(self):
        dic = {}
        dic['conv'] = (0, 1, 0)
        cnt = 1
        num = 1
        for key in self.setting:
            for k in self.setting[key]:
                tmp = len(self.setting[key][k]['val']) - 1
                dic[key + ' ' + k] = (cnt, cnt + tmp, num)
                num += 1
                cnt += tmp

        return dic

    def get_dim(self):
        return self.dim

    def get_parametets_subscript(self):
        return self.parameters_subscript

    # log
    def get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

    def init_p(self, op):

        '''
        :param op:
        [['64', '7'], ['pooling'], ['64', '3'], ['256', '3'], ['1024', '1'],
        ['1024', '1'], ['1024', '3'], ['1024', '3'], ['1024', '3'], ['512', '1'],
        ['128', '5'], ['64', '3'], ['1024', '1'], ['1024', '1'], ['256', '3']]
        :return:
        '''

        table = []
        l = 0
        r = 0
        for num in range(self.node_number):
            # 取一个节点大小的概率
            l = r
            r = l + len(self.dic_index) + len(self.crosslayer[num])
            p_node = self.p[l:r]
            # print('--'*20)
            # print(p_node)
            # print(op[num])
            if len(op[num]) != 1:
                # struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                a = -1
                b = -1
                c = -1
                for j, i in enumerate(self.setting['conv']['filter_size']['val']):
                    if str(i) == op[num][0]:
                        a = j
                        # p_node[self.dic_index['conv filter_size'][-1]] = j
                        # print(j, '##', self.dic_index['conv filter_size'][-1])

                for j, i in enumerate(self.setting['conv']['kernel_size']['val']):
                    if str(i) == op[num][1]:
                        b = j
                        # p_node[self.dic_index['conv kernel_size'][-1]] = j
                        # print(j, '##', self.dic_index['conv kernel_size'][-1])

                for j, i in enumerate(self.setting['conv']['activation']['val']):
                    if i == 'relu':
                        c = j
                        # p_node[self.dic_index['conv activation'][-1]] = j
                        # print(j, '##', self.dic_index['conv activation'][-1])
                # print(a,b,c)
                p_node[self.dic_index['conv'][-1]] = 0
                if a != -1:
                    p_node[self.dic_index['conv filter_size'][-1]] = a
                if b != -1:
                    p_node[self.dic_index['conv kernel_size'][-1]] = b
                if c != -1:
                   p_node[self.dic_index['conv activation'][-1]] = c

                table = table + p_node
                # print(p_node)

                    # tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            else:
                p_node[self.dic_index['conv'][-1]] = 1
                table = table + p_node

        return table


if __name__ == '__main__':

    init = [['64', '7'], ['pooling'], ['64', '3'], ['256', '3'], ['1024', '1'],
            ['1024', '1'], ['1024', '3'], ['1024', '3'], ['1024', '3'], ['512', '1'],
            ['128', '5'], ['64', '3'], ['1024', '1'], ['1024', '1'], ['256', '3']
            ]
    graph_part = [[1], [2], [3], [4], []]
    NU = [[[1], [2], [3], [4], []],
          [[1], [2, 5], [3], [4], [], [4]],
          [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6], [7]],
          # [[1, 3], [2], [3, 5], [4], [5, 7], [6], [7, 9], [8], [9, 11], [10],
          #  [11, 13], [12], [13, 15], [14], [15, 17], [16], [17], []]
          ]

    for graph_part in NU:
        print('##'*40)
        spl = Sampler(graph_part, 50)

        dic_index = spl.dic_index
        cl = spl.crosslayer

        print(dic_index)
        #
        opt = Optimizer(spl.get_dim(), spl.get_parametets_subscript())
        sample_size = 5  # the instance number of sampling in an iteration
        budget = 20000  # budget in online style
        positive_num = 2  # the set size of PosPop
        rand_probability = 0.99  # the probability of sample in model
        uncertain_bit = 10  # the dimension size that is sampled randomly
        opt.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        opt.clear()

        table = opt.sample()
        spl.renewp(table)

        __cell, graph_part_sample = spl.sample()

        print(__cell)
        # graph_part加入跨层连接的结构
        print(graph_part_sample)

        # init 为初始化的返回,调用init_p 基于操作配置初始化进行更新
        table = spl.init_p(init)
        spl.renewp(table)

        __cell, graph_part_sample = spl.sample()

        print(__cell)
        print(graph_part_sample)


