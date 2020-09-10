import tensorflow as tf
import numpy as np
from predictor import Feature
import os, math


cur_path = os.getcwd()
MODEL_TEMPLATE_PATH = os.path.join(cur_path, './use_priori/filter_topo/block{}.pb')

class TopologyEval:

    def _trans(self, graph):
        g_len = len(graph)
        terminal = graph.index([])
        order = [[i for i in range(g_len)], [i for i in range(g_len)]]
        for i in range(g_len):
            for j in graph[i]:
                if j > terminal:
                    order[1][j] = order[1][i] + 1
        order = np.array(order)
        order = order.T[np.lexsort(order)].T
        order = list(order[0])

        new_graph = [graph[i].copy() for i in order]
        for e in new_graph:
            for i in range(len(e)):
                e[i] = order.index(e[i])
        return new_graph

    def _list2mat(self, G):
        graph = np.zeros((len(G), len(G)), dtype=int)
        for i in range(len(G)):
            e = G[i]
            if e:
                for k in e:
                    graph[i][k] = 1
        return graph

    def _padding(self, input, length):
        shape = input.shape
        if len(input) < length:
            fill_num = np.zeros([length - shape[0], shape[1]])
            input = np.vstack((input, fill_num))
        return input

    def _concat(self, graphs):
        if len(graphs) == 1:
            return graphs[0]
        else:
            new_graph_length = 0
            for g in graphs:
                new_graph_length += len(g)
            new_graph = np.zeros((new_graph_length, new_graph_length), dtype=int)
            index = 0  # the staring connection position of next graph
            for g in graphs:
                new_graph[index:index + len(g), index:index + len(g)] = g
                if index + len(g) < new_graph_length:
                    new_graph[index + len(g) - 1][index + len(g)] = 1
                index = index + len(g)
            return new_graph

    def topo1vstopo2(self, topo1, topo2, block_id):
        '''

        :param topo1:
        :param topo2:
        :param block_id: the block to which topology belongs, ranges form 1 to 4
        :return:
        '''

        assert len(topo1)==len(topo2), 'topo1 and topo2 be must the same length'
        assert block_id in [1,2,3,4], 'the value of block_id ranges from 1 to 4'

        x_1 = []
        x_2 = []
        for t1, t2 in zip(topo1, topo2):
            t1 = list(map(self._trans, t1))
            t1 = list(map(self._list2mat, t1))
            t1 = self._concat(t1)
            t1 = Feature(t1)._feature_nodes()
            t1 = self._padding(t1, 71)
            x_1.append(t1)

            t2 = list(map(self._trans, t2))
            t2 = list(map(self._list2mat, t2))
            t2 = self._concat(t2)
            t2 = Feature(t2)._feature_nodes()
            t2 = self._padding(t2, 71)
            x_2.append(t2)
        batch_size = 512
        result = []
        with tf.Session() as sess:
            with tf.gfile.FastGFile(MODEL_TEMPLATE_PATH.format(block_id), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
            init = tf.global_variables_initializer()
            sess.run(init)
            graph = tf.get_default_graph()
            input_1 = graph.get_tensor_by_name("Placeholder:0")
            input_2 = graph.get_tensor_by_name("Placeholder_1:0")
            logits = graph.get_tensor_by_name("Softmax:0")
            pred = tf.argmax(logits, 1)  # return confidence
            # for i in range(int(len(x_1)/batch_size)+1):
            for i in range(math.ceil(len(x_1)/batch_size)):
                start = i*batch_size
                end = (i+1)*batch_size if (i+1)*batch_size<len(x_1) else len(x_1)
                out = sess.run(pred, feed_dict={input_1: x_1[start:end], input_2: x_2[start:end]})
                result.extend(out)
            return result


if __name__ == "__main__":
    a = [[[[1], [2], []], [[1], [2], []]], [[[1], [2], []], [[1, 3], [2], [], [2]]]]
    b = [[[[1], [2], []], [[1, 3], [2], [], [2]]], [[[1], [2], []], [[1], [2], []]]]
    TopologyEval().topo1vstopo2(a, b, 2)


