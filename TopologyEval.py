import tensorflow as tf
import numpy as np
from predictor import Feature
import os, math
import json

cur_path = os.getcwd()
from tensorflow.python.framework import graph_util
MODEL_TEMPLATE_PATH = os.path.join(cur_path, 'use_priori', 'filter_topo')

class CommonOps:
    def process(self, net):
        l = len(net)
        net_c = [e.copy() for e in net]
        [e.remove(l - 1) for e in net_c if l - 1 in e]
        net_c.pop(-1)

        order = [[i for i in range(l)], [i for i in range(l)]]
        for i in range(l - 1):
            for j in net_c[i]:
                if j > i:
                    order[1][j] = order[1][i] + 1
        order = np.array(order)
        order = order.T[np.lexsort(order)].T

        new_net = [net_c[i] for i in order[0][:-1]]
        for i in range(l - 1):
            for j in range(len(new_net[i])):
                new_net[i][j] = list(order[0]).index(new_net[i][j])
        [new_net[list(order[0]).index(i)].append(l - 1) for i in range(l - 1) if l - 1 in net[i]]
        new_net.append([])
        return new_net, order

    def trans(self, graph):
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

        new_graph = [graph[i] for i in order]
        for e in new_graph:
            for i in range(len(e)):
                e[i] = order.index(e[i])
        return new_graph

    def list2mat(self, G):
        graph = np.zeros((len(G), len(G)), dtype=int)
        for i in range(len(G)):
            e = G[i]
            if e:
                for k in e:
                    graph[i][k] = 1
        return graph

    def concat(self, graphs):
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

    def padding(self, input, length):
        shape = input.shape
        if len(input) < length:
            fill_num = np.zeros([length - shape[0], shape[1]])
            input = np.vstack((input, fill_num))
        return input

    def lstm_cell(self, x,
                  units,
                  prev_h,
                  prev_c,
                  kernel_w,
                  current_w,
                  use_bias=True,
                  bias=None,
                  activation='tanh',
                  recurrent_activation='sigmoid'
                  ):
        z = tf.matmul(x, kernel_w)
        z = tf.add(tf.matmul(prev_h, current_w), z)
        if use_bias:
            z = tf.add(z, bias)

        z0 = z[:, :units]
        z1 = z[:, units:2 * units]
        z2 = z[:, 2 * units:3 * units]
        z3 = z[:, 3 * units:]

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        c = f * prev_c + i * tf.tanh(z2)
        o = tf.sigmoid(z3)

        h = o * tf.tanh(c)
        return h, c


class DataProcess:
    def __init__(self, root_path, scene):
        self.root_path = root_path
        self.scene = scene
        self.cops = CommonOps()

    def read_data(self):
        if os.path.exists(self.root_path):
            file_tuples = os.walk(self.root_path)
            nas_configs = []
            network_infos = []
            for f_tup in file_tuples:
                if f_tup[0].split('\\')[-1].split('_')[-1]==self.scene:
                    nas_configs.append(f_tup[0]+'\\nas_config.json')
                    network_infos.append(f_tup[0]+'\\memory\\network_info.txt')
            block = self._merge_file(nas_configs, network_infos)
            print('----suceessful load data----')
            return block
        else:
            print(self.root_path+'is not find')

    def read_data_by_block(self,block_id):
        path = self.root_path+'\\'+self.scene+'\\'
        file_names = os.listdir(path)
        train_data = []
        for file_name in file_names:
            prefix = file_name.split('.')[0]
            file_block_id = int(prefix.split('-')[1])
            if file_block_id==block_id:
                train_data.append(np.load(path+file_name, allow_pickle=True))
        train_data = np.vstack(train_data)
        np.random.shuffle(train_data)
        x_1 = [i.tolist() for i in train_data[:, 0]]
        x_2 = [i.tolist() for i in train_data[:, 1]]

        split = int(len(train_data) * 3 / 4)
        train_x_1 = x_1[:split]
        val_x_1 = x_1[split:]
        train_x_2 = x_2[:split]
        val_x_2 = x_2[split:]
        label = train_data[:, 2]

        label = list(label)
        for i in range(len(label)):
            label[i] = [1, 0] if label[i] == 0 else [0, 1]
        train_label = label[:split]
        val_label = label[split:]
        return train_x_1, train_x_2, train_label, val_x_1, val_x_2, val_label

    def _merge_file(self, nas_configs, network_infos):
        num_file = len(nas_configs)
        max_block = 0
        depth_dict = {}
        for i in range(num_file):
            if os.path.isfile(nas_configs[i]):
                config_dict = json.load(open(nas_configs[i]))
                depth = config_dict['enum']['depth']
                if depth in depth_dict:
                    depth_dict[depth].append(network_infos[i])
                else:
                    depth_dict[depth]=[network_infos[i]]
        for depth in depth_dict:
            network_infos_by_depth = list(map(self._extract_graph_and_score, depth_dict[depth]))
            num_block = self._merge_by_depth(network_infos_by_depth, depth)
            max_block = num_block if num_block>max_block else max_block
        return max_block

    def _write(self, net, f2):
        s = ""
        s += net.graph_part
        for score in net.score_list:
            s += ' ' + str(score)
        f2.append(s)

    def _extract_graph_and_score(self, file):
        if not os.path.isfile(file):
            return []
        f = open(file,'r',encoding='UTF-8')
        lines = f.readlines()
        net = Net(0, 0)
        pre_block = ''
        index = 0
        result = []
        for line in lines:
            if 'block_num' in line:
                self._write(net, result)
                block_num = int(line.split('block_num:')[1].split('round')[0])
                network_left = int(line.split('network_left: ')[1].split('network_id')[0])
                net = Net(block_num, network_left)
                if network_left == 1:
                    index += 1
            if 'graph_part' in line:
                graph_part = line.split('graph_part:')[1][:-1]
                net.graph_part = pre_block + '+' + graph_part
                if index == 1:
                    pre_block = pre_block + '+' + graph_part
                    index = 0
            if 'score' in line:
                score = float(line.split('score:')[1][:-1])
                net.score_list.append(score)
        f.close()
        return result

    def _merge_by_depth(self, network_infos_by_depth, depth):
        struct_set = []
        score_set = []
        max_block_id = 0
        min = 6
        for network_info in network_infos_by_depth:
            for line in network_info[1:]:
                data = line.strip().split("]] ")
                if len(data)<2:
                    continue
                data[0] = data[0] + "]] "
                score = list(map(float, data[1].split(" ")))
                score = [0. if i < 0 else i for i in score]
                min = len(score) if len(score) < min else min

                struct = list(map(eval, data[0].split("+")[1:]))
                if len(struct) > max_block_id:
                    max_block_id = len(struct)
                if struct not in struct_set:
                    struct_set.append(struct)
                    score_set.append(score[:min])
                else:
                    i = struct_set.index(struct)
                    score_set[i].extend(score[:min])
        data_x = [[] for i in range(max_block_id)]
        label = [[] for i in range(max_block_id)]
        for struct, score in zip(struct_set, score_set):
            struct = list(map(self.cops.trans, struct))
            struct = list(map(self.cops.list2mat, struct))
            block_num = len(struct)
            struct = self.cops.concat(struct)
            data_x[block_num - 1].append(struct)

            mean = np.mean(score)
            var = np.var(score)
            label[block_num - 1].append(mean + var)

        data_x = [[Feature(x)._feature_nodes() for x in i] for i in data_x]
        data_x = [[self.cops.padding(x, 71) for x in i] for i in data_x]
        block_id = 1
        for block_set, label_set in zip(data_x, label):
            train_data = []
            for i in range(len(block_set)):
                for j in range(len(block_set)):
                    if i != j:
                        a = 0 if label_set[i] < label_set[j] else 1
                        train_data.append([block_set[i], block_set[j], a])
            train_data = np.array(train_data)
            path = self.root_path+'\\'+self.scene+'\\'
            filename = str(depth) + '-'+str(block_id) + '.npy'
            self._save_file(path, filename, train_data)
            block_id += 1
        return block_id-1

    def _save_file(self, path, filename, data):
        path = path.strip()
        path = path.rstrip("\\")
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+'\\'+filename, data)


class Net:
    def __init__(self, block_num, network_left):
        self.block_num = block_num
        self.network_left = network_left
        self.score_list = []
        self.graph_part = ''


class NetworkEvaluation:
    def __init__(self,
                 lstm_units=64,
                 features=25,
                 seq_length=71,
                 epoch=100,
                 save_step=5000,
                 lr=0.0001):
        self.lstm_units = lstm_units
        self.features = features
        self.seq_length = seq_length
        self.epoch = epoch
        self.save_step = save_step
        self.lr = lr
        self.X_1 = tf.placeholder(tf.float32, [None, seq_length, features])
        self.X_2 = tf.placeholder(tf.float32, [None, seq_length, features])
        self.Y = tf.placeholder(tf.float32, [None, 2])
        self.cops = CommonOps()
        self._params_init()
        self.sess = tf.Session()

    def _params_init(self):
        with tf.variable_scope("lstm_1"):
            self.kernel_w_1 = tf.get_variable("kernel_1",
                                              shape=[self.features, 4*self.lstm_units],
                                              initializer=tf.glorot_uniform_initializer)
            self.current_w_1 = tf.get_variable("current_1",
                                               shape=[self.lstm_units, 4*self.lstm_units],
                                               initializer=tf.orthogonal_initializer)
            self.bias_1 = tf.get_variable("bias_1",
                                          shape=[1, 4*self.lstm_units])
        with tf.variable_scope("lstm_2"):
            self.kernel_w_2 = tf.get_variable("kernel_2",
                                              shape=[self.features, 4 * self.lstm_units],
                                              initializer=tf.glorot_uniform_initializer)
            self.current_w_2 = tf.get_variable("current_2",
                                               shape=[self.lstm_units, 4 * self.lstm_units],
                                               initializer=tf.orthogonal_initializer)
            self.bias_2 = tf.get_variable("bias_2",
                                          shape=[1, 4 * self.lstm_units])

    def _lstm_layer(self, inputs, kernel_w, current_w):
        shape = tf.shape(inputs)
        prev_h = tf.zeros([shape[0], self.lstm_units], tf.float32)
        prev_c = tf.zeros([shape[0], self.lstm_units], tf.float32)
        output = []
        for i in range(inputs.shape[1]):
            x = inputs[:, i, :]
            prev_h, prev_c = self.cops.lstm_cell(x,
                                       units=self.lstm_units,
                                       prev_h=prev_h,
                                       prev_c=prev_c,
                                       kernel_w=kernel_w,
                                       current_w=current_w,
                                       bias=self.bias_1)
            output.append(prev_h)
        output = tf.concat(output, 1)
        return output

    def _dense_layer(self, inputs, units, use_bias=True):
        output = tf.layers.dense(inputs,
                                 units=units,
                                 use_bias=use_bias,
                                 kernel_initializer=tf.glorot_uniform_initializer)
        return output

    def _build_model(self):
        x_1 = self._lstm_layer(inputs=self.X_1,
                               kernel_w=self.kernel_w_1,
                               current_w=self.current_w_1)

        x_2 = self._lstm_layer(inputs=self.X_2,
                               kernel_w=self.kernel_w_1,
                               current_w=self.current_w_1)

        x = tf.concat([x_1, x_2], 1)
        x = self._dense_layer(inputs=x, units=1024)
        x = self._dense_layer(inputs=x, units=512)
        logits = self._dense_layer(inputs=x, units=2)
        logits = tf.nn.softmax(logits)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.Y)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.9).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
        accaurate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return optimizer, accaurate, x

    def save_model(self, path):
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=self.sess,
            input_graph_def=input_graph_def,
            output_node_names=["Softmax"]
        )
        with tf.gfile.GFile(path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    def reset_graph(self):
        tf.reset_default_graph()

    def train(self,train_x_1, train_x_2, train_label, val_x_1, val_x_2, val_label):
        train_op, acc, loss = self._build_model()
        init = tf.global_variables_initializer()
        batch_size = 128
        self.sess.run(init)
        for ep in range(self.epoch):
            max_step = len(train_x_1)//batch_size
            for step in range(max_step):
                start = step*batch_size%len(train_x_1)
                end = min(start+batch_size, len(train_x_1))
                batch_x_1 = train_x_1[start:end]
                batch_x_2 = train_x_2[start:end]
                batch_label = train_label[start:end]
                self.sess.run(train_op, feed_dict={self.X_1:batch_x_1, self.X_2:batch_x_2, self.Y:batch_label})
                if ep%10==0 and step==0:
                    train_acc=self.sess.run(acc, feed_dict={self.X_1:batch_x_1, self.X_2:batch_x_2, self.Y:batch_label})
                    count=0
                    for j in range(math.ceil(len(val_label)/batch_size)):
                        count+=self.sess.run(acc, feed_dict={self.X_1:val_x_1[j*batch_size:(j+1)*batch_size],
                                                        self.X_2:val_x_2[j*batch_size:(j+1)*batch_size],
                                                        self.Y:val_label[j*batch_size:(j+1)*batch_size]})
                    eval_acc=count/math.ceil(len(val_label)/batch_size)
                    print('train acc: {train}, eval_acc: {eval}'.format(train=train_acc, eval=eval_acc))



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
        config = os.path.join(cur_path, 'nas_config.json')
        config = json.load(open(config))
        scene_name = config['eva']['task_name']
        model_path = os.path.join(MODEL_TEMPLATE_PATH, scene_name,'block'+str(block_id)+'.pb')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_TEMPLATE_PATH, 'block'+str(block_id)+'.pb')
        with tf.Session() as sess:
            with tf.gfile.FastGFile(model_path, 'rb') as f:
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

    def train_model(self, root_path, scene):
        process = DataProcess(root_path, scene)
        num_block = process.read_data()
        save_path = os.path.join(MODEL_TEMPLATE_PATH, scene)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for block_id in range(num_block):
            train_x_1, train_x_2, label, val_x_1, val_x_2, val_label = process.read_data_by_block(block_id+1)
            ne = NetworkEvaluation()
            print('trainning model block'+str(block_id+1))
            ne.train(train_x_1, train_x_2, label, val_x_1, val_x_2, val_label)
            print('save model block '+str(block_id+1)+'...')
            filepath = os.path.join(save_path, 'block'+str(block_id+1)+'.pb')
            ne.save_model(filepath)
            ne.reset_graph()
            print('save model success')


if __name__ == "__main__":
    # a = [[[[1], [2], []], [[1], [2], []]], [[[1], [2], []], [[1, 3], [2], [], [2]]]]
    # b = [[[[1], [2], []], [[1, 3], [2], [], [2]]], [[[1], [2], []], [[1], [2], []]]]
    # TopologyEval().topo1vstopo2(a, b, 2)
    te = TopologyEval()
    te.train_model(root_path='D:\\工作\\NAS项目', scene='c100')
    process = DataProcess(root_path='D:\\工作\\NAS项目\\refactor_exp\\huawei', scene='c100')
    num_block = process.read_data()
    print(num_block)
