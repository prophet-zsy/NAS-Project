import tensorflow as tf
import numpy as np
import pickle, random
import sys, os, time, copy

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG, Logger, EvaScheduleItem
EVA_COFIG = NAS_CONFIG['eva']

# config params for eva alone
INSTANT_PRINT = False  # set True only in run the eva alone
IMAGE_SIZE = 32
DATA_RATIO_FOR_EVAL = 0.2
TASK_NAME = EVA_COFIG['task_name']
DATA_PATH = "./data/"
MODEL_PATH = "./model"
BATCH_SIZE = 50
INITIAL_LEARNING_RATE = 0.025
WEIGHT_DECAY = 0.0003
MOMENTUM_RATE = 0.9

def _open_a_Session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

class DataSet:
    def __init__(self):
        self.num_classes = 10 if TASK_NAME == "cifar-10" else 100
        self.train_data, self.train_label, self.valid_data, self.valid_label,\
             self.test_data, self.test_label = self.inputs()
    
    def get_train_data(self, data_size):
        return self.train_data[:data_size], self.train_label[:data_size]

    def inputs(self):
        if INSTANT_PRINT:
            print("======Loading data======")
        if TASK_NAME == 'cifar-10':
            test_files = ['test_batch']
            train_files = ['data_batch_%d' % d for d in range(1, 6)]
        else:
            train_files = ['train']
            test_files = ['test']
        train_data, train_label = self._load(train_files)
        train_data, train_label, valid_data, valid_label = \
            self._shuffle_and_split_valid(train_data, train_label)
        test_data, test_label = self._load(test_files)
        if INSTANT_PRINT:
            print("======Data Process Done======")
        return train_data, train_label, valid_data, valid_label, test_data, test_label

    def _load_one(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        data = batch[b'data']
        label = batch[b'labels'] if TASK_NAME == 'cifar-10' else batch[b'fine_labels']
        return data, label

    def _load(self, files):
        file_name = 'cifar-10-batches-py' if TASK_NAME == 'cifar-10' else 'cifar-100-python'
        data_dir = os.path.join(DATA_PATH, file_name)
        data, label = self._load_one(os.path.join(data_dir, files[0]))
        for f in files[1:]:
            batch_data, batch_label = self._load_one(os.path.join(data_dir, f))
            data = np.append(data, batch_data, axis=0)
            label = np.append(label, batch_label, axis=0)
        label = np.array([[float(i == label)
                           for i in range(self.num_classes)] for label in label])
        data = data.reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE])
        data = data.transpose([0, 2, 3, 1])
        # pre-process
        data = self._normalize(data)
        return data, label

    def _shuffle_and_split_valid(self, data, label):
        # shuffle
        data_num = len(data)
        index = [i for i in range(data_num)]
        random.shuffle(index)
        data = data[index]
        label = label[index]

        eval_trian_bound = int(data_num * DATA_RATIO_FOR_EVAL)
        train_data = data[eval_trian_bound:]
        train_label = label[eval_trian_bound:]
        valid_data = data[:eval_trian_bound]
        valid_label = label[:eval_trian_bound]
        return train_data, train_label, valid_data, valid_label

    def _normalize(self, x_train):
        x_train = x_train.astype('float32')
        x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0]))\
             / np.std(x_train[:, :, :, 0])
        x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1]))\
             / np.std(x_train[:, :, :, 1])
        x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2]))\
             / np.std(x_train[:, :, :, 2])
        return x_train

    def data_augment(self, x):
        x = copy.deepcopy(x)
        x = self._random_flip_leftright(x)
        x = self._random_crop(x, [32, 32], 4)
        x = self._cutout(x)
        return x

    def _random_crop(self, batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        if padding:
            oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                           nw:nw + crop_shape[1]]
        return np.array(new_batch)

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def _cutout(self, x):
        for i in range(len(x)):
            cut_size = random.randint(0, IMAGE_SIZE // 2)
            s = random.randint(0, IMAGE_SIZE - cut_size)
            x[i, s:s + cut_size, s:s + cut_size, :] = 0
        return x

class DataFlowGraph:
    def __init__(self, task_item):  # DataFlowGraph object and task_item are one to one correspondent
        self.num_classes = 10 if TASK_NAME == "cifar-10" else 100
        self.input_shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]
        self.output_shape = [BATCH_SIZE, self.num_classes]
        
        self.task_item = task_item
        self.data_size = task_item.data_size
        self.epoch = task_item.epoch

        self.net_item = task_item.network_item  # is None when retrain
        self.pre_block = task_item.pre_block
        self.cur_block_id = len(self.pre_block)
        self.ft_sign = task_item.ft_sign
        self.is_bestNN = task_item.is_bestNN
        self.repeat_num = NAS_CONFIG['nas_main']['repeat_num']
        # we add a pooling layer for last repeat block if the following signal is set true
        self.use_pooling_blk_end = NAS_CONFIG['nas_main']['link_node']
        
        # need to feed sth. when training
        self.data_x, self.data_y = None, None
        self.train_flag = False
        self.run_ops = {}  # what sess.run
        self.saver = None
        self._construct_graph()

    def _find_ends(self):
        if self.cur_block_id > 0 and self.net_item:  # if there are some previous blocks and not in retrain mode
            self._load_pre_model()
            graph = tf.get_default_graph()
            data_x = graph.get_tensor_by_name("input:0")
            data_y = graph.get_tensor_by_name("label:0")
            train_flag = graph.get_tensor_by_name("train_flag:0")
            mid_plug = graph.get_tensor_by_name("block{}/last_layer:0".format(self.cur_block_id - 1))
            if not self.ft_sign:
                mid_plug = tf.stop_gradient(mid_plug, name="stop_gradient")
        else:  # if it's the first block or in retrain mode
            data_x = tf.placeholder(tf.float32, self.input_shape, name='input')
            data_y = tf.placeholder(tf.int32, self.output_shape, name="label")
            train_flag = tf.placeholder(tf.bool, name='train_flag')
            mid_plug = tf.identity(data_x)
        return data_x, data_y, train_flag, mid_plug

    def _construct_graph(self):
        tf.reset_default_graph()
        self.data_x, self.data_y, self.train_flag, mid_plug = self._find_ends()
        # tf.summary.histogram('mid_plug', mid_plug)
        if self.net_item:
            blks = [[self.net_item.graph, self.net_item.cell_list]]
            mid_plug = self._construct_nblks(mid_plug, blks, self.cur_block_id)
        else:  # retrain mode
            blks = [[net_item.graph, net_item.cell_list] for net_item in self.pre_block]
            mid_plug = self._construct_nblks(mid_plug, blks, first_blk_id=0)
        logits = tf.nn.dropout(mid_plug, keep_prob=1.0)
        logits = self._makedense(logits, ('', [self.num_classes], ''), with_bias=False)
        tf.summary.histogram('last_layer_logits', logits)
        global_step = tf.Variable(0, trainable=False, name='global_step' + str(self.cur_block_id))
        accuracy = self._cal_accuracy(logits, self.data_y)
        loss = self._loss(logits, self.data_y)
        train_op = self._train_op(global_step, loss)
        merged = tf.summary.merge_all()
        self.run_ops['logits'] = logits
        self.run_ops['merged'] = merged

    def _cal_accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.run_ops['acc'] = accuracy
        return accuracy

    def _loss(self, logits, labels):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = cross_entropy + l2 * WEIGHT_DECAY
        # tf.summary.scalar('cross_entropy', cross_entropy)
        # tf.summary.scalar('l2', l2)
        # tf.summary.scalar('loss', loss)
        self.run_ops['loss'] = loss
        self.run_ops['l2_loss'] = l2
        self.run_ops['ce'] = cross_entropy
        return loss

    def _train_op(self, global_step, loss):
        num_batches_per_epoch = self.data_size / BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * self.epoch)
        lr = tf.train.cosine_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, 0.0001)

        opt = tf.train.MomentumOptimizer(lr, MOMENTUM_RATE, name='Momentum' + str(self.cur_block_id),
                                         use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
        self.run_ops['train_op'] = train_op
        self.run_ops['lr'] = lr
        return train_op

    @staticmethod
    def _pad(inputs1, inputs2):
        # padding
        a = tf.shape(inputs2)[1]
        b = tf.shape(inputs1)[1]
        pad = tf.abs(tf.subtract(a, b))
        output = tf.where(tf.greater(a, b), 
                          tf.concat([tf.pad(inputs1, [[0, 0], [0, pad], [0, pad], [0, 0]]), inputs2], 3),
                          tf.concat([inputs1, tf.pad(inputs2, [[0, 0], [0, pad], [0, pad], [0, 0]])], 3))
        return output

    @staticmethod
    def _recode_repeat_blk(graph_full, cell_list, repeat_num):
        new_graph = [] + graph_full
        new_cell_list = [] + cell_list
        add = 0
        for i in range(repeat_num - 1):
            new_cell_list += cell_list
            add += len(graph_full)
            for sub_list in graph_full:
                new_graph.append([x + add for x in sub_list])
        return new_graph, new_cell_list

    def _construct_blk(self, blk_input, graph_full, cell_list, train_flag, blk_id):
        topo_order = self._toposort(graph_full)
        nodelen = len(graph_full)
        # input list for every cell in network
        inputs = [blk_input for _ in range(nodelen)]
        # bool list for whether this cell has already got input or not
        getinput = [False for _ in range(nodelen)]
        getinput[0] = True
        with tf.variable_scope('block' + str(blk_id)) as scope:
            for node in topo_order:
                layer = self._make_layer(inputs[node], cell_list[node], node, train_flag)
                self.run_ops['blk{}_node{}'.format(blk_id, node)] = layer
                # tf.summary.histogram('layer'+str(node), layer)
                # update inputs information of the cells below this cell
                for j in graph_full[node]:
                    if getinput[j]:  # if this cell already got inputs from other cells precedes it
                        inputs[j] = self._pad(inputs[j], layer)
                    else:
                        inputs[j] = layer
                        getinput[j] = True
            # give last layer a name
            last_layer = tf.identity(layer, name="last_layer")
        return last_layer

    def _construct_nblks(self, mid_plug, blks, first_blk_id):
        blk_id = first_blk_id
        for blk in blks:
            graph_full, cell_list = blk
            graph_full, cell_list = self._recode_repeat_blk(graph_full, cell_list, self.repeat_num)
            # add the last node
            graph_full = graph_full + [[]]
            if self.use_pooling_blk_end:
                cell_list = cell_list + [Cell('pooling', 'max', 2)]
            else:
                cell_list = cell_list + [Cell('id', 'max', 1)]
            mid_plug = self._construct_blk(mid_plug, graph_full, cell_list, self.train_flag, blk_id)
            self.run_ops['block{}_end'.format(blk_id)] = mid_plug
            blk_id += 1
        return mid_plug

    def _toposort(self, graph):
        node_len = len(graph)
        in_degrees = dict((u, 0) for u in range(node_len))
        for u in range(node_len):
            for v in graph[u]:
                in_degrees[v] += 1
        queue = [u for u in range(node_len) if in_degrees[u] == 0]
        result = []
        while queue:
            u = queue.pop()
            result.append(u)
            for v in graph[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    queue.append(v)
        return result

    def _make_layer(self, inputs, cell, node, train_flag):
        if cell.type == 'conv':
            layer = self._makeconv(inputs, cell, node, train_flag)
        elif cell.type == 'pooling':
            layer = self._makepool(inputs, cell)
        elif cell.type == 'id':
            layer = tf.identity(inputs)
        elif cell.type == 'sep_conv':
            layer = self._makesep_conv(inputs, cell, node, train_flag)
        # TODO add any other new operations here
        else:
            assert False, "Wrong cell type!"
        return layer

    def _makeconv(self, x, hplist, node, train_flag):
        with tf.variable_scope('conv' + str(node)) as scope:
            inputdim = x.shape[3]
            kernel = self._get_variable('weights',
                                        shape=[hplist.kernel_size, hplist.kernel_size, inputdim, hplist.filter_size])
            x = self._activation_layer(hplist.activation, x, scope)
            x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            x = self._batch_norm(tf.nn.bias_add(x, biases), train_flag)
        return x

    def _makesep_conv(self, inputs, hplist, node, train_flag):
        with tf.variable_scope('conv' + str(node)) as scope:
            inputdim = inputs.shape[3]
            dfilter = self._get_variable('weights', shape=[hplist.kernel_size, hplist.kernel_size, inputdim, 1])
            pfilter = self._get_variable('pointwise_filter', [1, 1, inputdim, hplist.filter_size])
            conv = tf.nn.separable_conv2d(inputs, dfilter, pfilter, strides=[1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            bn = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            conv_layer = self._activation_layer(hplist.activation, bn, scope)
        return conv_layer

    def _batch_norm(self, input, train_flag):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            updates_collections=None, is_training=train_flag)

    def _get_variable(self, name, shape):
        if name == "weights":
            ini = tf.contrib.keras.initializers.he_normal()
        else:
            ini = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape, initializer=ini)

    def _activation_layer(self, type, inputs, scope):
        if type == 'relu':
            layer = tf.nn.relu(inputs, name=scope.name)
        elif type == 'relu6':
            layer = tf.nn.relu6(inputs, name=scope.name)
        elif type == 'tanh':
            layer = tf.tanh(inputs, name=scope.name)
        elif type == 'sigmoid':
            layer = tf.sigmoid(inputs, name=scope.name)
        elif type == 'leakyrelu':
            layer = tf.nn.leaky_relu(inputs, name=scope.name)
        else:
            layer = tf.identity(inputs, name=scope.name)
        return layer

    def _makepool(self, inputs, hplist):
        if hplist.pooling_type == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.pooling_type == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.pooling_type == 'global':
            return tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    def _makedense(self, inputs, hplist, with_bias=True):
        inputs = tf.reshape(inputs, [BATCH_SIZE, -1])
        for i, neural_num in enumerate(hplist[1]):
            with tf.variable_scope('block' + str(self.cur_block_id) + 'dense' + str(i)) as scope:
                weights = self._get_variable('weights', shape=[inputs.shape[-1], neural_num])
                # tf.summary.histogram('dense_weights', weights)
                mul = tf.matmul(inputs, weights)
                if with_bias:
                    biases = self._get_variable('biases', [neural_num])
                    # tf.summary.histogram('dense_biases', biases)
                    mul += biases
                if neural_num == self.output_shape[-1]:
                    local3 = self._activation_layer('', mul, scope)
                else:
                    local3 = self._activation_layer(hplist[2], mul, scope)
            inputs = local3
        return inputs

    def _load_pre_model(self):  # for evaluate 
        front_blk_model_path = os.path.join(MODEL_PATH, 'model' + str(self.cur_block_id-1))
        assert os.path.exists(front_blk_model_path+".meta")
        self.saver = tf.train.import_meta_graph(front_blk_model_path+".meta")

    def _load_model(self):  # for test in retrain 
        graph = tf.get_default_graph()
        sess = _open_a_Session()
        front_blk_model_path = os.path.join(MODEL_PATH, 'model' + str(self.cur_block_id))
        assert os.path.exists(front_blk_model_path+".meta"), "model we will load does not exist"
        self.saver = tf.train.import_meta_graph(front_blk_model_path+".meta")
        self.saver.restore(sess, front_blk_model_path)
        self.data_x = graph.get_tensor_by_name("input:0")
        self.data_y = graph.get_tensor_by_name("label:0")
        self.train_flag = graph.get_tensor_by_name("train_flag:0")
        return sess

    def _save_model(self, sess):  # for evaluate and retrain
        saver = tf.train.Saver(tf.global_variables())
        if self.is_bestNN:
            if not os.path.exists(os.path.join(MODEL_PATH)):
                os.makedirs(os.path.join(MODEL_PATH))
            saver.save(sess, os.path.join(MODEL_PATH, 'model' + str(self.cur_block_id)))

    def _stats_graph(self):
        graph = tf.get_default_graph()
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        return flops.total_float_ops, params.total_parameters

class Evaluator:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.log = ''
        self.data_set = DataSet()
        self.epoch = 0
        self.data_size = 0

    def _set_epoch(self, e):
        self.epoch = e
        return self.epoch

    def _set_data_size(self, num):
        if num > len(list(self.data_set.train_label)) or num < 0:
            num = len(list(self.data_set.train_label))
            print('Warning! Data size has been changed to', num, ', all data is loaded.')
        assert num >= BATCH_SIZE, "data added should be more than one batch, and the batch is set {}".format(BATCH_SIZE)
        self.data_size = num
        return self.data_size

    def evaluate(self, task_item):
        self._log_item_info(task_item)
        computing_graph = DataFlowGraph(task_item)
        score = self._train(computing_graph, task_item)
        if not task_item.network_item:
            score = self._test(computing_graph)
        NAS_LOG = Logger()
        NAS_LOG << ('eva_eva', self.log)
        return score

    def _train(self, compute_graph, task_item):
        # get the data
        train_data, train_label = self.data_set.get_train_data(self.data_size)
        train_data = self.data_set.data_augment(train_data)  #TODO check here if it changed the original data???
        valid_data, valid_label = self.data_set.valid_data, self.data_set.valid_label

        sess = _open_a_Session()
        sess.run(tf.global_variables_initializer())
        if task_item.pre_block and task_item.network_item:  # if not in retrain and there are font blks
            compute_graph.saver.restore(sess, os.path.join(MODEL_PATH, 'model' + str(len(task_item.pre_block)-1)))

        for ep in range(self.epoch):
            if INSTANT_PRINT:
                print("epoch {}/{}".format(ep, self.epoch))
            start_epoch = time.time()
            # trian steps
            train_ops_keys = ['acc', 'loss', 'train_op']
            train_acc = self._iter_run_on_graph(train_data, train_label, train_ops_keys, compute_graph, sess, train_flag=True)
            # valid steps
            valid_ops_keys = ['acc', 'loss']
            valid_acc = self._iter_run_on_graph(valid_data , valid_label, valid_ops_keys, compute_graph, sess, train_flag=False)
            # writer = tf.summary.FileWriter('./log', sess.graph)
            # writer.add_summary(merged, step)
            epoch_log = 'epoch %d/%d: train_acc = %.3f, valid_acc = %.3f, cost time %.3f\n'\
                 % (ep, self.epoch, train_acc, valid_acc, float(time.time() - start_epoch))
            self.log += epoch_log
            if INSTANT_PRINT:
                print(epoch_log)

        compute_graph._save_model(sess)
        sess.close()
        return valid_acc

    def _test(self, compute_graph):
        # tf.reset_default_graph()
        sess = compute_graph._load_model()
        test_data, test_label = self.data_set.test_data, self.data_set.test_label
        test_ops_keys = ['acc', 'loss']
        acc = self._iter_run_on_graph(test_data, test_label, test_ops_keys, compute_graph, sess, train_flag=False, is_test=True)
        test_log = "test_acc: {}\n".format(acc)
        self.log += test_log
        if INSTANT_PRINT:
            print(test_log)
        sess.close()
        return acc

    def _iter_run_on_graph(self, data, label, run_ops_keys, compute_graph, sess, train_flag, is_test=False, batch_size=BATCH_SIZE):
        max_steps = len(label) // batch_size
        acc_cur_epoch = 0
        for step in range(max_steps):
            batch_x = data[step * batch_size:(step + 1) * batch_size]
            batch_y = label[step * batch_size:(step + 1) * batch_size]
            run_ops = [compute_graph.run_ops[key] for key in run_ops_keys]
            result = sess.run(run_ops, feed_dict={compute_graph.data_x: batch_x, \
                compute_graph.data_y: batch_y, compute_graph.train_flag: train_flag})
            acc, loss = result[0], result[1]
            acc_cur_epoch += acc / max_steps
            if INSTANT_PRINT and step % 50 == 0:
                stage_type = 'train' if train_flag else 'valid'
                stage_type = 'test' if is_test else stage_type
                print(">>%s %d/%d loss %.4f acc %.4f" % (stage_type, step, max_steps, loss, acc))
        return acc_cur_epoch

    def _cal_multi_target(self, compute_graph, precision, time):
        flops, model_size = compute_graph._stats_graph()
        return precision + 1 / time + 1 / flops + 1 / model_size

    def _log_item_info(self, task_item):
        #  we record the eva info in self.log and write it into the eva file once
        self.log = ''  # reset the self.log
        if task_item.network_item:  # not in retrain mode
            self.log += "-"*20+"blk_id:"+str(len(task_item.pre_block))+" nn_id:"+str(task_item.nn_id)\
                        +" item_id:"+str(task_item.network_item.id)+"-"*20+'\n'
            for block in task_item.pre_block:
                self.log += str(block.graph) + str(block.cell_list) + '\n'
            self.log += str(task_item.network_item.graph) +\
                        str(task_item.network_item.cell_list) + '\n'
        else:  # in retrain mode
            self.log += "-"*20+"retrain"+"-"*20+'\n'
            for block in task_item.pre_block:
                self.log += str(block.graph) + str(block.cell_list) + '\n'
        if INSTANT_PRINT:
            print(self.log)


if __name__ == '__main__':
    INSTANT_PRINT = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    eval = Evaluator()
    cur_data_size = eval._set_data_size(-1)
    cur_epoch = eval._set_epoch(200)
    
    # graph_full = [[1]]
    # cell_list = [Cell('conv', 64, 3, 'relu')]
    # network1 = NetworkItem(0, graph_full, cell_list, "")
    # task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network1,\
    #      pre_blk=[], ft_sign=True, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=20, data_size=cur_data_size)
    # e = eval.evaluate(task_item)
    # print(e)

    # retrain for 5/26 exp result
    graph_full = [[1, 4, 5, 3], [2, 6, 3], [3, 7], [7], [2], [2], [3]]
    cell_list = [Cell('conv', 64, 1, 'relu6'), Cell('conv', 32, 5, 'relu'), Cell('conv', 32, 5, 'leakyrelu'), 
                 Cell('sep_conv', 48, 3, 'relu'), Cell('conv', 32, 1, 'relu'), Cell('sep_conv', 32, 3, 'leakyrelu'), 
                 Cell('conv', 32, 3, 'relu6')]
    network1 = NetworkItem(0, graph_full, cell_list, "")
    graph_full = [[1, 4, 5, 6, 2, 3], [2, 3], [3], [8], [2], [3], [7], [3]]
    cell_list = [Cell('conv', 128, 5, 'relu6'), Cell('conv', 64, 5, 'relu6'), Cell('sep_conv', 64, 3, 'relu6'), 
                 Cell('conv', 48, 1, 'relu'), Cell('sep_conv', 128, 5, 'leakyrelu'), Cell('sep_conv', 64, 1, 'leakyrelu'), 
                 Cell('sep_conv', 48, 1, 'leakyrelu'), Cell('conv', 64, 3, 'relu')]
    network2 = NetworkItem(1, graph_full, cell_list, "")
    graph_full = [[1, 4, 5, 7, 2], [2, 3, 9], [3, 9], [9], [2], [6], [3], [8], [3]]
    cell_list = [Cell('conv', 128, 1, 'relu6'), Cell('conv', 64, 5, 'relu'), Cell('conv', 192, 3, 'leakyrelu'), 
                 Cell('sep_conv', 64, 1, 'relu6'), Cell('conv', 192, 3, 'relu6'), Cell('sep_conv', 192, 1, 'relu'), 
                 Cell('conv', 64, 1, 'relu6'), Cell('conv', 128, 1, 'relu'), Cell('conv', 192, 5, 'leakyrelu')]
    network3 = NetworkItem(2, graph_full, cell_list, "")
    graph_full = [[1, 4, 5], [2, 3, 7], [3, 7], [7], [2], [6], [3]]
    cell_list = [Cell('conv', 256, 1, 'relu6'), Cell('sep_conv', 192, 1, 'relu'), Cell('conv', 256, 5, 'relu'), 
                 Cell('conv', 256, 5, 'leakyrelu'), Cell('sep_conv', 192, 1, 'leakyrelu'), Cell('conv', 128, 1, 'relu'), 
                 Cell('conv', 192, 1, 'relu6')]
    network4 = NetworkItem(3, graph_full, cell_list, "")

    # simulate search process
    # task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network1,\
    #      pre_blk=[], ft_sign=False, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    # e = eval.evaluate(task_item)

    # task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network2,\
    #      pre_blk=[network1], ft_sign=False, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    # e = eval.evaluate(task_item)

    # task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network3,\
    #      pre_blk=[network1, network2], ft_sign=False, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    # e = eval.evaluate(task_item)

    # task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=network4,\
    #      pre_blk=[network1, network2, network3], ft_sign=False, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    # e = eval.evaluate(task_item)

    # simulate retrain process
    task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=None,\
         pre_blk=[network1, network2, network3, network4], ft_sign=True, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    e = eval.evaluate(task_item)

