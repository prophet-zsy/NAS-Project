import math, time, os, sys, random, pickle, copy
import numpy as np
import tensorflow as tf
from datetime import datetime
from utils_1 import *
from image_ops import *
from common_ops import *

CUTOUT = True
BATCH_SIZE = 100
SEED = None
# number of epochs to decay
LR_DEC_EVERY = 10000
DATA_FORMAT = "NHWC"
DATA_PATH = "../data/cifar-10-batches-py"
NUM_CLASSES = 10
# DATA_PATH = "../data/cifar-100-python"
# NUM_CLASSES = 100

GPU = 1
NUM_VALIDS = 0
IMAGE_SIZE = 32
# Constants describing the training process.
INITIAL_LEARNING_RATE = 0.025  # Initial learning rate.
EPOCH = 500
MOMENTUM_RATE = 0.9
LOG_SAVE_PATH = './logs'
MODEL_SAVE_PATH = './model/'

NUM_TRAIN_BATCHES, NUM_VALID_BATCHES, NUM_TEST_BATCHES = -1, -1, -1



def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp,encoding='iso-8859-1')
      batch_images = data["data"].astype(np.float32) / 255.0
      labels_key = "fine_labels" if "100" in data_path else "labels"
      batch_labels = np.array(data[labels_key], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  print("-" * 80)
  print("Reading data")


  images, labels = {}, {}

  if "100" in data_path:
    train_files = [
      'train',
    ]
    test_file = [
      'test',
    ]
  elif "10" in data_path:
    train_files = [
      "data_batch_1",
      "data_batch_2",
      "data_batch_3",
      "data_batch_4",
      "data_batch_5",
    ]
    test_file = [
      "test_batch",
    ]
  else:
    print(data_path)
    raise Exception("data_path is wrong!")

  images["train"], labels["train"] = _read_data(data_path, train_files)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print ("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print ("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

def prepare_data(images, labels):
  with tf.device("/cpu:0"):
    # training data
    num_train_examples = np.shape(images["train"])[0]
    global NUM_TRAIN_BATCHES
    NUM_TRAIN_BATCHES = (num_train_examples + BATCH_SIZE - 1) // BATCH_SIZE
    x_train, y_train = tf.train.shuffle_batch(
      [images["train"], labels["train"]],
      batch_size=BATCH_SIZE,
      capacity=50000,
      enqueue_many=True,
      min_after_dequeue=0,
      num_threads=16,
      seed=SEED,
      allow_smaller_final_batch=True,
    )

    def _pre_process(x):
      x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
      x = tf.random_crop(x, [32, 32, 3], seed=SEED)
      x = tf.image.random_flip_left_right(x, seed=SEED)
      if CUTOUT:
        cut_size = random.randint(0, IMAGE_SIZE // 2)
        mask = tf.ones([cut_size, cut_size], dtype=tf.int32)
        start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
        mask = tf.pad(mask, [[cut_size + start[0], 32 - start[0]],
                              [cut_size + start[1], 32 - start[1]]])
        mask = mask[cut_size: cut_size + 32,
                    cut_size: cut_size + 32]
        mask = tf.reshape(mask, [32, 32, 1])
        mask = tf.tile(mask, [1, 1, 3])
        x = tf.where(tf.equal(mask, 0), x=x, y=tf.zeros_like(x))
      if DATA_FORMAT == "NCHW":
        x = tf.transpose(x, [2, 0, 1])

      return x
    x_train = tf.map_fn(_pre_process, x_train, back_prop=False)
    y_train = y_train

    # valid data
    x_valid, y_valid = None, None
    if images["valid"] is not None:
      images["valid_original"] = np.copy(images["valid"])
      labels["valid_original"] = np.copy(labels["valid"])
      if DATA_FORMAT == "NCHW":
        images["valid"] = tf.transpose(images["valid"], [0, 3, 1, 2])
      num_valid_examples = np.shape(images["valid"])[0]
      global NUM_VALID_BATCHES
      NUM_VALID_BATCHES = ((num_valid_examples + BATCH_SIZE - 1) // BATCH_SIZE)
      x_valid, y_valid = tf.train.batch(
        [images["valid"], labels["valid"]],
        batch_size=BATCH_SIZE,
        capacity=5000,
        enqueue_many=True,
        num_threads=1,
        allow_smaller_final_batch=True,
      )

    # test data
    if DATA_FORMAT == "NCHW":
      images["test"] = tf.transpose(images["test"], [0, 3, 1, 2])
    num_test_examples = np.shape(images["test"])[0]
    global NUM_TEST_BATCHES
    NUM_TEST_BATCHES = ((num_test_examples + BATCH_SIZE - 1) // BATCH_SIZE)
    x_test, y_test = tf.train.batch(
      [images["test"], labels["test"]],
      batch_size=BATCH_SIZE,
      capacity=10000,
      enqueue_many=True,
      num_threads=1,
      allow_smaller_final_batch=True,
    )

  return x_train, y_train, x_valid, y_valid, x_test, y_test



def conv_custom(inputs, blk_id, node_id, out_channel, kernel_size, activation, is_training=False):
  # make the channel double for 0.9656
  out_channel *= 2
  
  x = inputs[node_id]  # get the input correspond to the node
  # concat for all the skipping 
  x = tf.concat(x, 3)  # list to tensor
  # add for all the skipping 
  # new_x = x[0]
  # for item in x[1:]:
  #   new_x = new_x + item
  # x = new_x

  input_channel = x.get_shape()[3].value
  activation_dict = {
    "relu": lambda x: tf.nn.relu(x),
    "leakyrelu": lambda x: tf.nn.leaky_relu(x),
    "relu6":lambda x: tf.nn.relu6(x)
  }

  # add 1*1conv
  # with tf.variable_scope(str(blk_id) + '1_1_conv' + str(node_id)) as scope:    
  #   w = create_weight("w", [1, 1, input_channel, 48])  # when use it please change the following input_channel into 48
  #   x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=DATA_FORMAT)
  #   x = batch_norm(x, is_training, data_format=DATA_FORMAT)
  #   x = activation_dict["relu"](x)

  with tf.variable_scope(str(blk_id) + 'conv' + str(node_id)) as scope:    
    w = create_weight("w", [kernel_size, kernel_size, input_channel, out_channel])
    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=DATA_FORMAT)
    x = batch_norm(x, is_training, data_format=DATA_FORMAT)
    x = activation_dict[activation](x)

  # add 1*1conv for add ops at the end of skipping end, when using it we change the concat into add 
  # with tf.variable_scope(str(blk_id) + '1_1_conv' + str(node_id)) as scope:    
  #   w = create_weight("w", [1, 1, out_channel, 48])
  #   x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=DATA_FORMAT)
  #   x = batch_norm(x, is_training, data_format=DATA_FORMAT)
  #   x = activation_dict["relu"](x)
  return x

def toposort(graph):
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

# custom model
def model(images, labels, blks, is_training=False, reuse=False):
  """
  args:
    images : tensor
    labels : tensor
    blks : a list which represent the network
  """
  with tf.variable_scope('model', reuse=reuse) as scope:
    x = images
    for blk_id, blk in enumerate(blks):
      print("####build block {} ####".format(blk_id))
      graph, cell_list = blk[0], blk[1]
      graph = copy.deepcopy(graph)
      graph.append([])  # len(graph) = len(cell_list) + 1 at this time
      
      topo_order = toposort(graph)
      assert topo_order[0] == 0, "the first topo order id is not 0, the topo order is {}".format(topo_order)
      assert graph[topo_order[-1]] == [], "the last topo order node is not [], the graph is {}, topo order is {}".format(graph, topo_order)
      inputs = [[] for _ in range(len(graph))]
      inputs[0].append(x)
      for node_id in topo_order:
        if node_id == len(topo_order) - 1:
          break  # break when the last pooling
        out_channel = cell_list[node_id][1]; kernel_size = cell_list[node_id][2]; activation = cell_list[node_id][3]
        outputs = conv_custom(inputs=inputs, blk_id=blk_id, node_id=node_id,\
                        out_channel=out_channel, kernel_size=kernel_size, activation=activation, is_training=is_training)
        # append the output to where it should be put 
        for out_id in graph[node_id]:
          inputs[out_id].append(outputs)
      # last pooling
      x = inputs[topo_order[-1]]
      x = tf.concat(x, 3)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    
    x = tf.reduce_mean(x, [1, 2])  # global avg pooling
    inp_c = x.get_shape()[1].value
    w = create_weight("w", [inp_c, NUM_CLASSES])
    x = tf.matmul(x, w)
    # loss
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=x, labels=labels)
    loss = tf.reduce_mean(log_probs)

    train_preds = tf.argmax(x, axis=1)
    train_preds = tf.to_int32(train_preds)
    train_acc = tf.equal(train_preds, labels)
    train_acc = tf.to_int32(train_acc)
    train_acc = tf.reduce_sum(train_acc)

    return loss, train_acc


def train(blks, data_x, data_y):
  loss, train_acc = model(data_x, data_y, blks=blks, is_training=True, reuse=False)

  global_step = tf.Variable(0, trainable=False, name='global_step')
  decay_steps = int(NUM_TRAIN_BATCHES * EPOCH)
  lr = tf.train.cosine_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, 0.0001)
  # lr = tf.train.polynomial_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, 0.0, 0.5, cycle=True)

  opt = tf.train.MomentumOptimizer(lr, MOMENTUM_RATE, name='Momentum',
                                    use_nesterov=True)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
      train_op = opt.minimize(loss, global_step=global_step, name='train_op')
  
  return loss, train_acc, train_op

def valid(blks, data_x, data_y):
  loss, valid_acc = model(data_x, data_y, blks=blks, is_training=False, reuse=True)
  return loss, valid_acc

def test(blks, data_x, data_y):
  loss, test_acc = model(data_x, data_y, blks=blks, is_training=False, reuse=True)
  return loss, test_acc


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
  images, labels = read_data(DATA_PATH, num_valids=NUM_VALIDS)
  # print(images); print(labels)
  x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_data(images, labels)
  # print(x_train); print(y_train); print(x_valid); print(y_valid); print(x_test); print(y_test)

  # 0.960  c10
  blks = \
  [
  [[[1, 6, 9, 3], [2, 3, 4], [3, 4], [4, 10], [5], [10], [7], [8], [4], [5]], 
  [('conv', 64, 1, 'relu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 64, 5, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 32, 3, 'relu'), ('conv', 48, 5, 'relu'), ('conv', 64, 3, 'relu'), ('conv', 32, 5, 'relu'), ('conv', 48, 3, 'leakyrelu')]],
  [[[1, 2, 3], [2, 6], [3, 4], [4, 7, 5], [5], [8], [4], [5]], 
  [('conv', 128, 5, 'leakyrelu'), ('conv', 48, 5, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 128, 3, 'leakyrelu'), ('conv', 48, 3, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('conv', 64, 1, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu')]],
  [[[1, 6, 7, 2, 3], [2, 3, 4], [3, 4], [4, 5], [5], [9], [5], 
  [8], [5]], [('conv', 64, 3, 'leakyrelu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 5, 'relu6'), ('conv', 64, 1, 'relu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 192, 1, 'relu'), ('conv', 192, 3, 'leakyrelu')]],
  [[[1, 6, 2, 3], [2, 8, 3, 4], [3, 5], [4, 5], [5], [11], [7], [5], [9], [10], [5]], 
  [('conv', 192, 1, 'relu'), ('conv', 128, 1, 'relu'), ('conv', 256, 5, 'relu6'), ('conv', 128, 5, 'relu6'), ('conv', 192, 3, 'leakyrelu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 128, 3, 'relu6'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 256, 1, 'leakyrelu'), ('conv', 192, 3, 'relu6')]],
  ]

  # 0.9656 c10 (0.954 double channel)
  # blks = \
  # [
  # [[[1, 3, 4, 2, 6, 5], [2, 6], [6], [6], [5, 6], [6]],
  # [('conv', 64, 3, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 32, 3, 'relu'), ('conv', 32, 1, 'relu'), ('conv', 64, 3, 'relu')]],
  # [[[1, 3, 4, 2, 5, 6], [2, 6], [6], [2, 6], [5, 6], [6]],
  # [('conv', 128, 3, 'relu'), ('conv', 128, 3, 'relu'), ('conv', 192, 3, 'relu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 192, 3, 'relu'), ('conv', 128, 3, 'relu'), ('conv', 192, 3, 'relu')]],
  # [[[1, 3, 4], [2], [4], [4]],
  # [('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 192, 3, 'relu'), ('conv', 192, 3, 'leakyrelu'), ('conv', 256, 3, 'relu')]],
  # [[[1, 3, 4], [2], [4], [4]],
  # [('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 192, 3, 'relu'), ('conv', 192, 3, 'leakyrelu'), ('conv', 256, 3, 'relu')]],
  # ]

  train_loss, train_acc, train_op = train(blks, x_train, y_train)
  if NUM_VALIDS != 0:
    valid_loss, valid_acc = valid(blks, x_valid, y_valid)
  test_loss, test_acc = test(blks, x_test, y_test)
  print("$$$$  build finished   $$$$")

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # related to load data
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)

    sess.run(tf.global_variables_initializer())
    for ep in range(EPOCH):
      start_epoch = time.time()
      # train
      epoch_train_acc = 0
      for step in range(NUM_TRAIN_BATCHES):
        loss_val, acc_val, _ = sess.run([train_loss, train_acc, train_op])
        if step % 50 == 0:
            print("train %d/%d loss %.4f acc %.4f" % (step, NUM_TRAIN_BATCHES, loss_val, acc_val/BATCH_SIZE))
        epoch_train_acc += acc_val
      epoch_train_acc /= (NUM_TRAIN_BATCHES*BATCH_SIZE)

      # valid
      epoch_valid_acc = 0
      if NUM_VALIDS != 0:
        for step in range(NUM_VALID_BATCHES):
          loss_val, acc_val = sess.run([valid_loss, valid_acc])
          if step % 50 == 0:
              print("valid %d/%d loss %.4f acc %.4f" % (step, NUM_VALID_BATCHES, loss_val, acc_val/BATCH_SIZE))
          epoch_valid_acc += acc_val
        epoch_valid_acc /= (NUM_VALID_BATCHES*BATCH_SIZE)

      # test
      epoch_test_acc = 0
      for step in range(NUM_TEST_BATCHES):
        loss_val, acc_val = sess.run([test_loss, test_acc])
        if step % 50 == 0:
            print(">>test %d/%d loss %.4f acc %.4f" % (step, NUM_TEST_BATCHES, loss_val, acc_val/BATCH_SIZE))
        epoch_test_acc += acc_val
      epoch_test_acc /= (NUM_TEST_BATCHES*BATCH_SIZE)

      end_epoch = time.time()
      print(">>>>epoch %d/%d train_acc %.4f valid_acc %.4f test_acc %.4f cost_time %4d" % (ep, EPOCH, epoch_train_acc, epoch_valid_acc, epoch_test_acc, (end_epoch-start_epoch)))


