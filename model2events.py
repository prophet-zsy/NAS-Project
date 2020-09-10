import tensorflow as tf
from tensorflow.python.platform import gfile

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()

# #这是从文件格式的meta文件加载模型
# # graphdef.ParseFromString(gfile.FastGFile("/data/TensorFlowAndroidMNIST/app/src/main/expert-graph.pb", "rb").read())
# # _ = tf.import_graph_def(graphdef, name="")
# _ = tf.train.import_meta_graph("./model/model0.meta")
# summary_write = tf.summary.FileWriter("./log" , graph)

# #这是从文件格式的pb文件加载模型
graphdef.ParseFromString(gfile.FastGFile("./model/model0.pb", "rb").read())
_ = tf.import_graph_def(graphdef, name="")

summary_write = tf.summary.FileWriter("./log" , graph)