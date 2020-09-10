import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(input_ckpt_path):
    # Before exporting our graph, we need to decide what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    # output_node_names = "img_inputs,dropout_rate,resnet_v1_50/E_BN2/Identity"
    output_node_names = "last_layer0"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    # import the meta graph and get a Saver
    saver = tf.train.import_meta_graph(meta_graph_or_file="{}.meta".format(input_ckpt_path),
                                       clear_devices=clear_devices,
                                       # input_map={}
                                       )
    # GraphDef object
    input_graph_def = tf.get_default_graph().as_graph_def()

    # start a session
    with tf.Session() as sess:
        # restore the graph weights
        saver.restore(sess, input_ckpt_path)

        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
#            output_node_names=output_node_names.split(',')
            output_node_names=[output_node_names]
        )
        output_pb_path = input_ckpt_path + ".pb"
        # serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_pb_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        # print('{} ops in the output graph'.format(len(output_graph_def.node)))
        # for op in output_graph_def.node:
        #     print(op.name, op.op)


if __name__ == '__main__':
    freeze_graph('./model/model0')

