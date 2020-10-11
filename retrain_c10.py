import os, sys, time, glob, copy
import numpy as np
import torch
import cnn.utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1500, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
cnn.utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
# CIFAR_CLASSES = 100


# class vgg(nn.Module):

#     def __init__(self, num_classes):
#         super(vgg, self).__init__()
#         self.num_classes = num_classes
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, (2, 2)),
#             nn.Conv2d(64, 128, 3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, (2, 2)),
#             nn.Conv2d(128, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, (2, 2)),
#             nn.Conv2d(256, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, (2, 2)),
#             nn.Conv2d(512, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, (2, 2)),
#         )
#         self.classifier = nn.Linear(512, self.num_classes)

#     # [[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17]], 
#     # [('conv', 64, 3, 'relu'), ('conv', 64, 3, 'relu'), ('pooling', 'max', 2), ('conv', 128, 3, 'relu'),
#     # ('conv', 128, 3, 'relu'), ('pooling', 'max', 2), ('conv', 256, 3, 'relu'),
#     # ('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('pooling', 'max', 2),
#     # ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
#     # ('pooling', 'max', 2), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
#     # ('conv', 512, 3, 'relu')]

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0),-1)
#         x = self.classifier(x)
#         return x



class Network(nn.Module):

    def __init__(self, num_classes, blks):
        super(Network, self).__init__()
        self.num_classes = num_classes
        activation_dict = {
            "relu": nn.ReLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(inplace=True),
            "relu6": nn.ReLU6(inplace=True)
        }
        
        self.blks = blks
        self.incs = []
        for k in range(len(blks)):
            blk = blks[k]
            inc = [0 for _ in range(len(blk[0])+1)]
            graph = copy.deepcopy(blk[0])
            graph.append([])
            topo_order = self.toposort(graph)
            for i in topo_order:
                if i >= len(blk[0]):
                    continue  # for the last node, because orginal graph do not have last node 
                for edge in blk[0][i]:
                    if blk[1][i][0] == 'conv' or blk[1][i][0] == 'sep_conv':
                        # print(i, "adding", blk[1][i][1], "to", edge, "cur state", inc[edge])
                        inc[edge] += blk[1][i][1]
                    elif blk[1][i][0] == 'pooling':
                        inc[edge] += inc[i]
            if k == 0:  # from image
                inc[0] = 3
            else:  # from last blk
                inc[0] = self.incs[-1][-1]
            self.incs.append(inc)
        
        self.ops = nn.ModuleList()
        for i in range(len(blks)):
            op = nn.ModuleList()
            for j in range(len(blks[i][1])):
                if self.blks[i][1][j][0] == 'conv':
                    input_c = self.incs[i][j]
                    output_c = self.blks[i][1][j][1]
                    kernel_size = self.blks[i][1][j][2]
                    activation = self.blks[i][1][j][3]

                    conv = nn.Sequential(
                        nn.Conv2d(input_c, output_c, kernel_size, padding=int((kernel_size-1)/2), bias=False),
                        nn.BatchNorm2d(output_c),
                        activation_dict[activation]
                    )
                    conv.cuda()
                    op.append(conv)
                elif self.blks[i][1][j][0] == 'sep_conv':
                    input_c = self.incs[i][j]
                    output_c = self.blks[i][1][j][1]
                    kernel_size = self.blks[i][1][j][2]
                    activation = self.blks[i][1][j][3]
                    sep_conv = nn.Sequential(
                        nn.Conv2d(input_c, input_c, kernel_size, padding=int((kernel_size-1)/2), groups=input_c, bias=False),
                        nn.Conv2d(input_c, output_c, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(output_c),
                        activation_dict[activation]
                    )
                    sep_conv.cuda()
                    op.append(sep_conv)
                elif self.blks[i][1][j][0] == 'pooling':
                    typ = self.blks[i][1][j][1]   #  max
                    kernel_size = self.blks[i][1][j][2]
                    pool = nn.Sequential(
                        nn.MaxPool2d(kernel_size, stride=2)
                    )
                    pool.cuda()
                    op.append(pool)
            self.ops.append(op)
        
        # print(self.incs)
        # for i in range(len(self.ops)):
        #     for j in range(len(self.ops[i])):
        #         print(self.blks[i][1][j])
        #         print(self.ops[i][j])

        # print(self.ops)

        self.reduce_pool = nn.MaxPool2d(2, (2, 2))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # print(self.incs)
        inp_c = self.incs[-1][-1]
        # print(inp_c)
        self.classifier = nn.Linear(inp_c, self.num_classes)

        self.auxiliary_head = nn.ModuleList()
        for i in range(len(self.incs)):
            self.auxiliary_head.append(
                nn.Sequential(
                    nn.Conv2d(self.incs[i][-1], self.incs[i][-1], 1, padding=0, bias=False),
                    nn.BatchNorm2d(self.incs[i][-1]),
                    activation_dict['relu'],
                    nn.AdaptiveAvgPool2d(1)
                )
            )
        self.auxiliary_classifiers = nn.ModuleList()
        for i in range(len(self.incs)):
            self.auxiliary_classifiers.append(
                nn.Linear(self.incs[i][-1], self.num_classes)
            )

    def op_cutom(self, inputs, blk_id, node_id):  
        x = inputs[node_id]  # get the input correspond to the node
        # concat for all the skipping 
        x = torch.cat(x, 1)  # list to tensor, note : it is NCHW
        # add for all the skipping 
        # new_x = x[0]
        # for item in x[1:]:
        #   new_x = new_x + item
        # x = new_x

        x = self.ops[blk_id][node_id](x)
        return x

    def toposort(self, graph):
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

    def forward(self, images, blks):
        x = images
        aux = []
        for blk_id, blk in enumerate(blks):
            # print("####build block {} ####".format(blk_id))
            graph, cell_list = blk[0], blk[1]
            graph = copy.deepcopy(graph)
            graph.append([])  # len(graph) = len(cell_list) + 1 at this time
            
            topo_order = self.toposort(graph)
            # print(topo_order)
            assert topo_order[0] == 0, "the first topo order id is not 0, the topo order is {}".format(topo_order)
            assert graph[topo_order[-1]] == [], "the last topo order node is not [], the graph is {}, topo order is {}".format(graph, topo_order)
            inputs = [[] for _ in range(len(graph))]
            inputs[0].append(x)
            for node_id in topo_order:
                if node_id == len(topo_order) - 1:
                    break  # break when the last pooling
                # print("constructing ", node_id, self.blks[blk_id][1][node_id])
                outputs = self.op_cutom(inputs=inputs, blk_id=blk_id, node_id=node_id)
                # append the output to where it should be put 
                # print("adding to ", graph[node_id])
                for out_id in graph[node_id]:
                    inputs[out_id].append(outputs)
            # last pooling
            x = inputs[topo_order[-1]]
            x = torch.cat(x, 1)  # note : it is NCHW

            x = self.reduce_pool(x)

            if blk_id != len(self.blks)-1:
                cur_aux = self.auxiliary_head[blk_id](x)
                cur_aux = cur_aux.view(cur_aux.size(0), -1)
                cur_aux = self.auxiliary_classifiers[blk_id](cur_aux)
                aux.append(cur_aux)

        x = self.global_pooling(x)
        x = self.classifier(x.view(x.size(0),-1))

        logits_aux = aux[0]
        for item in aux[1:]:
            logits_aux = logits_aux + item
        return x, logits_aux


def main():
    #  0.96 c10
#   blks = \
#   [
#   [[[1, 6, 9, 3], [2, 3, 4], [3, 4], [4, 10], [5], [10], [7], [8], [4], [5]], 
#   [('conv', 64, 1, 'relu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 64, 5, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 32, 3, 'relu'), ('conv', 48, 5, 'relu'), ('conv', 64, 3, 'relu'), ('conv', 32, 5, 'relu'), ('conv', 48, 3, 'leakyrelu')]],
#   [[[1, 2, 3], [2, 6], [3, 4], [4, 7, 5], [5], [8], [4], [5]], 
#   [('conv', 128, 5, 'leakyrelu'), ('conv', 48, 5, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 128, 3, 'leakyrelu'), ('conv', 48, 3, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('conv', 64, 1, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu')]],
#   [[[1, 6, 7, 2, 3], [2, 3, 4], [3, 4], [4, 5], [5], [9], [5], [8], [5]], 
#   [('conv', 64, 3, 'leakyrelu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 5, 'relu6'), ('conv', 64, 1, 'relu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 192, 1, 'relu'), ('conv', 192, 3, 'leakyrelu')]],
#   [[[1, 6, 2, 3], [2, 8, 3, 4], [3, 5], [4, 5], [5], [11], [7], [5], [9], [10], [5]], 
#   [('conv', 192, 1, 'relu'), ('conv', 128, 1, 'relu'), ('conv', 256, 5, 'relu6'), ('conv', 128, 5, 'relu6'), ('conv', 192, 3, 'leakyrelu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 128, 3, 'relu6'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 256, 1, 'leakyrelu'), ('conv', 192, 3, 'relu6')]],
#   ]
# 0.96 c10 double channel
#   blks = \
#   [
#   [[[1, 6, 9, 3], [2, 3, 4], [3, 4], [4, 10], [5], [10], [7], [8], [4], [5]], 
#   [('conv', 128, 1, 'relu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 128, 5, 'relu'), ('conv', 96, 1, 'leakyrelu'), ('conv', 128, 3, 'relu6'), ('conv', 64, 3, 'relu'), ('conv', 96, 5, 'relu'), ('conv', 128, 3, 'relu'), ('conv', 64, 5, 'relu'), ('conv', 96, 3, 'leakyrelu')]],
#   [[[1, 2, 3], [2, 6], [3, 4], [4, 7, 5], [5], [8], [4], [5]], 
#   [('conv', 256, 5, 'leakyrelu'), ('conv', 96, 5, 'leakyrelu'), ('conv', 128, 3, 'relu6'), ('conv', 256, 3, 'leakyrelu'), ('conv', 96, 3, 'relu'), ('conv', 96, 1, 'leakyrelu'), ('conv', 128, 1, 'leakyrelu'), ('conv', 256, 5, 'leakyrelu')]],
#   [[[1, 6, 7, 2, 3], [2, 3, 4], [3, 4], [4, 5], [5], [9], [5], [8], [5]], 
#   [('conv', 128, 3, 'leakyrelu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 256, 5, 'leakyrelu'), ('conv', 256, 1, 'leakyrelu'), ('conv', 384, 5, 'relu6'), ('conv', 128, 1, 'relu'), ('conv', 384, 1, 'leakyrelu'), ('conv', 384, 1, 'relu'), ('conv', 384, 3, 'leakyrelu')]],
#   [[[1, 6, 2, 3], [2, 8, 3, 4], [3, 5], [4, 5], [5], [11], [7], [5], [9], [10], [5]], 
#   [('conv', 384, 1, 'relu'), ('conv', 256, 1, 'relu'), ('conv', 512, 5, 'relu6'), ('conv', 256, 5, 'relu6'), ('conv', 384, 3, 'leakyrelu'), ('conv', 256, 3, 'leakyrelu'), ('conv', 256, 3, 'relu6'), ('conv', 256, 1, 'leakyrelu'), ('conv', 384, 1, 'leakyrelu'), ('conv', 512, 1, 'leakyrelu'), ('conv', 384, 3, 'relu6')]],
#   ]
#  0.96 c10  repeat the last blk
#   blks = \
#   [
#   [[[1, 6, 9, 3], [2, 3, 4], [3, 4], [4, 10], [5], [10], [7], [8], [4], [5]], 
#   [('conv', 64, 1, 'relu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 64, 5, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 32, 3, 'relu'), ('conv', 48, 5, 'relu'), ('conv', 64, 3, 'relu'), ('conv', 32, 5, 'relu'), ('conv', 48, 3, 'leakyrelu')]],
#   [[[1, 2, 3], [2, 6], [3, 4], [4, 7, 5], [5], [8], [4], [5]], 
#   [('conv', 128, 5, 'leakyrelu'), ('conv', 48, 5, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 128, 3, 'leakyrelu'), ('conv', 48, 3, 'relu'), ('conv', 48, 1, 'leakyrelu'), ('conv', 64, 1, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu')]],
#   [[[1, 6, 7, 2, 3], [2, 3, 4], [3, 4], [4, 5], [5], [9], [5], [8], [5]], 
#   [('conv', 64, 3, 'leakyrelu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 5, 'relu6'), ('conv', 64, 1, 'relu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 192, 1, 'relu'), ('conv', 192, 3, 'leakyrelu')]],
#   [[[1, 6, 2, 3], [2, 8, 3, 4], [3, 5], [4, 5], [5], [11], [7], [5], [9], [10], [5]], 
#   [('conv', 192, 1, 'relu'), ('conv', 128, 1, 'relu'), ('conv', 256, 5, 'relu6'), ('conv', 128, 5, 'relu6'), ('conv', 192, 3, 'leakyrelu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 128, 3, 'relu6'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 256, 1, 'leakyrelu'), ('conv', 192, 3, 'relu6')]],
#   [[[1, 6, 2, 3], [2, 8, 3, 4], [3, 5], [4, 5], [5], [11], [7], [5], [9], [10], [5]], 
#   [('conv', 192, 1, 'relu'), ('conv', 128, 1, 'relu'), ('conv', 256, 5, 'relu6'), ('conv', 128, 5, 'relu6'), ('conv', 192, 3, 'leakyrelu'), ('conv', 128, 3, 'leakyrelu'), ('conv', 128, 3, 'relu6'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 1, 'leakyrelu'), ('conv', 256, 1, 'leakyrelu'), ('conv', 192, 3, 'relu6')]],
#   ]
#    vgg encode
#   blks = \
#   [
#     [[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17]], 
#     [('conv', 64, 3, 'relu'), ('conv', 64, 3, 'relu'), ('pooling', 'max', 2), ('conv', 128, 3, 'relu'),
#     ('conv', 128, 3, 'relu'), ('pooling', 'max', 2), ('conv', 256, 3, 'relu'),
#     ('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('pooling', 'max', 2),
#     ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
#     ('pooling', 'max', 2), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
#     ('conv', 512, 3, 'relu')]]
#     ]

#   blks = \
#   [
#     [[[1]], 
#     [('conv', 64, 3, 'relu')]]
#     ]
#  0.95 with sep_conv
#   blks = [
#       [
#         [[1, 4, 5, 6, 8, 3, 7, 10], [2, 10], [3, 10], [10], [3, 10], [3, 10], [7, 10], [3, 10], [9, 3], [3, 10]],
#         [('conv', 32, 3, 'leakyrelu'), ('conv', 32, 5, 'relu'), ('sep_conv', 32, 1, 'leakyrelu'), ('sep_conv', 16, 5, 'relu'), ('conv', 32, 1, 'leakyrelu'), ('conv', 32, 3, 'leakyrelu'), ('conv', 32, 5, 'relu6'), ('conv', 16, 3, 'relu6'), ('sep_conv', 32, 3, 'relu6'), ('conv', 32, 5, 'leakyrelu')]
#       ],
#       [
#         [[1, 4, 6, 7, 5], [2, 6, 7, 8, 9], [3, 9], [9], [5, 9], [3, 9], [3, 9], [3, 9], [3, 9]],
#         [('conv', 64, 1, 'relu'), ('conv', 48, 5, 'relu6'), ('sep_conv', 32, 3, 'relu'), ('sep_conv', 48, 5, 'relu6'), ('conv', 32, 5, 'relu'), ('sep_conv', 64, 5, 'relu'), ('conv', 32, 1, 'leakyrelu'), ('conv', 48, 5, 'relu6'), ('conv', 64, 3, 'leakyrelu')]
#       ],
#       [
#         [[1, 4, 6, 8, 10, 2, 11, 3], [2, 3, 12], [3, 12], [12], [5, 12], [3, 12], [7, 3], [3], [9, 3, 12], [3, 12], [11, 3], [3]],
#         [('conv', 128, 1, 'relu'), ('conv', 128, 3, 'leakyrelu'), ('sep_conv', 64, 3, 'relu6'), ('sep_conv', 128, 5, 'leakyrelu'), ('conv', 96, 5, 'relu6'), ('sep_conv', 128, 5, 'leakyrelu'), ('conv', 128, 1, 'leakyrelu'), ('conv', 128, 5, 'relu6'), ('sep_conv', 64, 3, 'relu'), ('sep_conv', 128, 1, 'leakyrelu'), ('conv', 96, 5, 'relu'), ('conv', 128, 1, 'relu')]
#       ],
#       [
#         [[1, 4, 5, 7, 6, 8, 9], [2, 3, 9], [3, 9], [9], [3], [6, 3], [3, 9], [8, 3, 9], [3, 9]],
#         [('sep_conv', 128, 3, 'relu6'), ('sep_conv', 256, 1, 'relu6'), ('conv', 256, 3, 'leakyrelu'), ('sep_conv', 192, 1, 'relu'), ('conv', 192, 1, 'relu6'), ('conv', 256, 3, 'leakyrelu'), ('conv', 192, 1, 'leakyrelu'), ('sep_conv', 128, 3, 'relu'), ('conv', 128, 5, 'relu6')]
#       ]
#   ]
#   0.954  ==>  0.9656, try it in torch , before double channel 0.9632
#   blks = \
#   [
#       [
#         [[1, 4, 5, 2, 3, 6], [2, 3], [3], [7], [3], [6, 3], [3]],
#         [('conv', 64, 3, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 64, 3, 'leakyrelu'),
#         ('conv', 32, 3, 'relu'), ('conv', 32, 1, 'relu'), ('conv', 64, 3, 'relu')]
#       ],
#       [
#         [[1, 4, 5, 2, 6, 3], [2, 3], [3], [7], [2, 3], [6, 3], [3]],
#         [('conv', 128, 3, 'relu'), ('conv', 128, 3, 'relu'), ('conv', 192, 3, 'relu'), ('conv', 128, 3, 'leakyrelu'),
#         ('conv', 192, 3, 'relu'), ('conv', 128, 3, 'relu'), ('conv', 192, 3, 'relu')]
#       ],
#       [
#         [[1, 4, 3], [2], [3], [5], [3]],
#         [('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 192, 3, 'relu'),
#         ('conv', 192, 3, 'leakyrelu'), ('conv', 256, 3, 'relu')]
#       ],
#       [
#         [[1, 4, 3], [2], [3], [5], [3]],
#         [('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 192, 3, 'relu'),
#         ('conv', 192, 3, 'leakyrelu'), ('conv', 256, 3, 'relu')]
#       ]
#   ]
#   0.954  ==>  0.9656, try it in torch , after double channel
#   blks = \
#   [
#       [
#         [[1, 4, 5, 2, 3, 6], [2, 3], [3], [7], [3], [6, 3], [3]],
#         [('conv', 128, 3, 'relu'), ('conv', 96, 3, 'relu'), ('conv', 96, 3, 'relu'), ('conv', 128, 3, 'leakyrelu'),
#         ('conv', 64, 3, 'relu'), ('conv', 64, 1, 'relu'), ('conv', 128, 3, 'relu')]
#       ],
#       [
#         [[1, 4, 5, 2, 6, 3], [2, 3], [3], [7], [2, 3], [6, 3], [3]],
#         [('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 384, 3, 'relu'), ('conv', 256, 3, 'leakyrelu'),
#         ('conv', 384, 3, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 384, 3, 'relu')]
#       ],
#       [
#         [[1, 4, 3], [2], [3], [5], [3]],
#         [('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 384, 3, 'relu'),
#         ('conv', 384, 3, 'leakyrelu'), ('conv', 512, 3, 'relu')]
#       ],
#       [
#         [[1, 4, 3], [2], [3], [5], [3]],
#         [('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 384, 3, 'relu'),
#         ('conv', 384, 3, 'leakyrelu'), ('conv', 512, 3, 'relu')]
#       ]
#   ]
#  0.9548, c10
  blks =\
    [
    [[[1, 6, 7, 2, 3], [2, 4], [3, 4, 5], [4, 5, 10], [5, 10], [10], [4], [8], [9], [5]], 
    [('conv', 64, 1, 'relu'), ('conv', 48, 1, 'relu'), ('conv', 32, 5, 'relu'), ('conv', 64, 1, 'leakyrelu'), ('conv', 32, 5, 'leakyrelu'), ('conv', 32, 3, 'relu'), ('conv', 64, 5, 'relu'), ('conv', 32, 1, 'leakyrelu'), ('conv', 32, 1, 'leakyrelu'), ('conv', 32, 5, 'relu')]],
    [[[1, 6, 9, 3], [2, 3, 4], [3, 13, 4, 5], [4], [5], [14], [7], [8], [4], [10], [11], [12], [5], [5]], 
    [('conv', 48, 3, 'relu'), ('conv', 64, 3, 'leakyrelu'), ('conv', 64, 5, 'relu6'), ('conv', 48, 5, 'relu'), ('conv', 48, 1, 'relu'), ('conv', 48, 5, 'leakyrelu'), ('conv', 128, 5, 'relu6'), ('conv', 64, 3, 'relu6'), ('conv', 48, 1, 'relu6'), ('conv', 128, 1, 'relu'), ('conv', 64, 1, 'relu'), ('conv', 48, 1, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 48, 3, 'leakyrelu')]],
    [[[1, 3], [2, 6, 9, 4], [3, 12, 4, 5], [4, 5], [5, 13], [13], [7], [8], [5], [10], [11], [5], [4]], 
    [('conv', 128, 1, 'relu6'), ('conv', 192, 1, 'relu6'), ('conv', 64, 1, 'relu6'), ('conv', 192, 3, 'leakyrelu'), ('conv', 64, 3, 'relu6'), ('conv', 64, 5, 'relu6'), ('conv', 128, 5, 'relu6'), ('conv', 128, 3, 'relu6'), ('conv', 64, 5, 'leakyrelu'), ('conv', 128, 5, 'leakyrelu'), ('conv', 64, 5, 'relu6'), ('conv', 128, 1, 'leakyrelu'), ('conv', 192, 3, 'leakyrelu')]],
    [[[1, 6, 3], [2, 10, 12, 3, 4], [3, 4, 5], [4, 5, 15], [5], [15], [7], [8], [9], [5], [11], [4], [13], [14], [5]], 
    [('conv', 256, 1, 'relu'), ('conv', 256, 1, 'leakyrelu'), ('conv', 128, 1, 'relu'), ('conv', 128, 5, 'relu'), ('conv', 256, 3, 'relu'), ('conv', 128, 5, 'leakyrelu'), ('conv', 192, 5, 'relu'), ('conv', 256, 3, 'leakyrelu'), ('conv', 256, 3, 'relu6'), ('conv', 256, 5, 'relu'), ('conv', 256, 1, 'relu'), ('conv', 192, 5, 'relu6'), ('conv', 128, 5, 'relu'), ('conv', 192, 5, 'relu6'), ('conv', 256, 1, 'relu')]],
    ]


  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  model = Network(CIFAR_CLASSES, blks)
#   model = vgg(CIFAR_CLASSES)

  model = model.cuda()

#   print(len(list(model.parameters())))

  logging.info("param size = %fMB", cnn.utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = cnn.utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
#   train_transform, valid_transform = cnn.utils._data_transforms_cifar100(args)
#   train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
#   valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)   # read test data 

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, blks, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, blks, criterion)
    logging.info('valid_acc %f', valid_acc)

    cnn.utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, blks, criterion, optimizer):
  objs = cnn.utils.AvgrageMeter()
  top1 = cnn.utils.AvgrageMeter()
  top5 = cnn.utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input, blks)
    # logits = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = cnn.utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, blks, criterion):
  objs = cnn.utils.AvgrageMeter()
  top1 = cnn.utils.AvgrageMeter()
  top5 = cnn.utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input, blks)
    # logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = cnn.utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

