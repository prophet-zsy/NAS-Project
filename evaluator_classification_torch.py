import os, sys, time, glob, copy
import numpy as np
import torch
import torch_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG, Logger, EvaScheduleItem
from tiny_imagenet_input import get_data_mode

torch.cuda.current_device()

INSTANT_PRINT = False

EVA_COFIG = NAS_CONFIG['eva']
TASK_NAME = EVA_COFIG['task_name']
if TASK_NAME == "cifar-10":
    NUM_CLASSES = 10
    IMAGE_SIZE = 32
    # IMAGE_NUMS = 50000
elif TASK_NAME == "cifar-100":
    NUM_CLASSES = 100
    IMAGE_SIZE = 32
    # IMAGE_NUMS = 50000
elif TASK_NAME == "tiny-imagenet":
    NUM_CLASSES = 200
    IMAGE_SIZE = 64
    # IMAGE_NUMS = 100000
else:
    raise Exception("Wrong task_name")


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
torch_utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p', filename='memory/evaluator_log.txt')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = NUM_CLASSES


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

        x = self.global_pooling(x)
        x = self.classifier(x.view(x.size(0),-1))

        logits_aux = []
        return x, logits_aux


def train(train_queue, model, blks, criterion, optimizer, log):
    objs = torch_utils.AvgrageMeter()
    top1 = torch_utils.AvgrageMeter()
    top5 = torch_utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input, blks)

        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = torch_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            if INSTANT_PRINT:
                print('train %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))
            # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            log += 'train %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, blks, criterion, log):
    objs = torch_utils.AvgrageMeter()
    top1 = torch_utils.AvgrageMeter()
    top5 = torch_utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits, _ = model(input, blks)
        loss = criterion(logits, target)

        prec1, prec5 = torch_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            if INSTANT_PRINT:
                print('valid %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))
            # logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            log += 'valid %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


class Evaluator:
    def __init__(self):
        self.log = ''
        self.epoch = 0
        self.data_size = 0

    def _set_epoch(self, e):
        self.epoch = e
        return self.epoch

    def _set_data_size(self, num):
        self.data_size = num
        return self.data_size

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

    def evaluate(self, task_item):
        self._log_item_info(task_item)

        blks = []
        for blk in task_item.pre_block:
            blks.append([blk.graph, [tuple(i) for i in blk.cell_list]])
        if task_item.network_item:
            blks.append([task_item.network_item.graph, [tuple(i) for i in task_item.network_item.cell_list]])

        if not torch.cuda.is_available():
            # logging.info('no gpu device available')
            print('no gpu device available', flush=True)
            self.log += 'no gpu device available'
            sys.exit(1)

        np.random.seed(args.seed)
        # print("set device ... ", task_item.gpu_info)
        # torch.cuda.set_device(task_item.gpu_info)
        cudnn.benchmark = True
        torch.manual_seed(args.seed)
        cudnn.enabled=True
        torch.cuda.manual_seed(args.seed)
        # logging.info('gpu device = %d' % args.gpu)
        # logging.info("args = %s", args)
        self.log += 'gpu device = %d' % task_item.gpu_info
        self.log += "args = %s" % args

        model = Network(CIFAR_CLASSES, blks)

        model = model.cuda()

        #   print(len(list(model.parameters())))
        task_item.model_params = torch_utils.count_parameters_in_MB(model)
        # logging.info("param size = %fMB", task_item.model_params)
        self.log += "param size = %fMB" % task_item.model_params

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )

        if TASK_NAME == "cifar-10":
            train_transform, valid_transform = torch_utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
            # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        elif TASK_NAME == "cifar-100":
            train_transform, valid_transform = torch_utils._data_transforms_cifar100(args)
            train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
            # valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)   # read test data 

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(task_item.epoch))

        res_acc = 0

        for epoch in range(task_item.epoch):
            scheduler.step()
            if INSTANT_PRINT:
                print('epoch %d lr %e' % (epoch, scheduler.get_lr()[0]))
            # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            self.log += 'epoch %d lr %e' % (epoch, scheduler.get_lr()[0]) 
            model.drop_path_prob = args.drop_path_prob * epoch / task_item.epoch

            train_acc, train_obj = train(train_queue, model, blks, criterion, optimizer, self.log)
            if INSTANT_PRINT:
                print('train_acc %f' % train_acc)
            # logging.info('train_acc %f', train_acc)
            self.log += 'train_acc %f' % train_acc

            valid_acc, valid_obj = infer(valid_queue, model, blks, criterion, self.log)
            if INSTANT_PRINT:
                print('valid_acc %f' % valid_acc)
            # logging.info('valid_acc %f', valid_acc)
            self.log += 'valid_acc %f'% valid_acc

            if res_acc < valid_acc:
                res_acc = valid_acc
            # torch_utils.save(model, os.path.join(args.save, 'weights.pt'))

        NAS_LOG = Logger()
        NAS_LOG << ('eva_eva', self.log)
        torch.cuda.empty_cache()

        return res_acc/100


if __name__ == '__main__':
    INSTANT_PRINT = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(1)
    eval = Evaluator()
    cur_data_size = eval._set_data_size(-1)
    cur_epoch = eval._set_epoch(1000)

    graph_full = [[1, 6, 9, 3], [2, 3, 4], [3, 4], [4, 10], [5], [10], [7], [8], [4], [5]]
    cell_list = [Cell('conv', 64, 1, 'relu'), Cell('conv', 64, 3, 'leakyrelu'), Cell('conv', 64, 5, 'relu'), Cell('conv', 48, 1, 'leakyrelu'), Cell('conv', 64, 3, 'relu6'), Cell('conv', 32, 3, 'relu'), Cell('conv', 48, 5, 'relu'), Cell('conv', 64, 3, 'relu'), Cell('conv', 32, 5, 'relu'), Cell('conv', 48, 3, 'leakyrelu')]
    network1 = NetworkItem(0, graph_full, cell_list, "")
    graph_full = [[1, 2, 3], [2, 6], [3, 4], [4, 7, 5], [5], [8], [4], [5]]
    cell_list = [Cell('conv', 128, 5, 'leakyrelu'), Cell('conv', 48, 5, 'leakyrelu'), Cell('conv', 64, 3, 'relu6'), Cell('conv', 128, 3, 'leakyrelu'), Cell('conv', 48, 3, 'relu'), Cell('conv', 48, 1, 'leakyrelu'), Cell('conv', 64, 1, 'leakyrelu'), Cell('conv', 128, 5, 'leakyrelu')]
    network2 = NetworkItem(1, graph_full, cell_list, "")
    graph_full = [[1, 6, 7, 2, 3], [2, 3, 4], [3, 4], [4, 5], [5], [9], [5], [8], [5]] 
    cell_list = [Cell('conv', 64, 3, 'leakyrelu'), Cell('conv', 64, 3, 'leakyrelu'), Cell('conv', 128, 5, 'leakyrelu'), Cell('conv', 128, 1, 'leakyrelu'), Cell('conv', 192, 5, 'relu6'), Cell('conv', 64, 1, 'relu'), Cell('conv', 192, 1, 'leakyrelu'), Cell('conv', 192, 1, 'relu'), Cell('conv', 192, 3, 'leakyrelu')]
    network3 = NetworkItem(2, graph_full, cell_list, "")
    graph_full = [[1, 6, 2, 3], [2, 8, 3, 4], [3, 5], [4, 5], [5], [11], [7], [5], [9], [10], [5]]
    cell_list = [Cell('conv', 192, 1, 'relu'), Cell('conv', 128, 1, 'relu'), Cell('conv', 256, 5, 'relu6'), Cell('conv', 128, 5, 'relu6'), Cell('conv', 192, 3, 'leakyrelu'), Cell('conv', 128, 3, 'leakyrelu'), Cell('conv', 128, 3, 'relu6'), Cell('conv', 128, 1, 'leakyrelu'), Cell('conv', 192, 1, 'leakyrelu'), Cell('conv', 256, 1, 'leakyrelu'), Cell('conv', 192, 3, 'relu6')]
    network4 = NetworkItem(3, graph_full, cell_list, "")

    task_item = EvaScheduleItem(nn_id=0, alig_id=0, graph_template=[], item=None,\
         pre_blk=[network1, network2, network3, network4], ft_sign=True, bestNN=True, rd=0, nn_left=0, spl_batch_num=6, epoch=cur_epoch, data_size=cur_data_size)
    e = eval.evaluate(task_item)
    print(e)


