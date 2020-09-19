import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkDenoise as Network
from denoise_data import DenoiseDataSetPrepare, BasicDataset


parser = argparse.ArgumentParser("denoise")
parser.add_argument('--data', type=str, default='../data/denoise/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=100, help='batch size') # 16
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs') # 250
parser.add_argument('--init_channels', type=int, default=4, help='num of init channels') # 32
parser.add_argument('--layers', type=int, default=2, help='total number of layers') # 7
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  DenoiseDataSetPrepare(args.data, args.batch_size)   # make sure the denoise data is prepared

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.layers, args.auxiliary, genotype)
  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = torch.nn.MSELoss(reduce=True, size_average=True)
  criterion = criterion.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'test')

  trian_noisy_dir = os.path.join(traindir, "noisy")
  trian_clear_dir = os.path.join(traindir, "original")
  train_dataset = BasicDataset(traindir, trian_noisy_dir, trian_clear_dir)
  val_ratio = 0.1
  n_val = int(len(train_dataset) * val_ratio)
  n_train = len(train_dataset) - n_val
  train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])
  # test_noisy_dir = os.path.join(validdir, "noisy")
  # test_clear_dir = os.path.join(validdir, "original")
  # test_dataset = BasicDataset(validdir, test_noisy_dir, test_clear_dir)

  train_queue = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  best_acc_top1 = 0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc_top1, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, item in enumerate(train_queue):
    input = item['image']
    target = item['label']
    target = target.cuda(async=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    psnr = utils.psnr(logits, target)
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(psnr.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, item in enumerate(valid_queue):
    input = item['image']
    target = item['label']
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1 = utils.psnr(logits, target)
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
