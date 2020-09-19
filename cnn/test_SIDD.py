import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkDenoise as Network
from denoise_data import DenoiseDataSetPrepare, BasicDataset
from utils import save_image_tensor, image_splitor


parser = argparse.ArgumentParser("denoise")
parser.add_argument('--data', type=str, default='../data/SIDD_Small_sRGB_Only/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=4, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='./eval-EXP-20200919-124157/model_best.pth.tar', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


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
  model = model.cuda()
  utils.load_adaptive(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = torch.nn.MSELoss(reduce=True, size_average=True)
  criterion = criterion.cuda()


  testdir = os.path.join(args.data, 'test')

  test_noisy_dir = os.path.join(testdir, "noisy")
  test_clear_dir = os.path.join(testdir, "original")
  test_dataset = BasicDataset(testdir, test_noisy_dir, test_clear_dir)
  
  test_queue = torch.utils.data.DataLoader(
      test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
  
  
  model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj = infer(test_queue, model, criterion)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  model.eval()

  img_split = image_splitor(100)

  for step, item in enumerate(test_queue):
    idx = item['idx']
    input = item['image']
    target = item['label']

    # cut the img
    input = input.permute(0,2,3,1)  # from NCHW to NHWC in torch
    input = input.numpy()
    input = img_split._split_huge_img(input)
    input = torch.from_numpy(input)
    input = input.permute(0,3,1,2)

    mini_batch_size = 10
    res = []
    for i in range(0, input.shape[0], mini_batch_size):
      tem_input = input[i: i + mini_batch_size]
      tem_input = Variable(tem_input, volatile=True).cuda()

      logits, _ = model(tem_input)
      res.append(logits)
    input = torch.cat(res, 0)

    # concat the img
    input = input.permute(0,2,3,1)  # from NCHW to NHWC in torch
    input = input.numpy()
    input = img_split._recover_huge_img(input)
    input = torch.from_numpy(input)
    input = input.permute(0,3,1,2)

    save_pred_dir = os.path.join(args.data, 'test', 'pred')
    if not os.path.exists(save_pred_dir):
      os.mkdir(save_pred_dir)
    file_name = os.path.join(save_pred_dir, idx[0]+'.png')
    save_image_tensor(logits, file_name)

    target = Variable(target, volatile=True).cuda(async=True)    
    loss = criterion(logits, target)

    psnr = utils.psnr(logits, target)
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(psnr.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

