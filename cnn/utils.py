import os
import math
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import utils as vutils


# for eva denoise pred huge img, we can cut it into many small pieces, and put them into model in a batch
# one ins for one image
class image_splitor:
    def __init__(self, pieces_len):
        self.pieces_len = pieces_len  # pieces is square
        self.img_h = -1
        self.img_w = -1

    def _split_huge_img(self, data):
        # np data shape is [batch_size, h, w, c], batch_size = 1
        print("spliting the img {}".format(data.shape), flush=True)
        self.img_h = data.shape[1]
        self.img_w = data.shape[2]
        for i in range(0, self.img_h-1, self.pieces_len):
            for j in range(0, self.img_w-1, self.pieces_len):
              print(i, j)
              end_h = i+self.pieces_len if i+self.pieces_len <= self.img_h else self.img_h
              end_w = j+self.pieces_len if j+self.pieces_len <= self.img_w else self.img_w
              # print(data[:,i:end_h,j:end_w,:].shape)
              data_for_append = np.pad(data[:,i:end_h,j:end_w,:],((0,0),(0,i+100-end_h),(0,j+100-end_w),(0,0)),'constant')
              if i == 0 and j == 0:
                  new_data = data_for_append
              else:
                  new_data = np.concatenate([new_data, data_for_append], axis=0)
        print("into {}".format(new_data.shape), flush=True)
        return new_data

    def _recover_huge_img(self, data):
        # np data shape is [batch_size, h, w, c], batch_size represent many pieces
        print("concating the img {}".format(data.shape), flush=True)
        for i in range(0, self.img_h-1, self.pieces_len):
            for j in range(0, self.img_w-1, self.pieces_len):
                end_h = self.pieces_len if i+self.pieces_len <= self.img_h else self.img_h%self.pieces_len
                end_w = self.pieces_len if j+self.pieces_len <= self.img_w else self.img_w%self.pieces_len
                idx = (i//self.pieces_len)*math.ceil(self.img_w/self.pieces_len)+(j//self.pieces_len)
                # print(data[idx:idx+1,:end_h,:end_w,:].shape)
                if j == 0:
                    new_data_y = data[idx:idx+1,:end_h,:end_w,:]
                else:
                    new_data_y = np.concatenate([new_data_y, data[idx:idx+1,:end_h,:end_w,:]], axis=2)
            if i == 0:
                new_data_x = new_data_y
            else:
                new_data_x = np.concatenate([new_data_x, new_data_y], axis=1)
        print("into {}".format(new_data_x.shape), flush=True)
        return new_data_x


def save_image_tensor(input_tensor, filename):
  assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
  # 复制一份
  input_tensor = input_tensor.clone().detach()
  # 到cpu
  input_tensor = input_tensor.to(torch.device('cpu'))
  # 反归一化
  # input_tensor = unnormalize(input_tensor)
  vutils.save_image(input_tensor, filename)


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def psnr(output, target):
  mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)
  mse_fn.cuda()
  mse = mse_fn(output.float(), target.float())
  psnr = 10.0 * (torch.log(torch.tensor([255.0 ** 2 / mse])) / torch.log(torch.tensor([10.0])))
  return psnr


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load_adaptive(model, model_path):
  model_dict = model.state_dict()
  import_model = torch.load(model_path)
  pretrained_dict = {k: v for k, v in import_model.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

