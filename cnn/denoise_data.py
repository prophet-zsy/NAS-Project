import os, sys, cv2, random, time, copy
import numpy as np
from glob import glob
# from skimage.measure import compare_ssim

# config params for eva alone
INSTANT_PRINT = False  # set True only in run the eva alone
DATA_AUG_TIMES = 3
DATA_SIZE_FOR_SEARCH = 50000  # max: 355120
DATA_RATIO_FOR_EVAL = 0.1


class DenoiseDataSetPrepare:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.all_train_data, self.all_train_label, self.train_data, self.train_label,\
             self.valid_data, self.valid_label, self.test_data, self.test_label = self.inputs()

        if os.path.exists(os.path.join(self.data_path, 'custom_test')):
            self.custom_test_data, self.custom_test_label = self.custom_inputs()
        else:
            self.custom_test_data, self.custom_test_label = [], []

    def get_train_data(self, data_size):
        return self.train_data[:data_size], self.train_label[:data_size]

    def custom_inputs(self):
        custom_test_data = []
        custom_test_label = []

        custom_path = os.path.join(self.data_path, 'custom_test')
        noisy_imgs = [os.path.join(custom_path, 'noisy', item) for item in os.listdir(os.path.join(custom_path, 'noisy'))]
        original_imgs = [os.path.join(custom_path, 'original', item) for item in os.listdir(os.path.join(custom_path, 'original'))]
        if noisy_imgs and original_imgs:  # original img and noisy img
            noisy_imgs = sorted(noisy_imgs)
            original_imgs = sorted(original_imgs)
            custom_test_data = [cv2.imread(img) for img in noisy_imgs]
            custom_test_label = [cv2.imread(img) for img in original_imgs]

        elif original_imgs and not noisy_imgs:  # only original imgs 
            for original_img_path in original_imgs:
                img_file = os.path.basename(original_img_path).split('.')[0]
                original_img = cv2.imread(original_img_path)
                sigma = np.random.randint(0, 50)
                noisy_img = self.gaussian_noise(sigma, original_img)
                cv2.imwrite(os.path.join(custom_path, 'noisy') + img_file + ".png", noisy_img)
                custom_test_data.append(noisy_img)
                custom_test_label.append(original_img)
                
        elif noisy_imgs and not original_imgs:  # only noisy imgs, we create fake original imgs for the model input
            for noisy_img_path in noisy_imgs:
                img_file = os.path.basename(noisy_img_path).split('.')[0]
                noisy_img = cv2.imread(noisy_img_path)
                img_size = noisy_img.shape
                original_img = np.random.randint(0, 255, size=img_size)
                cv2.imwrite(os.path.join(custom_path, 'original') + img_file + ".png", original_img)
                custom_test_data.append(noisy_img)
                custom_test_label.append(original_img)
        return self._process_test_data(custom_test_data, custom_test_label)

    def add_noise(self):
        imgs_path = glob(self.data_path + "pristine_images/*.bmp")
        num_of_samples = len(imgs_path)
        imgs_path_train = imgs_path[:int(num_of_samples * 0.7)]
        imgs_path_test = imgs_path[int(num_of_samples * 0.7):]

        sigma_train = np.linspace(0, 50, int(num_of_samples * 0.7) + 1)
        print("[*] Creating original-noisy train set...")
        for i in range(int(num_of_samples * 0.7)):
            if INSTANT_PRINT and i % 50 == 0:
                print("{}/{}".format(i, int(num_of_samples * 0.7)))
            img_path = imgs_path_train[i]
            img_file = os.path.basename(img_path).split('.bmp')[0]
            sigma = sigma_train[i]
            img_original = cv2.imread(img_path)
            img_noisy = self.gaussian_noise(sigma, img_original)

            cv2.imwrite(self.data_path + "train/noisy/" + img_file + ".png", img_noisy)
            cv2.imwrite(self.data_path + "train/original/" + img_file + ".png", img_original)
        print("[*] Creating original-noisy test set...")
        for i in range(int(num_of_samples * 0.3)):
            if INSTANT_PRINT and i % 50 == 0:
                print("{}/{}".format(i, int(num_of_samples * 0.3)))
            img_path = imgs_path_test[i]
            img_file = os.path.basename(img_path).split('.bmp')[0]
            sigma = np.random.randint(0, 50)

            img_original = cv2.imread(img_path)
            img_noisy = self.gaussian_noise(sigma, img_original)

            cv2.imwrite(self.data_path + "test/noisy/" + img_file + ".png", img_noisy)
            cv2.imwrite(self.data_path + "test/original/" + img_file + ".png", img_original)

    def gaussian_noise(self, sigma, image):
        gaussian = np.random.normal(0, sigma, image.shape)
        noisy_image = image + gaussian
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    def normalize(self, data, is_test=False):
        if is_test:  # when in test, we normalize them one by one 
            norm_data = [item.astype(np.float32) / 255.0 for item in data]
        else:
            norm_data = data.astype(np.float32) / 255.0
        return norm_data

    def _expend_test_dim(self, data):
        new_data = [item[np.newaxis, :] for item in data]
        return new_data

    def _process_data(self, all_train_data, all_train_label, test_data, test_label):
        all_train_data = self.normalize(all_train_data)
        all_train_label = self.normalize(all_train_label)
        train_data, train_label, valid_data, valid_label =\
            self._shuffle_and_split_valid(all_train_data, all_train_label)
        test_data = self._expend_test_dim(test_data)
        test_label = self._expend_test_dim(test_label)
        test_data = self.normalize(test_data, is_test=True)
        test_label = self.normalize(test_label, is_test=True)
        return all_train_data, all_train_label, train_data, train_label, valid_data, valid_label, test_data, test_label

    def _process_test_data(self, test_data, test_label):
        test_data = self._expend_test_dim(test_data)
        test_label = self._expend_test_dim(test_label)
        test_data = self.normalize(test_data, is_test=True)
        test_label = self.normalize(test_label, is_test=True)
        return test_data, test_label

    def _read_traindata_frompath_save2npy(self, pat_size=50, stride=100):
        global DATA_AUG_TIMES
        count = 0
        filepaths = glob(
            self.data_path + "train/original/" + '/*.png')  # takes all the paths of the png files in the train folder
        filepaths.sort(key=lambda x: int(os.path.basename(x)[:-4]))  # order the file list
        filepaths_noisy = glob(self.data_path + "train/noisy/" + '/*.png')
        filepaths_noisy.sort(key=lambda x: int(os.path.basename(x)[:-4]))
        print("[*] Number of training samples: %d" % len(filepaths))
        scales = [1, 0.8]

        # calculate the number of patches
        for i in range(len(filepaths)):
            img = cv2.imread(filepaths[i])
            for s in range(len(scales)):
                newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
                img_s = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)
                im_h = img_s.shape[0]
                im_w = img_s.shape[1]
                for x in range(0, (im_h - pat_size), stride):
                    for y in range(0, (im_w - pat_size), stride):
                        count += 1

        origin_patch_num = count * DATA_AUG_TIMES

        if origin_patch_num % self.batch_size != 0:
            numPatches = (origin_patch_num // self.batch_size + 1) * self.batch_size  # round
        else:
            numPatches = origin_patch_num
        print("[*] Number of patches = %d, batch size = %d, total batches = %d" % \
            (numPatches, self.batch_size, numPatches / self.batch_size))

        # data matrix 4-D
        train_label = np.zeros((numPatches, pat_size, pat_size, 3), dtype="uint8")  # clean patches
        train_data = np.zeros((numPatches, pat_size, pat_size, 3), dtype="uint8")  # noisy patches

        count = 0
        # generate patches
        for i in range(len(filepaths)):
            img = cv2.imread(filepaths[i])
            img_noisy = cv2.imread(filepaths_noisy[i])
            for s in range(len(scales)):
                newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
                img_s = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)
                img_s_noisy = cv2.resize(img_noisy, newsize, interpolation=cv2.INTER_CUBIC)
                img_s = np.reshape(np.array(img_s, dtype="uint8"),
                                (img_s.shape[0], img_s.shape[1], 3))  # extend one dimension
                img_s_noisy = np.reshape(np.array(img_s_noisy, dtype="uint8"),
                                        (img_s_noisy.shape[0], img_s_noisy.shape[1], 3))  # extend one dimension

                for j in range(DATA_AUG_TIMES):
                    im_h = img_s.shape[0]
                    im_w = img_s.shape[1]
                    for x in range(0, im_h - pat_size, stride):
                        for y in range(0, im_w - pat_size, stride):
                            a = random.randint(0, 7)
                            train_label[count, :, :, :] = self.process(
                                img_s[x:x + pat_size, y:y + pat_size, :], a)
                            train_data[count, :, :, :] = self.process(
                                img_s_noisy[x:x + pat_size, y:y + pat_size, :], a)
                            count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            train_label[-to_pad:, :, :, :] = train_label[:to_pad, :, :, :]
            train_data[-to_pad:, :, :, :] = train_data[:to_pad, :, :, :]
        
        train_data = train_data[:DATA_SIZE_FOR_SEARCH]
        train_label = train_label[:DATA_SIZE_FOR_SEARCH]
        np.save(self.data_path + "train/img_noisy_pats.npy", train_data)
        np.save(self.data_path + "train/img_clean_pats.npy", train_label)

        all_train_data, all_train_label = train_data, train_label

        return all_train_data, all_train_label

    def inputs(self):
        if not os.path.exists(self.data_path + "train/noisy/") or not os.listdir(self.data_path + "train/noisy/"):
            self.add_noise()
        noisy_eval_files = glob(self.data_path + 'test/noisy/*.png')
        noisy_eval_files = sorted(noisy_eval_files)
        test_data = [cv2.imread(img) for img in noisy_eval_files]

        eval_files = glob(self.data_path + 'test/original/*.png')
        eval_files = sorted(eval_files)
        test_label = [cv2.imread(img) for img in eval_files]
        if os.path.exists(self.data_path + "train/img_noisy_pats.npy"):  # read train data from npy directly
            all_train_data = np.load(self.data_path + "train/img_noisy_pats.npy")
            all_train_label = np.load(self.data_path + "train/img_clean_pats.npy")
            if all_train_data.shape[0] < DATA_SIZE_FOR_SEARCH:  # size of cur npy is not enough
                all_train_data, all_train_label = self._read_traindata_frompath_save2npy()
                # print(all_train_data.shape)
            else:
                all_train_data, all_train_label = all_train_data[:DATA_SIZE_FOR_SEARCH], all_train_label[:DATA_SIZE_FOR_SEARCH]
            return self._process_data(all_train_data, all_train_label, test_data, test_label)
        else:  # read train data and save it to npy
            all_train_data, all_train_label = self._read_traindata_frompath_save2npy()
            return self._process_data(all_train_data, all_train_label, test_data, test_label)

    def process(self, image, mode):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down
            return np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)

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


from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, data_dir, denoise_dir, label_dir):
        self.data_dir = data_dir
        self.denoise_dir = denoise_dir
        self.label_dir = label_dir
        if os.path.exists(self.data_dir + "img_noisy_pats.npy"):  # read train data from npy directly
            self.data = np.load(self.data_dir + "img_noisy_pats.npy")
            self.label = np.load(self.data_dir + "img_clean_pats.npy")

            self.ids = [i for i in range(len(self.data))]
        else:
            self.ids = [splitext(file)[0] for file in listdir(denoise_dir)
                        if not file.startswith('.')]
        
        print("Creating dataset with {} examples".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        if os.path.exists(self.data_dir + "img_noisy_pats.npy"):  # read train data from npy directly
            idx = self.ids[i]
            img = self.data[idx]
            label = self.label[idx]
        else:
            idx = self.ids[i]
            label_file = glob(self.label_dir + idx + '.*')
            img_file = glob(self.denoise_dir + idx + '.*')

            assert len(label_file) == 1, \
                "Either no label or multiple labels found for the ID {}: {}".format(idx, label_file)
            assert len(img_file) == 1, \
                "Either no image or multiple images found for the ID {}: {}".format(idx, img_file)
            label = Image.open(label_file[0])
            img = Image.open(img_file[0])

            assert img.size == label.size, \
                "Image and label {} should be the same size, but are {} and {}".format(idx, img.size, label.size)

            img = self.preprocess(img)
            label = self.preprocess(label)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor)
        }


if __name__ == "__main__":
    data_path = "../data/denoise/"
    batch_size = 50
    tem = DenoiseDataSetPrepare(data_path, batch_size)
    for i in range(10):
        image = tem.train_data[i] * 255.0
        cv2.imwrite("./{}image_o.png".format(i), image)
        image = tem.train_label[i] * 255.0
        cv2.imwrite("./{}image_l.png".format(i), image)

