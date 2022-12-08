import torch
import torch.utils.data as data
import cv2
import numpy as np
import torchvision.transforms as transforms
import random

from PIL import ImageFilter
from PIL import Image

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None, aug_strong=False):
        self.preproc = preproc
        self.strong = aug_strong
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                # print('label: ', label)
                labels.append(label)

        self.words.append(labels)

        ####
        augmentation = []
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        '''# randomcrop
        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)'''

        self.transform = transforms.Compose(augmentation)


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img_weak, target = self.preproc(img, target)

        '''if self.strong:
            img_weak = np.asarray(img_weak).transpose(1, 2, 0)
            img_weak = img_weak.astype(np.float32)
            img_weak += (104, 117, 123)

            img_weak_PIL = Image.fromarray(cv2.cvtColor(img_weak.astype('uint8'), cv2.COLOR_BGR2RGB), "RGB")

            image_strong_aug = np.array(self.transform(img_weak_PIL))
            name = "./show/" + str(index) + "image_strong_aug.jpg"
            cv2.imwrite(name, image_strong_aug)
            # print('strong: ', index)
            image_strong_aug = image_strong_aug.astype(np.float32)
            image_strong_aug -= (104, 117, 123)

            img_strong = image_strong_aug.transpose(2, 0, 1)

            data_strong = torch.from_numpy(img_strong).float(), target
            return data_strong

        else:
            data_weak = torch.from_numpy(img_weak).float(), target

            img_weak = np.asarray(img_weak).transpose(1, 2, 0)
            img_weak = img_weak.astype(np.float32)
            img_weak += (104, 117, 123)
            name = "./show/" + str(index) + "img_weak.jpg"
            # print('weak: ', index)
            cv2.imwrite(name, img_weak)

            return data_weak#, data_strong
        '''
        data_weak = torch.from_numpy(img_weak).float()

        img_weak = np.asarray(img_weak).transpose(1, 2, 0)
        img_weak = img_weak.astype(np.float32)
        img_weak += (104, 117, 123)
        # name = "./show/" + str(index) + "img_weak.jpg"
        # print('weak: ', index)
        # cv2.imwrite(name, img_weak)

        img_weak_PIL = Image.fromarray(cv2.cvtColor(img_weak.astype('uint8'), cv2.COLOR_BGR2RGB), "RGB")

        image_strong_aug = np.array(self.transform(img_weak_PIL))
        # name = "./show/" + str(index) + "image_strong_aug.jpg"
        # cv2.imwrite(name, image_strong_aug)
        # print('strong: ', index)
        image_strong_aug = image_strong_aug.astype(np.float32)
        image_strong_aug -= (104, 117, 123)

        img_strong = image_strong_aug.transpose(2, 0, 1)

        data_strong = torch.from_numpy(img_strong).float()

        return [data_weak, data_strong], target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    '''targets = []
    imgs = []
    #print('batch: ', batch)
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    print('imgs: ', len(imgs))
    print('targets: ', len(targets))
    return (torch.stack(imgs, 0), targets)'''

    targets = []
    imgs_weak = []
    imgs_strong = []
    # print('batch: ', batch)
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup[0]):
                imgs_weak.append(tup[0])
                imgs_strong.append(tup[1])
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs_weak, 0), torch.stack(imgs_strong, 0), targets)