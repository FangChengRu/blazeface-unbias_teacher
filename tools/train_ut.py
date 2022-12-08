from __future__ import print_function
import sys
sys.path.append('./')
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from blazeface.data.dataset.wider_face import WiderFaceDetection, detection_collate
from blazeface.data.transform.data_augment import preproc
from config.config import cfg_blaze
from blazeface.models.loss.custom_loss2 import MultiBoxLoss as CustomLoss2
from blazeface.models.module.prior_box import PriorBox
import time
import datetime
import math
from blazeface.utils.box_utils import decode
from blazeface.models.module.py_cpu_nms import py_cpu_nms
import cv2
from blazeface.models.net_blaze import Blaze
from blazeface.utils.data_loader import data_prefetcher, data_prefetcher2
import logging
import numpy as np
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from blazeface.evaluator.widerface_evaluate import evaluation
from PIL import Image
import torch.nn.functional as F


def seed_everything(seed: int):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed = 41
seed_everything(seed)

LOG_FORMAT = "%(asctime)s %(levelname)s : %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--training_dataset', default='./widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, help='momentum')
parser.add_argument('--resume_net', default=None , help='resume net for retraining , ex : \'./weights/Blaze_burn_in.pth\'')
parser.add_argument('--resume_net_teacher', default=None, help='resume teacher net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./runs/', help='Location to save checkpoint models')

args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = cfg_blaze
net = Blaze(cfg=cfg)
net_teacher = Blaze(cfg=cfg)
for param in net_teacher.parameters():
    param.detach_()

'''print("Printing student net...")
print(net)
print("Printing teacher net...")
print(net_teacher)'''

writer = SummaryWriter(comment=cfg['name'])

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
epoch_burn_in = cfg['epoch_burn_in']
epoch_max = cfg['epoch_max']
gpu_train = cfg['gpu_train']
ema_keep_rate = cfg['ema_keep_rate']
sup_loss_weight = cfg['sup_loss_weight']
unsup_loss_weight = cfg['unsup_loss_weight']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder
device = 0

labeled_split = 0.01  # 1% data
shuffle_dataset = True
random_seed= 42

if args.resume_net is not None:
    logging.info('Loading resume student network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.resume_net_teacher is not None:
    logging.info('Loading resume teacher network...')
    state_dict = torch.load(args.resume_net_teacher)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net_teacher.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
    net_teacher = torch.nn.DataParallel(net_teacher).cuda()

else:
    net = net.cuda()
    net_teacher = net_teacher.cuda()

cudnn.benchmark = True


# optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.RMSprop(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
# criterion = MultiBoxLoss(cfg, 0.35, True, 0, True, 7, 0.35, False)
criterion = CustomLoss2(cfg, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    # net.train()
    epoch = 0 + args.resume_epoch
    logging.info('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))
    # data_strong = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean), aug_strong=True)

    # dataset = Two_Dataset(data_weak, data_strong)

    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(labeled_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    unlabeled_indices, labeled_indices = indices[split:], indices[:split]


    # Creating PT data samplers and loaders:
    np.random.seed(seed)
    unlabeled_sampler = SubsetRandomSampler(unlabeled_indices)
    # print('unlabeled_indices: ', unlabeled_indices)
    labeled_sampler = SubsetRandomSampler(labeled_indices)
    # print('labeled_indices: ', labeled_indices)

    unlabeled_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=unlabeled_sampler, num_workers=num_workers,
                                                        collate_fn=detection_collate, pin_memory=True)

    labeled_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      sampler=labeled_sampler, num_workers=num_workers,
                                                      collate_fn=detection_collate, pin_memory=True)

    # epoch & iteration
    epoch_size_labeled = math.ceil(len(labeled_loader)) * 2
    burn_in_iter = epoch_burn_in * epoch_size_labeled
    epoch_size_unlabeled = math.ceil(len(unlabeled_loader))
    max_iter = burn_in_iter + (epoch_max - epoch_burn_in) * epoch_size_unlabeled

    stepvalues = (cfg['decay1'] * epoch_size_labeled,
                  cfg['decay2'] * epoch_size_labeled,
                  cfg['decay3'] * epoch_size_labeled,
                  cfg['decay4'] * epoch_size_labeled)
    step_index = 0

    if 200 >= args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size_labeled
    elif args.resume_epoch > 200:
        start_iter = 200 * epoch_size_labeled + (args.resume_epoch-200) * epoch_size_unlabeled
    else:
        start_iter = 0

    weak = 0
    # burn-in
    for iteration in range(start_iter, max_iter):
        if iteration < burn_in_iter:
            if iteration % epoch_size_labeled == 0:
                batch_iterator_label = data_prefetcher2(labeled_loader, device)
                epoch += 1

            if weak % 2 == 0:
                labeled_images_weak, labeled_images_strong, targets = batch_iterator_label.next()
                images = labeled_images_strong
                # print('strong data')
            else:
                images = labeled_images_weak
                # print('weak data')
            weak += 1

            load_t0 = time.time()
            net.train()

            if iteration in stepvalues:
                step_index += 1
            lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size_labeled)
            writer.add_scalar('learning_rate', lr, iteration)

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss =  loss_l + loss_c
            writer.add_scalar('loss_location', loss_l, iteration)
            writer.add_scalar('loss_class', loss_c, iteration)
            writer.add_scalar('loss', loss, iteration)

            ## fp32 training
            loss.backward()

            optimizer.step()

            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if iteration % 1 == 0:
                logging.info('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Total: {:.4f} Loc: {:.4f} Cla: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                      .format(epoch, epoch_max, (iteration % epoch_size_labeled) + 1,
                      epoch_size_labeled, iteration + 1, max_iter, loss.item(), loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

            if (iteration+1) == burn_in_iter:
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_burn_in.pth')
                # student
                print('********************Student model performance: ********************')
                validate(net, show_image=True)
                pred = './blazeface/evaluator/widerface_evaluate/widerface_txt/'
                gt_path = './blazeface/evaluator/widerface_evaluate/ground_truth/'
                easy, med, hard = evaluation.evaluation_output(pred, gt_path)
                writer.add_scalar('student_model_easy', easy, iteration)
                writer.add_scalar('student_model_med', med, iteration)
                writer.add_scalar('student_model_hard', hard, iteration)

        # semi-supervised
        else:
            # Load unlabeled data
            if (iteration - burn_in_iter) % epoch_size_unlabeled == 0:
                # create batch iterator
                batch_iterator_unlabel = data_prefetcher2(unlabeled_loader, device)
                epoch += 1

            unlabeled_images_weak, unlabeled_images_strong, _ = batch_iterator_unlabel.next()

            # Load labeled data
            if (iteration - burn_in_iter - start_iter) % epoch_size_labeled == 0:
                batch_iterator_label = data_prefetcher2(labeled_loader, device)

            if weak % 2 == 0:
                labeled_images_weak, labeled_images_strong, labeled_targets = batch_iterator_label.next()
                labeled_images = labeled_images_strong
                # print('strong data')
            else:
                labeled_images = labeled_images_weak
                # print('weak data')
            weak += 1

            load_t0 = time.time()
            net.train()

            if iteration == burn_in_iter:
                # update_ema_variables(net, net_teacher, 0)  # only update train parameters, but should update all parameters
                import copy
                net_teacher = copy.deepcopy(net)
            else:
                net_teacher.train()
                update_ema_variables(net, net_teacher, ema_keep_rate)

            pseudo = pseudo_generator(net_teacher, unlabeled_images_weak, show_image=True)
            show(net, unlabeled_images_strong)

            lr = initial_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            '''
            # learning rate decay
            if (epoch-200) < (epoch_max*0.65):
                lr = initial_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif ((epoch_max-200)*0.65) <= (epoch-200) < ((epoch_max-200)*0.8):
                lr = initial_lr * (10 ** (-1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif ((epoch_max-200)*0.8) <= (epoch-200) < ((epoch_max-200)*0.875):
                lr = initial_lr * (10 ** (-2))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif ((epoch_max-200) * 0.875) <= (epoch-200) < ((epoch_max-200) * 0.975):
                lr = initial_lr * (10 ** (-3))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = initial_lr * (10 ** (-4))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr'''
            writer.add_scalar('learning_rate', lr, iteration)

            # forward
            # unsupervised
            out_pseudo = net(unlabeled_images_strong)
            # supervised
            out = net(labeled_images)

            optimizer.zero_grad()
            loss_l_pseudo, loss_c_pseudo = criterion(out_pseudo, priors, pseudo)
            loss_l, loss_c = criterion(out, priors, labeled_targets)
            loss = (loss_l_pseudo + loss_c_pseudo) * unsup_loss_weight + (loss_l + loss_c) * sup_loss_weight
            writer.add_scalar('loss_location', loss_l, iteration)
            writer.add_scalar('loss_class', loss_c, iteration)
            writer.add_scalar('loss_location_pseudo', loss_l_pseudo, iteration)
            writer.add_scalar('loss_class_pseudo', loss_c_pseudo, iteration)
            writer.add_scalar('loss', loss, iteration)

            ## fp32 training
            loss.backward()

            optimizer.step()

            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if iteration % 40 == 0:
                logging.info(
                    'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Total: {:.4f} Loc: {:.4f} Cla: {:.4f} Loc_pseudo: {:.4f} Cla_pseudo: {:.4f}|| LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch, epoch_max, ((iteration-burn_in_iter) % epoch_size_unlabeled) + 1,
                            epoch_size_unlabeled, iteration + 1, max_iter, loss.item(), loss_l.item(), loss_c.item(),
                            loss_l_pseudo.item(), loss_c_pseudo.item(), lr, batch_time,
                            str(datetime.timedelta(seconds=eta))))

            # validation
            if (iteration - burn_in_iter + 1) % (epoch_size_unlabeled * 1) == 0:
                # student
                print('********************Student model performance: ********************')
                validate(net, show_image=True)
                pred = './blazeface/evaluator/widerface_evaluate/widerface_txt/'
                gt_path = './blazeface/evaluator/widerface_evaluate/ground_truth/'
                easy, med, hard = evaluation.evaluation_output(pred, gt_path)
                writer.add_scalar('student_model_easy', easy, iteration)
                writer.add_scalar('student_model_med', med, iteration)
                writer.add_scalar('student_model_hard', hard, iteration)

                print('********************Teacher model performance: ********************')
                # teacher
                validate(net_teacher, show_image=True)
                pred = './blazeface/evaluator/widerface_evaluate/widerface_txt/'
                gt_path = './blazeface/evaluator/widerface_evaluate/ground_truth/'
                easy, med, hard = evaluation.evaluation_output(pred, gt_path)
                writer.add_scalar('teacher_model_easy', easy, iteration)
                writer.add_scalar('teacher_model_med', med, iteration)
                writer.add_scalar('teacher_model_hard', hard, iteration)

            if (iteration - burn_in_iter + 1) % (epoch_size_unlabeled * 1) == 0:
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_student_epoch_' + str(epoch) + '.pth')
                torch.save(net_teacher.state_dict(), save_folder + cfg['name'] + '_teacher_epoch_' + str(epoch) + '.pth')

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_student_final.pth')
    torch.save(net_teacher.state_dict(), save_folder + cfg['name'] + '_teacher_final.pth')


def show(model, images, show_image=True):
    save_image = show_image  # Show pseudo images
    confidence_threshold = 0.2
    nms_threshold = 0.5
    model.eval()
    for i, image in enumerate(images):
        _, im_height, im_width = image.shape
        image = image.unsqueeze(0)
        loc, conf = model(image)  # forward pass
        conf = F.softmax(conf, dim=-1)
        # loc = Variable(loc.detach().data, requires_grad=False)
        # conf = Variable(conf.detach().data, requires_grad=False)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes.clamp(max=1, min=0.00001)
        # boxes = boxes * im_height
        boxes = boxes.cpu().numpy()
        boxes = np.nan_to_num(boxes)
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        # save image
        if save_image:
            image = image.squeeze(0)
            image = np.asarray(image.cpu()).transpose(1, 2, 0)
            image = image.astype(np.float32)
            image += (104, 117, 123)
            img_PIL = Image.fromarray(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB), "RGB")
            img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            for b in dets:
                b[0] = int(b[0] * im_height)
                b[1] = int(b[1] * im_height)
                b[2] = int(b[2] * im_height)
                b[3] = int(b[3] * im_height)
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 2
                cv2.putText(img, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # save image
            if not os.path.exists("./pseudo/"):
                os.makedirs("./pseudo/")
            name = "./pseudo/" + str(i) + "_student_output.jpg"
            cv2.imwrite(name, img)


def pseudo_generator(model, images, show_image=False):
    save_image = show_image  # Show pseudo images
    pseudo = []
    confidence_threshold = 0.5
    nms_threshold = 0.5
    model.eval()
    for i, image in enumerate(images):
        _, im_height, im_width = image.shape
        image = image.unsqueeze(0)
        loc, conf = model(image)  # forward pass
        conf = F.softmax(conf, dim=-1)
        loc = Variable(loc.detach().data, requires_grad=False)
        conf = Variable(conf.detach().data, requires_grad=False)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes.clamp(max=1, min=0.00001)
        # boxes = boxes * im_height
        boxes = boxes.cpu().numpy()
        boxes = np.nan_to_num(boxes)
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        annotations = np.zeros((0, 5))
        if len(dets) == 0:
            pseudo.append(torch.from_numpy(annotations).float().to(device))
        else:
            for box in dets:
                annotation = np.zeros((1, 5))
                annotation[0, 0] = box[0]
                annotation[0, 1] = box[1]
                annotation[0, 2] = box[2]
                annotation[0, 3] = box[3]
                annotation[0, 4] = 1

                annotations = np.append(annotations, annotation, axis=0)

            pseudo.append(torch.from_numpy(annotations).float().to(device))

        # save image
        if save_image:
            image = image.squeeze(0)
            image = np.asarray(image.cpu()).transpose(1, 2, 0)
            image = image.astype(np.float32)
            image += (104, 117, 123)
            img_PIL = Image.fromarray(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB), "RGB")
            img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            for b in dets:
                b[0] = int(b[0] * im_height)
                b[1] = int(b[1] * im_height)
                b[2] = int(b[2] * im_height)
                b[3] = int(b[3] * im_height)
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 2
                cv2.putText(img, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # save image
            if not os.path.exists("./pseudo/"):
                os.makedirs("./pseudo/")
            name = "./pseudo/" + str(i) + ".jpg"
            cv2.imwrite(name, img)
    return pseudo


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = 1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def update_ema_variables(model, ema_model, alpha):  # , global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def validate(model, show_image=False):
    long_side=320
    save_folder='./blazeface/evaluator/widerface_evaluate/widerface_txt'
    cpu=False
    dataset_folder='./widerface/val/images/'
    confidence_threshold=0.01
    nms_threshold=0.5
    save_image=show_image
    vis_thres=0.17

    model.eval()

    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    model = model.to(device)

    # testing dataset
    testset_folder = dataset_folder
    testset_list = dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = long_side
        max_size = long_side
        im_shape = img.shape
        im_size_max = np.max(im_shape[0:2])

        # blazeface resize
        height, width, _ = im_shape
        image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
        image_t[:, :] = (104, 117, 123)
        image_t[0:0 + height, 0:0 + width] = img
        img = cv2.resize(image_t, (max_size, max_size))
        resize = float(target_size) / float(im_size_max)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf = model(img)  # forward pass
        conf = F.softmax(conf, dim=-1)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes.clamp(max=1, min=0.00001)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        boxes = np.nan_to_num(boxes)
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        # --------------------------------------------------------------------
        save_name = save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        # save image
        if save_image:
            for b in dets:
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

if __name__ == '__main__':
    train()
