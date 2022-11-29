# tensorboard --logdir /home/erik/phd/code/dacs_fork/saved/DeepLabv2/
# python3 ss_training.py -n UDA -c ~/phd/code/dacs_fork/DACS/configs/configSS.json
from utils_uda.parse_tasks import parse_tasks_od
import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Res_Deeplab, Extractor

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateUDA import evaluate

import time

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()



def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label), device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])


            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch)+ id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages', str(epoch)+ id + '.png'))

def _save_checkpoint(iteration, model, optimizer, config, ema_model, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir, f'checkpoint-iter{iteration - save_checkpoint_every}.pth'))
            except:
                pass

def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model
    

def main():
    ss_params = {}
    ss_params["cityscapes"]= {}
    ss_params["gta"] = {}
    print(config)

    best_mIoU = 0

    cudnn.enabled = True

    # create network
    model = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    #saved_state_dict = torch.load(args.restore_from)
        # load pretrained parameters
    if restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(restore_from)
    else:
        saved_state_dict = torch.load(restore_from)

    # Copy loaded parameters to model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
    model.load_state_dict(new_params)

    # init ema-model
    ema_model = None

    # convert to synch batch norm. What is that?
    print(f"len gpus: {len(gpus)}")
    if len(gpus)>1:
        if use_sync_batchnorm:
            model = convert_model(model)
            model = DataParallelWithCallback(model, device_ids=gpus)
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus)
    model.train()
    model.cuda()

    cudnn.benchmark = True
    # if dataset == 'cityscapes':
    #     data_loader = get_loader('cityscapes')
    #     data_path = get_data_path('cityscapes')
    #     if random_crop:
    #         data_aug = Compose([RandomCrop_city(input_size)])
    #     else:
    #         data_aug = None

    #     #data_aug = Compose([RandomHorizontallyFlip()])
    #     train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug, img_size=input_size, img_mean = IMG_MEAN)
        
    # train_dataset_size = len(train_dataset)
    # print ('dataset size: ', train_dataset_size)

    #New loader for Domain transfer
    if True:
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        if random_crop:
            data_aug = Compose([RandomCrop_gta(input_size)])
        else:
            data_aug = None

        #data_aug = Compose([RandomHorizontallyFlip()])
        train_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt',
            augmentations=data_aug, img_size=(1280,720), mean=IMG_MEAN)

    trainloader = data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    trainloader_iter = iter(trainloader)
    print('gta size:',len(trainloader))

    # gather the layers from DeepLab as the feature extractor which is used by the self-supervised training
    extractor = Extractor(model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4)

    # define the parameters for the Ss source and target dataset
    ss_params["cityscapes"]["data_path"] = get_data_path('cityscapes')
    ss_params["cityscapes"]["is_transform"] = True
    ss_params["cityscapes"]["data_aug"] = None
    ss_params["cityscapes"]["img_mean"] = IMG_MEAN

    ss_params["gta"]["data_path"] = get_data_path('gta')
    ss_params["gta"]["list_path"] = './data/gta5_list/train.txt'
    ss_params["gta"]["data_aug"] = None
    ss_params["gta"]["img_size"] = (1280,720)
    ss_params["gta"]["img_mean"] = IMG_MEAN

    if train_ss:
        # create the self-supervised task
        ss_task = parse_tasks_od(config, ss_params, extractor, log_dir)


    #Load new data for domain_transfer

    # optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer = optim.SGD(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        if len(gpus) > 1:
            optimizer = optim.Adam(model.module.optim_parameters(learning_rate_object),
                        lr=learning_rate, momentum=momentum,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.optim_parameters(learning_rate_object),
                        lr=learning_rate, weight_decay=weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    start_iteration = 0

    if args.resume:
        start_iteration, model, optimizer, ema_model = _resume_checkpoint(args.resume, model, optimizer, ema_model)

    accumulated_loss_l = []
    accumulated_loss_u = []
    accumulated_loss_src = []
    accumulated_loss_trg = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)

    epochs_since_start = 0

    n_correct_src = 0
    n_correct_trg = 0
    n_total_ss = 0
    for i_iter in range(start_iteration, num_iterations):
        model.train()

        loss_u_value = 0
        loss_l_value = 0

        optimizer.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer, i_iter)

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        #if random_flip:
        #    weak_parameters={"flip":random.randint(0,1)}
        #else:
        weak_parameters={"flip": 0}

        if True:
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.cuda().long()

            #images, labels = weakTransform(weak_parameters, data = images, target = labels)

            pred = interp(model(images))
            L_l = loss_calc(pred, labels) # Cross entropy loss for labeled data
            #L_l = torch.Tensor([0.0]).cuda()

            loss = L_l

            if len(gpus) > 1:
                #print('before mean = ',loss)
                loss = loss.mean()
                #print('after mean = ',loss)
                loss_l_value += L_l.mean().item()
                # if train_unlabeled:
                #     loss_u_value += L_u.mean().item()
            else:
                loss_l_value += L_l.item()
                # if train_unlabeled:
                #     loss_u_value += L_u.item()


            # TODO 
            loss.backward()
            optimizer.step()
        else:
            loss = 0.

        if train_ss:
            for ss_steps_per_batch in range(1):
                n_total_ss += 1
                s = f"ss_task_"
                (source_outputs, source_labels, target_outputs, target_labels,
                    source_loss, target_loss, source_inputs, target_inputs) = ss_task.train_batch_separate()
                if config["training"]["ss"]["attenuation_loss"]:
                    source_pred_label = torch.argmax(source_outputs[:, 0:4], axis=1)
                    target_pred_label = torch.argmax(target_outputs[:, 0:4], axis=1)
                else:
                    source_pred_label = torch.argmax(source_outputs, axis=1)
                    target_pred_label = torch.argmax(target_outputs, axis=1) 
                source_correct = source_pred_label == source_labels
                if source_correct.item():
                    n_correct_src += 1
                target_correct = target_pred_label == target_labels
                if target_correct.item():
                    n_correct_trg += 1

        s = 'iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f}'.format(
                i_iter, num_iterations, loss_l_value, loss_u_value)
        if train_ss:
            # print losses
            s = s + ', ss_loss_src = {0:.3f}, ss_loss_trg = {1:.3f}'.format(
                source_loss, target_loss)
        print(s)

        # if i_iter % 100 == 0:
        #     print("ss_acc_src = {0:.2f}, ss_acc_trg {1:.2f}".format(n_correct_src / n_total_ss, n_correct_trg / n_total_ss))
        #     n_total_ss = 0
        #     n_correct_src = 0
        #     n_correct_trg = 0

        # save checkpoint
        if i_iter % save_checkpoint_every == 0 and i_iter!=0:
            if epochs_since_start * len(trainloader) < save_checkpoint_every:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, overwrite=False)
            else:
                _save_checkpoint(i_iter, model, optimizer, config, ema_model)

        # write losses to tensorboard
        if config['utils']['tensorboard']:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l.append(loss_l_value)
            if train_unlabeled:
                accumulated_loss_u.append(loss_u_value)
            if train_ss:
                accumulated_loss_src.append(source_loss)
                accumulated_loss_trg.append(target_loss)
            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss', np.mean(accumulated_loss_l), i_iter)
                accumulated_loss_l = []
                tensorboard_writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], i_iter)

                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/Unsupervised loss', np.mean(accumulated_loss_u), i_iter)
                    accumulated_loss_u = []

                if train_ss:
                    print("ss_acc_src = {0:.2f}, ss_acc_trg {1:.2f}".format(n_correct_src / n_total_ss, n_correct_trg / n_total_ss))
                    tensorboard_writer.add_scalar('Training/ss_acc_src', n_correct_src / n_total_ss, i_iter)
                    tensorboard_writer.add_scalar('Training/ss_acc_trg', n_correct_trg / n_total_ss, i_iter)
                    tensorboard_writer.add_scalar('Training/SS src loss', np.mean(accumulated_loss_src), i_iter)
                    accumulated_loss_src = []
                    tensorboard_writer.add_scalar('Training/SS trg loss', np.mean(accumulated_loss_trg), i_iter)
                    accumulated_loss_trg = []
                    n_total_ss = 0
                    n_correct_src = 0
                    n_correct_trg = 0

        # evaluate model on both cityscapes and gta
        if i_iter % val_per_iter == 0 and i_iter != 0:
            model.eval()
            # if dataset == 'cityscapes':
            print("Evaluating on gta...")
            gta_mIoU, gta_eval_loss = evaluate(model, 'gta', ignore_label=255, input_size=None, save_dir=checkpoint_dir)
            print("Evaluating on Cityscapes...")
            cs_mIoU, cs_eval_loss = evaluate(model, 'cityscapes', ignore_label=250, input_size=(512,1024), save_dir=checkpoint_dir)

            model.train()

            if cs_mIoU > best_mIoU and save_best_model:
                best_mIoU = cs_mIoU
                _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)

            if config['utils']['tensorboard']:
                tensorboard_writer.add_scalar('Validation/cs-mIoU', cs_mIoU, i_iter)
                tensorboard_writer.add_scalar('Validation/cs-Loss', cs_eval_loss, i_iter)
                tensorboard_writer.add_scalar('Validation/gta-mIoU', gta_mIoU, i_iter)
                tensorboard_writer.add_scalar('Validation/gta-Loss', gta_eval_loss, i_iter)

        if save_unlabeled_images and i_iter % save_checkpoint_every == 0:
            _, pred_u_s = torch.max(pred, dim=1)
            save_image(pred_u_s[0].cpu(),i_iter,'pred1',palette.CityScpates_palette)

    # after finished training, save checkpoint and evaluate model
    _save_checkpoint(num_iterations, model, optimizer, config, ema_model)
    if train_ss:
        torch.save(ss_task.head.state_dict(), os.path.join(checkpoint_dir, "head_{}".format("final")))
    model.eval()
    if dataset == 'cityscapes':
        mIoU, val_loss = evaluate(model, dataset, ignore_label=250, input_size=(512,1024), save_dir=checkpoint_dir)
    model.train()
    if mIoU > best_mIoU and save_best_model:
        best_mIoU = mIoU
        _save_checkpoint(i_iter, model, optimizer, config, ema_model, save_best=True)
    if config['utils']['tensorboard']:
        tensorboard_writer.add_scalar('Validation/mIoU', mIoU, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss', val_loss, i_iter)
    end = timeit.default_timer()
    print('Total time: ' + str(end-start) + 'seconds')


if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if False:#args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']


    if config['pretrained'] == 'coco':
        restore_from = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

    num_classes=19
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label'] 

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    #unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    train_ss = config['training']['ss']['train_ss']


    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0,1,2,3)[:args.gpus]

    main()


