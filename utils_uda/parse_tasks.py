import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from utils_uda.train import test
from utils_uda.SSTask import SSTask
from utils_uda.SSHead import linear_on_layer3, linear_on_layer3_square_img, linear_on_layer3_square_img_small
from dset_classes.DsetNoLabel import DsetNoLabel
from torch.utils import data as data_

from data.dataset import SsSim10k, SsCityscapes


def attenuation_loss(pred, target):
    # assuming that pred is (N x 5)... i.e. 4 classes and 1 variance prediction
    n_classes = 4
    n_iter = 100
    variance = torch.exp(pred[:, n_classes:])
    pred_class_logits = pred[:, 0:n_classes]

    expectation_softmax = None
    for t in range(n_iter):
        # draw a sample from a normal distribution with zero mean and unit variance
        eps = torch.normal(torch.zeros_like(pred_class_logits), torch.ones_like(pred_class_logits))
        # add noise with zero mean and predicted variance to the predicted class logits
        #print(eps)
        x = pred_class_logits + eps * torch.sqrt(variance)

        softmax = torch.nn.Softmax(dim=1)(x)

        if expectation_softmax is None:
            expectation_softmax = softmax / n_iter
        else:
            expectation_softmax += softmax / n_iter
    
    log_probabilities = torch.log(expectation_softmax)

    loss = torch.nn.NLLLoss()(log_probabilities, target)

    return loss

def alternative_attenuation_loss(pred, target):
    # assuming that pred is (N x 5)... i.e. 4 classes and 1 variance prediction
    n_classes = 4

    log_variance = pred[0, n_classes:] # assumes batchsize 1
    pred_class_logits = pred[:, 0:n_classes]

    # old implementation
    softmax = torch.nn.Softmax(dim=1)(pred_class_logits)
    log_probabilities = torch.log(softmax)
    loss = torch.nn.NLLLoss()(log_probabilities, target) / (torch.exp(log_variance)) + torch.abs(log_variance) / 10

    # new implementation... better in three aspects: first, logsoftmax is more stable than softmax followed by log (avoids overflow in exp).
    # Second, I turn the division of NLLLoss with exp(log_variance) into a substraction of log(NLLLoss) and log_variance (which again avoids overflow in exp)
    # Third, the abs is replaced by smoothL1Loss
    # log_probabilities = torch.nn.LogSoftmax(dim=1)(pred_class_logits)
    # smooth_l1 = torch.nn.SmoothL1Loss()(log_variance, torch.zeros_like(log_variance))
    # # loss = torch.exp(torch.log(torch.nn.NLLLoss()(log_probabilities, target)) - log_variance) + smooth_l1 / 10.
    # loss = torch.nn.NLLLoss()(log_probabilities, target) / (torch.exp(log_variance)) + smooth_l1 / 10. # turns out that log(NLLLoss) becomes a problem relatively quickly, since
    # the NLLLoss for highly confident examples is exactly 0. In practice, this was a more common problem than overflow in torch.exp(log_variance).

    return loss
    

def parse_tasks_od(opt, feature_extractor):

    ss_tasks = []

    if opt.rotation:
        source_ds = SsSim10k(opt, task="rotation")
        source_dl = data_.DataLoader(source_ds,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        pin_memory=True
                                        )

        target_ds = SsCityscapes(opt, task="rotation")
        target_dl = data_.DataLoader(target_ds,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        pin_memory=True
                                        )  

        head = linear_on_layer3(4, opt.width, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(feature_extractor.parameters()) + list(head.parameters()), 
                                lr=opt.lr_ss_tasks, momentum=0.9, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [opt.milestone_1 + opt.pre_training_steps, opt.milestone_2 + opt.pre_training_steps], gamma=0.1, last_epoch=-1)
        sstask = SSTask(feature_extractor, head, criterion, optimizer, scheduler,
                        source_dl, target_dl)
        sstask.assign_test(test)

        ss_tasks.append(sstask)

    if opt.quadrant:
        source_ds = SsSim10k(opt, task="quadrant")
        source_dl = data_.DataLoader(source_ds,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        pin_memory=True
                                        )

        target_ds = SsCityscapes(opt, task="quadrant")
        target_dl = data_.DataLoader(target_ds,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        pin_memory=True
                                        )  
        head = linear_on_layer3(4, opt.width, 4).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(feature_extractor.parameters()) + list(head.parameters()), 
                                lr=opt.lr_ss_tasks, momentum=0.9, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [opt.milestone_1 + opt.pre_training_steps, opt.milestone_2 + opt.pre_training_steps], gamma=0.1, last_epoch=-1)
        sstask = SSTask(feature_extractor, head, criterion, optimizer, scheduler,
                        source_dl, target_dl)
        sstask.assign_test(test)

        ss_tasks.append(sstask)


    if opt.crop_rot:
        source_ds = SsSim10k(opt, task="crop_rot")
        source_dl = data_.DataLoader(source_ds,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        pin_memory=True
                                        )

        target_ds = SsCityscapes(opt, task="crop_rot")
        target_dl = data_.DataLoader(target_ds,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False, \
                                        pin_memory=True
                                        )

        if opt.attenuation_loss:
            criterion = alternative_attenuation_loss
            head = linear_on_layer3_square_img(5, opt.width, int(opt.crop_size / 128)).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
            head = linear_on_layer3_square_img(4, opt.width, int(opt.crop_size / 128)).cuda()

        optimizer = optim.SGD(list(feature_extractor.parameters()) + list(head.parameters()), 
                                lr=opt.lr_ss_tasks, momentum=0.9, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [opt.milestone_1 + opt.pre_training_steps, opt.milestone_2 + opt.pre_training_steps], gamma=0.1, last_epoch=-1)
        sstask = SSTask(feature_extractor, head, criterion, optimizer, scheduler,
                        source_dl, target_dl)
        sstask.assign_test(test)

        ss_tasks.append(sstask)


    return ss_tasks
    

def parse_tasks_uda(args, ext, sc_tr_dataset, sc_te_dataset, tg_tr_dataset, tg_te_dataset):
    sstasks = []

    if args.rotation:
        print('Task: rotation prediction')
        from dset_classes.DsetSSRotRand import DsetSSRotRand

        digit = False
        if args.source in ['mnist', 'mnistm', 'svhn', 'svhn_exta', 'usps']:
            print('No rotation 180 for digits!')
            digit = True

        su_tr_dataset = DsetSSRotRand(DsetNoLabel(sc_tr_dataset), digit=digit)
        su_te_dataset = DsetSSRotRand(DsetNoLabel(sc_te_dataset), digit=digit)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        tu_tr_dataset = DsetSSRotRand(DsetNoLabel(tg_tr_dataset), digit=digit)
        tu_te_dataset = DsetSSRotRand(DsetNoLabel(tg_te_dataset), digit=digit)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        head = linear_on_layer3(4, args.width, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                                lr=args.lr_rotation, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler,
                     su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)

    if args.quadrant:
        print('Task: quadrant prediction')
        from dset_classes.DsetSSQuadRand import DsetSSQuadRand

        su_tr_dataset = DsetSSQuadRand(DsetNoLabel(sc_tr_dataset))
        su_te_dataset = DsetSSQuadRand(DsetNoLabel(sc_te_dataset))
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        tu_tr_dataset = DsetSSQuadRand(DsetNoLabel(tg_tr_dataset))
        tu_te_dataset = DsetSSQuadRand(DsetNoLabel(tg_te_dataset))
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        head = linear_on_layer3(4, args.width, 4).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                                lr=args.lr_quadrant, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler,
                     su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)
    
    if args.flip:
        print('Task: flip prediction')
        from dset_classes.DsetSSFlipRand import DsetSSFlipRand

        digit = False
        su_tr_dataset = DsetSSFlipRand(DsetNoLabel(sc_tr_dataset), digit=digit)
        su_te_dataset = DsetSSFlipRand(DsetNoLabel(sc_te_dataset), digit=digit)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        tu_tr_dataset = DsetSSFlipRand(DsetNoLabel(tg_tr_dataset), digit=digit)
        tu_te_dataset = DsetSSFlipRand(DsetNoLabel(tg_te_dataset), digit=digit)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        head = linear_on_layer3(2, args.width, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                                lr=args.lr_flip, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler,
                     su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)
    return sstasks
