import torch
from utils_uda.get_mmd import get_mmd
# from utils.make_tsne import create_and_plot_tsne

def output_to_labels(output):

    _, predicted = output.max(1)

    return predicted

def compute_error(labels, preds):
    total = labels.size(0)
    correct = preds.eq(labels).sum().item()
    return 1-correct/total


def test(dataloader, model):
    """Compute error_rate = 1 - accuracy for input model and dataloader."""
    model.eval()
    correct = 0.0
    total = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return 1-correct/total


def test_with_loss(dataloader, model, criterion):
    """Compute error_rate = 1 - accuracy for input model and dataloader, as well as loss with given criterion.
    
    args:
        dataloader
        model
        criterion
    
    returns:
        error_rate, mean loss over the dataset"""
    model.eval()
    correct = 0.0
    total = 0.0
    loss_list = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)

        _, predicted = outputs.max(1)
        batch_loss = criterion(outputs, labels)
        loss_list.append(batch_loss)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return 1-correct/total, sum(loss_list)/len(loss_list)

def train(args, net, ext, sstasks, criterion_cls, optimizer_cls, scheduler_cls,
            sc_tr_loader, sc_te_loader, tg_te_loader):
    """Train one epoch.
    
    Args:
        net: network that performs the original classification task (i.e. mnist), which is ResNet in our case
        ext: the feature extractor on which the heads for auxiliary self-supervised tasks are attached
        sstasks: self-supervised tasks
        criterion_cls: loss criterion/function used for the original classifcation task (i.e. classify mnist digits)
        optimizer_cls: the optimizer used to train the network that performs the original classification task
        scheduler_cls: not even used lmao
        sc_tr_loader: source training data loader
        sc_te_loader: source test data loader
        tg_te_loader: target test data loader
    
    returns:
        epoch_stats: [batch_idx, len(sc_tr_loader), mmd, target_test_error, source_test_error, ss_avg_error_per_task]
        supervised_train_stats: [train_loss, train_error]
        ss_train_stats: [source_train_loss, source_train_error, target_train_loss, target_train_error] for each task
        ss_test_stats: [source_test_error, target_test_error] for each task
    """
    net.train()
    for sstask in sstasks:
        sstask.head.train()
        sstask.scheduler.step()

    epoch_stats = []


    supervised_train_stats = []

    # ss_training_stats holds the source_loss, source_error, target_loss and target_error for each ss-task
    ss_train_stats = []
    ss_test_stats = []
    for sstask in sstasks:
        ss_train_stats.append([])
        ss_test_stats.append([])


    for batch_idx, (sc_tr_inputs, sc_tr_labels) in enumerate(sc_tr_loader):

        # do one batch of training for each self-supervised task.
        # since this happens inside the loop over sc_tr_loader (above), exactly one batch per self-supervised task is 
        # done for each batch done on the sc_tr_loader
        for i, sstask in enumerate(sstasks):
            # ss_loss, output = sstask.train_batch()

            # compute sstask loss and error on source and target training data separately
            source_outputs, source_labels, target_outputs, target_labels, source_loss, target_loss = sstask.train_batch_separate()

            source_preds = output_to_labels(source_outputs)
            source_error = compute_error(source_labels, source_preds)

            target_preds = output_to_labels(target_outputs)
            target_error = compute_error(target_labels, target_preds)

            ss_train_stats[i].append((batch_idx, len(sc_tr_loader), source_loss, source_error, target_loss, target_error))

        sc_tr_inputs, sc_tr_labels = sc_tr_inputs.cuda(), sc_tr_labels.cuda()
        optimizer_cls.zero_grad()
        outputs_cls = net(sc_tr_inputs)
        loss_cls = criterion_cls(outputs_cls, sc_tr_labels)

        preds = output_to_labels(outputs_cls)
        train_error = compute_error(sc_tr_labels, preds)
        supervised_train_stats.append((batch_idx, len(sc_tr_loader), loss_cls.item(), train_error))

        loss_cls.backward()
        optimizer_cls.step()


        if batch_idx % args.num_batches_per_test == 0:
            sc_te_err, source_test_loss = test_with_loss(sc_te_loader, net, criterion_cls)
            tg_te_err, target_test_loss = test_with_loss(tg_te_loader, net, criterion_cls)
            source_test_loss = source_test_loss.to(torch.device("cpu"))
            target_test_loss = target_test_loss.to(torch.device("cpu"))
            mmd, output_loader1, output_loader2, labels_loader1, labels_loader2 = get_mmd(sc_te_loader, tg_te_loader, ext)
            torch.save(output_loader1, args.outf + "/source_features.pth")
            torch.save(output_loader2, args.outf + "/target_features.pth")
            torch.save(labels_loader1, args.outf + "/source_labels.pth")
            torch.save(labels_loader2, args.outf + "/target_labels.pth")
            # create_and_plot_tsne(output_loader1, output_loader2, args.outf)

            us_te_err_av = []
            for i, sstask in enumerate(sstasks):
                err_av, err_sc, err_tg = sstask.test()
                ss_test_stats[i].append((batch_idx, len(sc_tr_loader), err_sc, err_tg))
                us_te_err_av.append(err_av)
            
            epoch_stats.append((batch_idx, len(sc_tr_loader), mmd, tg_te_err, sc_te_err, us_te_err_av, source_test_loss, target_test_loss))
            display = ('Iteration %d/%d:' %(batch_idx, len(sc_tr_loader))).ljust(24)
            display += '%.2f\t%.2f\t\t%.2f\t\t' %(mmd, tg_te_err*100, sc_te_err*100)
            for err in us_te_err_av:
                display += '%.2f\t' %(err*100)
            print(display)
    return epoch_stats, supervised_train_stats, ss_train_stats, ss_test_stats
