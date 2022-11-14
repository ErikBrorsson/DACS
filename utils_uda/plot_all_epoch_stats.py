import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def prune_ticks_labels(ticks, labels):
    ticks = np.asarray(ticks)
    labels = np.asarray(labels)

    if len(labels) > 15 and len(labels) <= 50:
        idx = np.where(labels % 5 == 0)
        labels = labels[idx]
        ticks = ticks[idx]
    elif len(labels) > 50 and len(labels) <= 100:
        idx = np.where(labels % 10 == 0)
        labels = labels[idx]
        ticks = ticks[idx]
    elif len(labels) > 100:
        idx = np.where(labels % 25 == 0)
        labels = labels[idx]
        ticks = ticks[idx]
    return ticks, labels

def parse_all_epoch_stats(all_epoch_stats, prune=True):
    base_iter_count = 0
    ticks = []
    labels = []
    xs = []
    tg_te_err = []
    sc_te_err = []
    us_te_err = []
    mmd = []

    source_test_loss = []
    target_test_loss = []

    for epoch, epoch_stats in enumerate(all_epoch_stats):
        for stats in epoch_stats:
            mmd.append(stats[2])
            tg_te_err.append(stats[3])
            sc_te_err.append(stats[4])
            us_te_err.append(stats[5])
            xs.append(base_iter_count + stats[0])
            source_test_loss.append(stats[6])
            target_test_loss.append(stats[7])
        base_iter_count += stats[1]
        ticks.append(base_iter_count)
        labels.append(epoch+1)

    if prune:
        ticks, labels = prune_ticks_labels(ticks, labels)
    us_te_err = np.asarray(us_te_err)
    return ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd, source_test_loss, target_test_loss

# def parse_ss_train_stats(supervised_train_stats, prune=True):
#     base_iter_count = 0
#     ticks = []
#     labels = []
#     xs = []
#     source_error = []
#     target_loss = []
#     target_error = []
#     source_loss = []

#     for epoch, epoch_stats in enumerate(supervised_train_stats):
#         for stats in epoch_stats:
#             source_loss.append(stats[2])
#             source_error.append(stats[3])
#             target_loss.append(stats[4])
#             target_error.append(stats[5])
#             xs.append(base_iter_count + stats[0])
#         base_iter_count += stats[1]
#         ticks.append(base_iter_count)
#         labels.append(epoch+1)

#     if prune:
#         ticks, labels = prune_ticks_labels(ticks, labels)
#     return ticks, labels, xs, source_loss, source_error, target_loss, target_error

def parse_supervised_train_stats(supervised_train_stats, prune=True):
    base_iter_count = 0
    ticks = []
    labels = []
    xs = []
    source_train_error = []
    source_train_loss = []

    for epoch, epoch_stats in enumerate(supervised_train_stats):
        for stats in epoch_stats:
            source_train_loss.append(stats[2])
            source_train_error.append(stats[3])
            xs.append(base_iter_count + stats[0])
        base_iter_count += stats[1]
        ticks.append(base_iter_count)
        labels.append(epoch+1)

    if prune:
        ticks, labels = prune_ticks_labels(ticks, labels)
    return ticks, labels, xs, source_train_loss, source_train_error

def parse_ss_train_stats(ss_train_stats, prune=True):
    base_iter_count = 0
    ticks = []
    labels = []
    xs = []
    source_train_error = []
    source_train_loss = []
    target_train_error = []
    target_train_loss = []

    for epoch, epoch_stats in enumerate(ss_train_stats):
        for stats in epoch_stats[0]: # TODO [0] for sstask rotation, should add loop over sstasks
            source_train_loss.append(stats[2])
            source_train_error.append(stats[3])
            target_train_loss.append(stats[4])
            target_train_error.append(stats[5])
            xs.append(base_iter_count + stats[0])
        base_iter_count += stats[1]
        ticks.append(base_iter_count)
        labels.append(epoch+1)

    if prune:
        ticks, labels = prune_ticks_labels(ticks, labels)
    return ticks, labels, xs, source_train_loss, source_train_error, target_train_loss, target_train_error

def parse_ss_test_stats(ss_test_stats, prune=True):
    base_iter_count = 0
    ticks = []
    labels = []
    xs = []
    source_test_error = []
    target_test_error = []

    for epoch, epoch_stats in enumerate(ss_test_stats):
        for stats in epoch_stats[0]: # TODO [0] here corresponds to the rotation task, add loop over tasks here
            source_test_error.append(stats[2])
            target_test_error.append(stats[3])
            xs.append(base_iter_count + stats[0])
        base_iter_count += stats[1]
        ticks.append(base_iter_count)
        labels.append(epoch+1)

    if prune:
        ticks, labels = prune_ticks_labels(ticks, labels)
    return ticks, labels, xs, source_test_error, target_test_error

def plot_all_epoch_stats(all_epoch_stats, outf):
    ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd, source_test_loss, target_test_loss = parse_all_epoch_stats(all_epoch_stats)

    mmd = np.asarray(mmd)
    plt.plot(xs, mmd / np.max(mmd)*100, color='k', label='normalized mmd')
    plt.plot(xs, np.asarray(tg_te_err)*100, color='r', label='target')
    plt.plot(xs, np.asarray(sc_te_err)*100, color='b', label='source')

    colors = ['g', 'm', 'c']
    for i in range(us_te_err.shape[1]):
        plt.plot(xs, np.asarray(us_te_err[:,i])*100, color=colors[i], label='self-sup %d' %(i+1))

    plt.xticks(ticks, labels)
    plt.xlabel('epoch')
    plt.ylabel('test error (%)')
    plt.legend()
    plt.savefig('%s/loss.pdf' %(outf))
    plt.close()

def plot_train_val_loss(supervised_train_stats, all_epoch_stats, outf):
    import matplotlib.pyplot as plt
    ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd, source_test_loss, target_test_loss = parse_all_epoch_stats(all_epoch_stats)

    fig, ax = plt.subplots(ncols=2, figsize=(8,4))

    # plot val error for source and target data
    ax[0].plot(xs, np.asarray(tg_te_err)*100, color='r', linestyle="solid", label='target_val')
    ax[0].plot(xs, np.asarray(sc_te_err)*100, color='b', linestyle="solid", label='source_val')
    plt.sca(ax[0])
    plt.xticks(ticks, labels)

    ax[1].plot(xs, np.asarray(target_test_loss), color='r', linestyle="solid", label='target_val')
    ax[1].plot(xs, np.asarray(source_test_loss), color='b', linestyle="solid", label='source_val')
    plt.sca(ax[1])
    plt.xticks(ticks, labels)

    ticks, labels, xs, source_train_loss, source_train_error = parse_supervised_train_stats(supervised_train_stats)
    ax[0].plot(xs, np.asarray(source_train_error)*100, color='b', linestyle="dashed", label='source_train')
    ax[1].plot(xs, np.asarray(source_train_loss), color='b', linestyle="dashed", label='source_train')



    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('test error (%)')
    ax[0].legend()

    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend()


    plt.savefig('%s/train_val_loss.pdf' %(outf))
    plt.close()

def plot_saved_classification_stats(supervised_stats_path, metrics_path, outf):

    supervised_train_stats = torch.load(supervised_stats_path)
    all_epoch_stats = torch.load(metrics_path)

    ticks, labels, xs, tg_te_err, sc_te_err, us_te_err, mmd, source_test_loss, target_test_loss = parse_all_epoch_stats(all_epoch_stats)

    fig, ax = plt.subplots(ncols=2, figsize=(16,8))
    fig.tight_layout(pad=10.0)

    ax0_twin = ax[0].twinx()
    ax1_twin = ax[1].twinx()

    # plot val error for source and target data
    p0 = ax0_twin.plot(xs, np.asarray(tg_te_err)*100, color='r', linestyle="dotted", label='target_val')
    p1 = ax[0].plot(xs, np.asarray(sc_te_err)*100, color='r', linestyle="solid", label='source_val')
    plt.sca(ax[0])
    plt.xticks(ticks, labels)

    # p20 = ax1_twin.plot(xs, np.asarray(target_test_loss), color='r', linestyle="dotted", label='target_val')
    p21 = ax[1].plot(xs, np.asarray(source_test_loss), color='r', linestyle="solid", label='source_val')
    plt.sca(ax[1])
    plt.xticks(ticks, labels)

    test_xs = xs
    ticks, labels, xs, source_train_loss, source_train_error = parse_supervised_train_stats(supervised_train_stats)

    avg_sc_tr_loss = []
    avg_sc_tr_err = []
    prev_indx = 0
    for t in test_xs:
        if t == 0:
            continue
        avg_sc_tr_loss.append(np.mean(source_train_loss[prev_indx:t]))
        avg_sc_tr_err.append(np.mean(source_train_error[prev_indx:t]))

        prev_indx = t
                
    p2 = ax[0].plot(xs, np.asarray(source_train_error)*100, color='b', linestyle="solid",  alpha=0.2)
    p3 = ax[0].plot(test_xs[1:], np.asarray(avg_sc_tr_err)*100, color='b', linestyle="solid", label='source_train')


    lns = p3 + p1 + p0
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc="upper left", framealpha=0.90)
    ax0_twin.legend(lns, labs, loc="upper left", framealpha=0.90)

    p22 = ax[1].plot(xs, np.asarray(source_train_loss), color='b', linestyle="solid", alpha=0.2)
    p23 = ax[1].plot(test_xs[1:], np.asarray(avg_sc_tr_loss), color='b', linestyle="solid", label='source_train')

    ax[0].set_title("Classification Error")
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('source error (%)')
    ax[0].set_ylim(0, 20)
    ax0_twin.set_ylabel("target error (%)")
    ax0_twin.set_ylim(0, 20)


    lns = p23 + p21
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc="upper left", framealpha=0.90)
    ax1_twin.legend(lns, labs, loc="upper left", framealpha=0.90)

    ax[1].set_title("Classification Loss")
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('source loss')
    ax[1].set_ylim(0, 2)
    ax1_twin.set_ylabel("target loss")
    ax1_twin.set_ylim(0, 2)

    plt.savefig('%s/classification_temp_plot.pdf' %(outf))
    plt.close()

def plot_saved_ss_stats(ss_train_path, ss_test_path, outf):
    
    ss_test_stats = torch.load(ss_test_path)
    ss_train_stats = torch.load(ss_train_path)

    ticks, labels, xs, source_test_error, target_test_error = parse_ss_test_stats(ss_test_stats)

    fig, ax = plt.subplots(ncols=2, figsize=(16,8))
    fig.tight_layout(pad=10.0)

    ax0_twin = ax[0].twinx()
    ax1_twin = ax[1].twinx()

    # plot val error for source and target data
    p0 = ax0_twin.plot(xs, np.asarray(target_test_error)*100, color='r', linestyle="dotted", label='target_val')
    p1 = ax[0].plot(xs, np.asarray(source_test_error)*100, color='r', linestyle="solid", label='source_val')
    plt.sca(ax[0])
    plt.xticks(ticks, labels)

    # p20 = ax1_twin.plot(xs, np.asarray(target_train_loss), color='b', linestyle="dotted", label='target_train')
    # p21 = ax[1].plot(xs, np.asarray(source_train_loss), color='b', linestyle="solid", label='source_train')


    test_xs = xs

    ticks, labels, xs, source_train_loss, source_train_error, target_train_loss, target_train_error = parse_ss_train_stats(ss_train_stats)

    avg_sc_tr_loss = []
    avg_sc_tr_err = []
    avg_tg_tr_loss = []
    avg_tg_tr_err = []
    prev_indx = 0
    for t in test_xs:
        if t == 0:
            continue
        avg_sc_tr_loss.append(np.mean(source_train_loss[prev_indx:t]))
        avg_sc_tr_err.append(np.mean(source_train_error[prev_indx:t]))

        avg_tg_tr_loss.append(np.mean(target_train_loss[prev_indx:t]))
        avg_tg_tr_err.append(np.mean(target_train_error[prev_indx:t]))

        prev_indx = t
    
    # plot source train error
    p2 = ax[0].plot(xs, np.asarray(source_train_error)*100, color='b', linestyle="solid",  alpha=0.2)
    p3 = ax[0].plot(test_xs[1:], np.asarray(avg_sc_tr_err)*100, color='b', linestyle="solid", label='source_train')

    # plot target train error
    p4 = ax[0].plot(xs, np.asarray(target_train_error)*100, color='b', linestyle="dotted",  alpha=0.2)
    p5 = ax[0].plot(test_xs[1:], np.asarray(avg_tg_tr_err)*100, color='b', linestyle="dotted", label='target_train')

    lns = p3 + p1 + p5 + p0
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc="upper left", framealpha=0.90)
    ax0_twin.legend(lns, labs, loc="upper left", framealpha=0.90)

    # plot loss graph
    plt.sca(ax[1])
    plt.xticks(ticks, labels)
    # plot source train loss
    p20 = ax[1].plot(xs, np.asarray(source_train_loss), color='b', linestyle="solid", alpha=0.2)
    p21 = ax[1].plot(test_xs[1:], np.asarray(avg_sc_tr_loss), color='b', linestyle="solid", label='source_train')
    # plot target train loss
    p22 = ax[1].plot(xs, np.asarray(target_train_loss), color='b', linestyle="dotted", alpha=0.2)
    p23 = ax[1].plot(test_xs[1:], np.asarray(avg_tg_tr_loss), color='b', linestyle="dotted", label='target_train')

    ax[0].set_title("SStask Error")
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('source error (%)')
    ax[0].set_ylim(0, 30)
    ax0_twin.set_ylabel("target error (%)")
    ax0_twin.set_ylim(0, 30)


    lns = p20 + p21 + p22 + p23
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc="upper left", framealpha=0.90)
    ax1_twin.legend(lns, labs, loc="upper left", framealpha=0.90)

    ax[1].set_title("SStask Loss")
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('source loss')
    ax[1].set_ylim(0, 0.6)
    ax1_twin.set_ylabel("target loss")
    ax1_twin.set_ylim(0, 0.6)


    plt.savefig('%s/ss_temp_plot.pdf' %(outf))
    plt.close()


if __name__ == "__main__":
    # folder = "/home/erik/phd/courses/deep learning/dl_project/output/mnist_mnistm_r"
    folder = "/home/erik/phd/courses/deep learning/dl_project/results/classification/mnist_mnistm_r/experiment_1"
    supervised_train_path = os.path.join(folder, "supervised_train_stats.pth")
    metrics_path = os.path.join(folder, "loss.pth")


    ss_train_path = os.path.join(folder, "ss_train_stats.pth")
    ss_test_path = os.path.join(folder, "ss_test_stats.pth")
    plot_saved_classification_stats(supervised_train_path, metrics_path, folder)
    plot_saved_ss_stats(ss_train_path, ss_test_path, folder)

