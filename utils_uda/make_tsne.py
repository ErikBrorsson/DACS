
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import os



def get_tsne_features(features):
    """Expecting features of size (n x d) where n is the number of samples and d is the dimensionality of each sample's feature.
    
    returns:
        tsne features of size (n x 2)
    """
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)

    # features = torch.cat(features)
    features_flat = features.view(features.shape[0], -1)

    tsne_features = tsne.fit_transform(features_flat)

    return tsne_features

def plot_tsne(tsne_source_features, tsne_target_features, outf, source_labels, target_labels):

    fig, ax = plt.subplots(ncols=2, figsize=(16,8))
    fig.tight_layout(pad=10.0)

    # plt.figure(figsize=(16, 8))

    ax[0].set_xlabel("tsne feature 1", fontsize=18)
    ax[0].set_ylabel("tsne feature 2", fontsize=18)
    ax[0].set_title("tsne plot of resnet features", fontsize=24)

    ax[1].set_xlabel("tsne feature 1")
    ax[1].set_ylabel("tsne feature 2")
    ax[1].set_title("tsne plot of resnet features")

    for i in range(10):
        source_feautres_i = tsne_source_features[source_labels == i]
        target_feautres_i = tsne_target_features[target_labels == i]
        ax[0].scatter(source_feautres_i[:, 0], source_feautres_i[:, 1], label="source label={}".format(i), alpha=0.3)#, s=100)
        ax[1].scatter(target_feautres_i[:, 0], target_feautres_i[:, 1], label="target label={}".format(i), alpha=0.3)#, s=100)


    ax[0].scatter(tsne_target_features[:, 0], tsne_target_features[:, 1], label="target data", color="black", alpha=0.2, marker="*")
    ax[0].legend(prop={"size":5})
    ax[1].scatter(tsne_source_features[:, 0], tsne_source_features[:, 1], label="source data", color="black", alpha=0.2, marker="*")
    ax[1].legend(prop={"size":5})


    plt.savefig('%s/tsne_plot.pdf' %(outf))
    plt.close()

def compute_tsne_features(folder: str, source_features: str, target_features: str):

    source_features = torch.load(os.path.join(folder, source_features))
    target_features = torch.load(os.path.join(folder, target_features))

    all_features = torch.cat([source_features, target_features])
    all_tsne_features = get_tsne_features(all_features)

    torch.save(all_tsne_features, os.path.join(folder, "tsne_features.pth"))

def create_plot(folder: str, source_features: str, target_features: str, tsne_features: str,
        source_labels: str, target_labels: str):

    # laod source and target features to now the shape of tsne features
    source_features = torch.load(os.path.join(folder, source_features))
    target_features = torch.load(os.path.join(folder, target_features))

    # load all tsne features and split into source and target
    tsne_features = torch.load(os.path.join(folder, tsne_features))
    source_tsne = tsne_features[0:source_features.shape[0],:]
    target_tsne = tsne_features[source_features.shape[0]:,:]

    # load labels
    source_labels = torch.load(os.path.join(folder, source_labels))
    target_labels = torch.load(os.path.join(folder, target_labels))


    plot_tsne(source_tsne, target_tsne, folder, source_labels, target_labels)


if __name__ == "__main__":
    folder = "/home/erik/phd/courses/deep learning/dl_project/results/classification/dummy/experiment_15"
    folder = "/home/erik/phd/courses/deep learning/dl_project/results/classification/mnist_mnistm_r/experiment_3"
    folder = "/home/erik/phd/courses/deep learning/dl_project/results/classification/mnist_mnistm/experiment_4"

    # compute tsne features of saved source and target features
    compute_tsne_features(folder, "source_features.pth", "target_features.pth")

    # create plot from saved tsne features
    create_plot(folder, "source_features.pth", "target_features.pth", "tsne_features.pth", "source_labels.pth", "target_labels.pth")