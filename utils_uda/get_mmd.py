import torch
import numpy as np

def get_features(dataloader, model):
	all_outputs = []
	all_labels = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		inputs, labels = inputs.cuda(), labels.cuda()
		with torch.no_grad():
			outputs = model(inputs)
			all_outputs.append(outputs.cpu())
			all_labels.append(labels.cpu())

	# torch.save(all_outputs, "/home/erik/phd/courses/deep learning/dl_project/results/classification/dummy/features.pth")
	return torch.cat(all_outputs), torch.cat(all_labels)

def get_centroid(dataloader, model):
	# all_outputs = []
	# for batch_idx, (inputs, labels) in enumerate(dataloader):
	# 	inputs, labels = inputs.cuda(), labels.cuda()
	# 	with torch.no_grad():
	# 		outputs = model(inputs)
	# 		all_outputs.append(outputs.cpu())
	all_outputs, all_labels = get_features(dataloader, model)
	return torch.mean(all_outputs, dim=0), all_outputs, all_labels

def get_mmd(loader_1, loader_2, model):
	model.eval()
	centroid_1, output_loader1, labels_loader1 = get_centroid(loader_1, model)
	centroid_2, output_loader2, labels_loader2 = get_centroid(loader_2, model)
	model.train()
	return torch.dist(centroid_1, centroid_2, 2).item(), output_loader1, output_loader2, labels_loader1, labels_loader2

def mmd_select_naive(mmd):
	return np.argmin(mmd)

def mmd_select_scale(mmd, sce):
	sce = np.asarray(sce)
	mmd = np.asarray(mmd)
	scl = np.min(sce) / np.min(mmd)
	return np.argmin(sce + mmd * scl)