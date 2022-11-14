import torch
import numpy as np

def get_features(dataloader, model):
	all_outputs = []
	for batch_idx, (inputs, labels) in enumerate(dataloader):
		inputs, labels = inputs.cuda(), labels.cuda()
		with torch.no_grad():
			outputs = model(inputs)
			all_outputs.append(outputs.cpu())
	return all_outputs
