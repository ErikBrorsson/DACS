import torch
import torch.utils.data

class DsetNoLabel(torch.utils.data.Dataset):
	# Make sure that your dataset actually returns many elements!

	def __init__(self, dset):
		self.dset = dset

	def __getitem__(self, index):
        # assumes that the first element in each "batch" is the image...
        # i.e., drop all other information, including the label
		return self.dset[index][0]

	def __len__(self):
		return len(self.dset)