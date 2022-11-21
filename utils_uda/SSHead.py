from torch import nn

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

# def extractor_from_layer3(net):
# 	layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu]
# 	return nn.Sequential(*layers)

def linear_on_layer3(classes, width, pool):
	layers = [nn.AvgPool2d(pool), ViewFlatten(), nn.Linear(512*4*8, classes)] # 64*width
	# layers = [ViewFlatten(), nn.Linear(512*34*62, classes)] # 64*width
	return nn.Sequential(*layers)

def linear_on_layer3_square_img(classes, width, pool):
    # TODO implement 1x1 conv to reduce the number of channels from 2048 to ~256
	# 2048, 17x17
	layers = [nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1), nn.AvgPool2d(pool), ViewFlatten(), nn.Linear(128*17*17, classes)] # 64*width
	# layers = [ViewFlatten(), nn.Linear(512*34*62, classes)] # 64*width
	return nn.Sequential(*layers)


def last_layer_head(classes, width, pool):
    # TODO implement 1x1 conv to reduce the number of channels from 2048 to ~256
	# 2048, 17x17
	layers = [nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=1), nn.AvgPool2d(pool), ViewFlatten(), nn.Linear(128*16*16, classes)] # 64*width
	# layers = [ViewFlatten(), nn.Linear(512*34*62, classes)] # 64*width
	return nn.Sequential(*layers)

def linear_on_layer3_square_img_small(classes, width, pool):
	layers = [nn.AvgPool2d(pool), ViewFlatten(), nn.Linear(512*2*2, classes)] # 64*width
	# layers = [ViewFlatten(), nn.Linear(512*34*62, classes)] # 64*width
	return nn.Sequential(*layers)
