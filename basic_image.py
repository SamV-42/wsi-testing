import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.conv1 = nn.Conv2d(1,6,3)
		self.conv2 = nn.Conv2d(6,16,3)

		self.fc1 = nn.Linear(16*6*6,120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_feat(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_feat(self, x):
		size = x.size()[1:]
		num_feat = 1
		for s in size:
			num_feat *= s
		return num_feat

net = Net()
criterion = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=0.01)


# training loop:
for i in range(1,100):	

	# generate random in/out data
	input = torch.randn(1, 1, 32, 32)
	target = torch.randn(10).view(1,-1)

	output = net(input)	# run input through the net
	loss = criterion(output, target) # calculate overall loss

	net.zero_grad()		# reset gradients
	loss.backward()		# calculate gradients
	optim.step()		# update weights

	print(output)
