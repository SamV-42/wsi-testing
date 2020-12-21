import torch
import torchvision.transforms as transforms
import torchvision

# Load the data from CIFAR and normalize it from 0,1 to -1,1.
# note that num_workers is set to 0 because it broke things for me as 2

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Display functions for convenience' sake

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()


# The NN from the other file, but with 3 input channels

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)

		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
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



import sys
PATH = './cifar_net.pth'
TRAIN = not ('--no-train' in sys.argv)

if TRAIN:
	net = Net()
	criterion = nn.CrossEntropyLoss()
	optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# training loop:
	for epoch in range(2):

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data			# get data, [inputs, labels]

			net.zero_grad()				# reset gradients
			outputs = net(inputs)			# run input through the net
			loss = criterion(outputs, labels)	# calculate overall loss
			loss.backward()				# calculate gradients
			optim.step()				# update weights

			running_loss += loss.item()
			if i % 2000 == 1999: 	# print every 2k batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print("Training complete")
	torch.save(net.state_dict(), PATH)


EXAMPLE = not ('--no-example' in sys.argv)
if EXAMPLE:
	dataiter = iter(testloader)
	images, labels = dataiter.next()

	net = Net()
	net.load_state_dict(torch.load(PATH))
	outputs = net(images)
	_, predicted = torch.max(outputs, 1)

	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
	imshow(torchvision.utils.make_grid(images))



TEST = not ('--no-test' in sys.argv)
if TEST:
	net = Net()
	net.load_state_dict(torch.load(PATH))

	correct = 0
	total = 0 
	with torch.no_grad():	# disables gradient descent within the block
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
	for i in range(10):
		print('Accuracy of %5s : %2d %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))
