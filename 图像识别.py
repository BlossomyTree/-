import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

to_normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
to_tensor = transforms.ToTensor()
trans_compose = transforms.Compose([to_tensor])

train_data = datasets.CIFAR10(r"D:\python\ml\pytorch学习\数据", train=True, download=True, transform=trans_compose)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=True)


class Hecx(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(64, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 64)
		)
		self.logit_out = nn.Sequential(
			nn.Linear(3072, 10),
			nn.Softmax(1)
		)

	def forward(self, input_x):
		input_x = self.flatten(input_x)
		input_x = input_x.T
		input_x = self.linear_relu_stack(input_x)
		input_x = input_x.T
		input_x = self.logit_out(input_x)

		return input_x


hecx = Hecx()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(hecx.parameters(), lr=0.01)

for epoch in range(20):
	for data in train_loader:
		images, targets = data
		output_y = hecx.forward(images)
	
		result_loss = loss(output_y, targets)
		optimizer.zero_grad()
		result_loss.backward()
		optimizer.step()

	print(epoch)

torch.save(hecx.state_dict(), r"D:\a.pth")
