from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5, ))])

iris_dataset = MNIST('.')
print(MNIST)

# for key, value in zip(iris_dataset.keys(), iris_dataset.values()):
#     print("{}:\n{}\n".format(key, value))