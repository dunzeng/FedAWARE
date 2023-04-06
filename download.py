import torchvision
import torchvision.transforms as transforms

test_data = torchvision.datasets.CIFAR10(root="./datasets/cifar10/",
                                        download=True,
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))
                                        ]))