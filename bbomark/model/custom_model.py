import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from bbomark.constants import VISIBLE_TO_OPT
from bbomark.core import TestFunction, AbstractBaseModel

METRICS = tuple(sorted(['acc', 'loss']))

device = 'cpu'



# 网络结构定义
class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super().__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model定义
class BaseModel(AbstractBaseModel):

    def __init__(self, **params):
        super().__init__(**params)
        self.net = LeNet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.params['lr'],
            momentum=self.params['momentum'])



    def fit(self, cifar_train):
        self.trainloader = torch.utils.data.DataLoader(
            cifar_train, batch_size=self.params['batch_size'], shuffle=True
        )

        for epoch in range(self.params['epoch']):
            # 我们的dataloader派上了用场
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # 注意需要复制到GPU
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # loss_record += loss.item()
                # if i % 100 == 99:
                #     print('[Epoch %d, Batch %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, loss100 / 100))
                #     loss_record = 0.0

    def get_score(self, cifar_test, metric):
        '''
        score 指标 越大越好
        '''

        self.testloader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True)
        correct = 0
        total = 0
        loss = 0
        # 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # 预测
                outputs = self.net(images)
                loss += self.criterion(outputs, labels).item() * labels.shape[0]
                # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if metric == 'loss':
            return -loss / len(cifar_test)
        # print('Accuracy of the network on the 10000 test images: %d %%' % (
        #         100 * correct / total))
        elif metric == 'acc':
            return 100 * correct / total


class Model(TestFunction):
    '''
    自定义model的类名必须为Model，且继承自TestFunction类
    '''
    objective_names = (VISIBLE_TO_OPT, "generalization")

    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):
        """Build class that wraps sklearn classifier/regressor CV score for use as an objective function.

        Parameters
        ----------
        model : str
            Which classifier to use, must be key in `MODELS_CLF` or `MODELS_REG` dict depending on if dataset is
            classification or regression.
        dataset : str
            Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
        metric : str
            Which sklearn scoring metric to use, in `SCORERS_CLF` list or `SCORERS_REG` dict depending on if dataset is
            classification or regression.
        shuffle_seed : int
            Random seed to use when splitting the data into train and validation in the cross-validation splits. This
            is needed in order to keep the split constant across calls. Otherwise there would be extra noise in the
            objective function for varying splits.
        data_root : str
            Root directory to look for all custom csv files.
        """
        super().__init__()

        self.train_data, self.test_data = self.load_data(dataset, data_root=data_root)

        api_config = self.load_api_config()
        base_model = self.load_base_model()

        # New members for model
        self.base_model = base_model
        self.api_config = api_config


        assert metric in METRICS, "Unknown metric %s" % metric
        self.metric = metric


    def load_api_config(self):
        return {
            'lr': {'type': 'real', 'space': 'log', 'range': (1e-6, 0.1)},
            'momentum':{'type': 'real', 'space': 'linear', 'range': (0.8, 0.999)},
            'epoch': {'type': 'int', 'space': 'linear', 'range': (1, 50)},
            'batch_size': {'type': 'ord', 'space': 'linear', 'values': list(range(4, 129, 4))},
        }

    def load_data(self, dataset, data_root=None):
        if data_root is None:
            data_root = './data'
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar_train = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                                   download=True, transform=transform)
        cifar_test = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                                  transform=transform)

        return cifar_train, cifar_test

    def evaluate(self, params):
        """Evaluate the sklearn CV objective at a particular parameter setting.

        Parameters
        ----------
        params : dict(str, object)
            The varying (non-fixed) parameter dict to the sklearn model.

        Returns
        -------
        cv_loss : float
            Average loss over CV splits for sklearn model when tested using the settings in params.
        """
        params = dict(params)  # copy to avoid modification of original
        # params.update(self.fixed_params)  # add in fixed params

        # now build the skl object
        model = self.base_model(**params)

        model.fit(self.train_data)
        score = model.get_score(self.test_data, self.metric)
        # Do the x-val, ignore user warn since we expect BO to try weird stuff

        cv_score = np.mean(score)


        # get_scorer makes everything a score not a loss, so we need to negate to get the loss back
        cv_loss = -cv_score
        assert np.isfinite(cv_loss), "loss not even finite"


        generalization_loss = None
        # For now, score with same objective. We can later add generalization error
        return cv_loss, generalization_loss

    def load_base_model(self):
        return BaseModel