import mxnet as mx
import argparse
from mxnet.gluon import nn
from mxnet import gluon, ndarray
from typing import NewType

Fn = NewType('Fn', ndarray)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and mx.context.num_gpus() > 0

mx.random.seed(args.seed)

ctx = mx.gpu() if args.cuda else mx.cpu()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

train_loader = gluon.data.DataLoader(
    gluon.data.vision.MNIST(root='./data', train=True,
                            transform=gluon.data.vision.transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs
)

train_loader = gluon.data.DataLoader(
    gluon.data.vision.MNIST(root='./data', train=False,
                            transform=gluon.data.vision.transforms.ToTensor()),
    batch_size=args.batch_size, **kwargs
)


class VAE(nn.HybridBlock):

    def __init__(self, activation='relu', **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(400)
            self.fc21 = nn.Dense(20)
            self.fc22 = nn.Dense(20)
            self.fc3 = nn.Dense(400)
            self.fc4 = nn.Dense(784)

    @property
    def ctx(self):
        return self.collect_params().list_ctx()

    def encode(self, F: Fn, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, F: Fn, mu, logvar):
        std = F.exp(0.5 * logvar)
        eps = F.random_normal(
            loc=0, scale=1, shape=F.shape_array(std), ctx=self.ctx
        )
        return mu + eps*std

    def decode(self, F: Fn, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def hybrid_forward(self, F: Fn, x):
        mu, logvar = self.encode(F, F.flatten(x))
        z = self.reparameterize(F, mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
model.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
model.hybridize()

optimizer = gluon.Trainer(model.collect_params(), 'adam', dict(
    learning_rate=1e-3,
))


class LossFunction(nn.HybridBlock):

    pass
