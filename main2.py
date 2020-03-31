import argparse
import os
from pathlib import Path
from typing import NewType

import mxnet as mx
import torch
from mxnet import gluon, nd, sym
from mxnet.gluon import nn
from torchvision.utils import save_image

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
parser.add_argument('--hybrid', action='store_true',
                    help='hybridize the model and loss function')
parser.add_argument('--results', default='mxresults', help='results dir')
args = parser.parse_args()

args.cuda = not args.no_cuda and mx.context.num_gpus() > 0
args.results_path = Path(args.results)

os.makedirs(args.results, exist_ok=True)

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
    gluon.data.vision.MNIST(
        root='./data', train=True).transform_first(gluon.data.vision.transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs
)

test_loader = gluon.data.DataLoader(
    gluon.data.vision.MNIST(
        root='./data', train=False).transform_first(gluon.data.vision.transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs
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

    def encode(self, x):
        h1 = nd.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = nd.exp(0.5 * logvar)

        # shape = F.shape_array(std)
        # print(shape[1])
        # eps = F.random_normal(
        #     loc=0, scale=1, shape=None, ctx=self.ctx
        # )
        eps = nd.random.normal_like(data=std)
        return mu + eps * std

    def decode(self, z):
        h3 = nd.relu(self.fc3(z))
        return nd.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(nd.flatten(x))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class LossFunction(gluon.loss.Loss):

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super().__init__(weight, batch_axis, **kwargs)
        self.soft_zero = 1e-10

    def forward(self, recon_x, x, mu, logvar):
        # bce = self.bce_loss(recon_x, F.flatten(x))
        # # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # kld = -0.5 * F.mean(1 + logvar - F.power(mu, 2) - F.exp(logvar), axis=self._batch_axis, exclude=True)
        x = nd.flatten(x)
        kl = 0.5 * nd.sum(1 + logvar - nd.power(mu, 2) - nd.exp(logvar), axis=1)
        logloss = nd.sum(
            x * nd.log(recon_x + self.soft_zero) + (1 - x) * nd.log(1 - recon_x + self.soft_zero), axis=1
        )
        # import ipdb; ipdb.set_trace()
        return -logloss - kl


model = VAE()
model.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
loss_function = LossFunction()

if args.hybrid:
    model.hybridize()
    loss_function.hybridize()

trainer = gluon.Trainer(model.collect_params(), 'adam', dict(
    learning_rate=1e-3,
))


def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.as_in_context(ctx)

        with mx.autograd.record():
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        trainer.step(len(data))

        train_loss += loss.sum().asscalar()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader._dataset),
                       100. * batch_idx / len(train_loader),
                       loss.sum().asscalar() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader._dataset)))


def test(epoch):
    test_loss = 0

    for i, (data, _) in enumerate(test_loader):
        data = data.as_in_context(ctx)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data,
                                   mu, logvar).sum().asscalar()
        if i == 0:
            n = min(data.shape[0], 8)
            # comparison = torch.cat([data[:n],
            #                         recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            # save_image(comparison.cpu(),
            #             'results/reconstruction_' + str(epoch) + '.png', nrow=n)

            comparison = nd.concat(
                data[:n], recon_batch.reshape(
                    args.batch_size, 1, 28, 28)[:n], dim=0
            )
            save_ndarray_as_image(comparison.asnumpy(),
                                  args.results_path / ('reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader._dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def save_ndarray_as_image(array, filename, nrow=8):
    tensor = torch.from_numpy(array)
    save_image(tensor, filename, nrow=nrow)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')

        sample = nd.random.randn(64, 20).as_in_context(ctx)
        sample = model.decode(sample).asnumpy()
        save_ndarray_as_image(sample.reshape(64, 1, 28, 28),
                              args.results_path / ('sample_' + str(epoch) + '.png'))
