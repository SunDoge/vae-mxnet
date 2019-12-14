import mxnet as mx
import argparse
from mxnet.gluon import nn
from mxnet import gluon, ndarray, symbol
from typing import NewType

Sym = NewType('Sym', symbol)

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
    gluon.data.vision.MNIST(
        root='./data', train=True).transform_first(gluon.data.vision.transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs
)

test_loader = gluon.data.DataLoader(
    gluon.data.vision.MNIST(
        root='./data', train=False).transform_first(gluon.data.vision.transforms.ToTensor()),
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

    def encode(self, F: Sym, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, F: Sym, mu, logvar):
        std = F.exp(0.5 * logvar)

        # shape = F.shape_array(std)
        # print(shape[1])
        # eps = F.random_normal(
        #     loc=0, scale=1, shape=None, ctx=self.ctx
        # )
        eps = F.random.normal_like(data=std)
        return mu + eps*std

    def decode(self, F: Sym, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def hybrid_forward(self, F: Sym, x):
        mu, logvar = self.encode(F, F.flatten(x))
        z = self.reparameterize(F, mu, logvar)
        return self.decode(F, z), mu, logvar


class LossFunction(gluon.loss.Loss):

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super().__init__(weight, batch_axis, **kwargs)
        self.bce_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(
            from_sigmoid=True)

    def hybrid_forward(self, F: Sym, recon_x, x, mu, logvar):
        bce = F.sum(self.bce_loss(recon_x, x))
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = -0.5 * F.sum(
            data=1 + logvar - F.power(mu, 2) - F.exp(logvar)
        )

        return bce + kld


model = VAE()
model.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)
model.hybridize()

trainer = gluon.Trainer(model.collect_params(), 'adam', dict(
    learning_rate=1e-3,
))

loss_function = LossFunction()


def train(epoch):

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.as_in_context(ctx)

        with mx.autograd.record():
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        trainer.step(len(data))

        train_loss += loss.asscalar()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader._dataset),
                100. * batch_idx / len(train_loader),
                loss.asscalar() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader._dataset)))


def test(epoch):

    test_loss = 0

    for i, (data, _) in enumerate(test_loader):
        data = data.as_in_context(ctx)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).asscalar()
        if i == 0:
            n = min(data.shape[0], 8)
            # comparison = torch.cat([data[:n],
            #                         recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            # save_image(comparison.cpu(),
            #             'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader._dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
