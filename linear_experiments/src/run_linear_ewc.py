# %%
import torch
from scipy.linalg import null_space, orth
from tqdm import tqdm
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import os
import csv

import load_split_mnist
import simple_fisher

# %%

# parse lambdas from command line
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_filter", type=float, default=1.0)
parser.add_argument("--lambda_anti_filter", type=float, default=1.0)
parser.add_argument("--ewc_lambda", type=float, default=0.0)
parser.add_argument("--logdir", type=str, default="../results")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

# %%
SEED = 42
NHIDDEN = 11
EPOCHS = 30
BATCHSIZE = 16
LR = 5e-4
MOMENTUM = 0.0
LAMBDA_FILTER = args.lambda_filter
LAMBDA_ANTI_FILTER = args.lambda_anti_filter
LOGDIR = args.logdir
SHOW = args.show
EWC_LAMBDA = args.ewc_lambda

# fix all seeds for reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GradFilterMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, filter, anti_filter):
        ctx.save_for_backward(x, w, filter, anti_filter)
        y = x @ w.T
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, filter, anti_filter = ctx.saved_tensors

        # grad_input = grad_output @ filter @ weight  # TODO check if this is correct!! Not needed for single layer
        grad_input = grad_output @ weight  # unfiltered
        grad_weight = grad_output.t().mm(input)
        grad_weight_filter = filter @ grad_weight
        grad_weight_anti_filter = anti_filter @ grad_weight
        grad_weight = (
            model.lambda_filter * grad_weight_filter
            + model.lambda_anti_filter * grad_weight_anti_filter
        )
        return (
            grad_input,
            grad_weight,
            None,
            None,
        )  # filter does get updated with gradient descent


class FilteredLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, filter, anti_filter):
        return GradFilterMatmul.apply(x, self.weight, filter, anti_filter)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FilteredLinear(784, NHIDDEN)
        self.register_buffer("filter", torch.eye(NHIDDEN))
        self.register_buffer("anti_filter", torch.eye(NHIDDEN))

        self.readout1 = torch.nn.Linear(NHIDDEN, 5, bias=False)
        self.readout2 = torch.nn.Linear(NHIDDEN, 5, bias=False)
        self.readouts = torch.nn.ModuleList([self.readout1, self.readout2])

        self.lambda_filter = 1.0
        self.lambda_anti_filter = 1.0

    def forward(self, x, t):
        # flatten
        x = x.view(-1, 784)
        x = self.fc1(x, self.filter, self.anti_filter)
        x = self.readouts[t](x)
        return x

    def get_embedding(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x, self.filter, self.anti_filter)
        return x

    @torch.no_grad()
    def update_filter(self, readout_idx=[]):
        print("updating filter")
        if len(readout_idx) == 0:
            print("no readouts selected")
            return  # nothing to see here, move along

        # select and concatenate readouts
        ws = (
            torch.concat([self.readouts[i].weight for i in readout_idx])
            .detach()
            .numpy()
        )
        # cast to double precision for better inverses
        ws = ws.astype(np.float64)
        print("combined readout shape: ", ws.shape)
        # get orthonormal basis
        C = orth(ws.T)
        print("orthonormal basis 'C' shape: ", C.shape)
        CC_T = C @ C.T
        print("CC_T shape: ", CC_T.shape)
        # get null space
        N = null_space(C.T)
        anti_N = orth(C)
        print("null space 'N' shape: ", N.shape)
        NN_T = N @ N.T
        anti_filter = anti_N @ anti_N.T
        print("NN_T shape: ", NN_T.shape)
        # cast back to filter type tensor
        self.filter = torch.from_numpy(NN_T).type_as(self.filter)
        self.anti_filter = torch.from_numpy(anti_filter).type_as(self.filter)
        print("updated filter.")
        print("filter shape: ", self.filter.shape)


# %%


def train_step(model, optim, criterion, x, y, t, regularizer):
    model.train()
    optim.zero_grad()
    y_hat = model(x, t)
    loss = criterion(y_hat, y)
    reg_loss = regularizer.compute_loss(model.fc1.weight)
    loss += reg_loss
    loss.backward()
    optim.step()
    return loss.item()


def validate(model, dataloader, t):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_hat = model(x, t)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total


def train_task(
    model, optim, criterion, epochs, trainloader, testloaders, task, regularizer
):
    log = {"phase": [], "epoch": [], "task": [], "acc": []}

    for i in range(epochs):
        print(f"Epoch {i}")
        for x, y in tqdm(trainloader):
            loss = train_step(model, optim, criterion, x, y, task, regularizer)
        print("validating ...")
        for t, testloader in enumerate(testloaders):
            acc = validate(model, testloader, t)
            print("task", t, "acc", acc)
            log["phase"].append(task)
            log["epoch"].append(i + task * EPOCHS)
            log["task"].append(t)
            log["acc"].append(acc)
        print()
    return log


def embed_dataset(model, dataloader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            embedding = model.get_embedding(x)
            embeddings.append(embedding)
            labels.append(y)
    model.train()
    return torch.cat(embeddings), torch.cat(labels)


# %%
class EWC_Regularizer:
    def __init__(self, old_weights, info_mat, lambda_ewc):
        self.old_weights = old_weights
        self.info_mat = info_mat
        self.lambda_ewc = lambda_ewc

    def compute_loss(self, new_weights):
        return (
            self.lambda_ewc * self.info_mat * (new_weights - self.old_weights) ** 2
        ).sum()


class ZeroRegularizer:
    def __init__(self):
        pass

    def compute_loss(self, new_weights):
        return 0


# %%
# load data
train, test = load_split_mnist.load()

train_t1, train_t2 = train
test_t1, test_t2 = test

# dataloaders
train_loader_t1 = torch.utils.data.DataLoader(
    train_t1, batch_size=BATCHSIZE, shuffle=True
)
train_loader_t2 = torch.utils.data.DataLoader(
    train_t2, batch_size=BATCHSIZE, shuffle=True
)
test_loader_t1 = torch.utils.data.DataLoader(
    test_t1, batch_size=BATCHSIZE, shuffle=False
)  # test data must not be shuffled!!! Otherwise embeddings are not paired
test_loader_t2 = torch.utils.data.DataLoader(
    test_t2, batch_size=BATCHSIZE, shuffle=False
)

# %%

# create model
model = Model()
optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
criterion = torch.nn.CrossEntropyLoss()

# %%

# train on first task, no filter
regularizer = ZeroRegularizer()
acc_t0 = train_task(
    model,
    optim,
    criterion,
    EPOCHS,
    train_loader_t1,
    [test_loader_t1, test_loader_t2],
    0,
    regularizer,
)

# %%
# set lambdas, restricting learning trajectory for second task
model.lambda_filter = LAMBDA_FILTER
model.lambda_anti_filter = LAMBDA_ANTI_FILTER
embeddings_t1_p1, _ = embed_dataset(model, test_loader_t1)

# update filter
model.update_filter([0])
fisher_info = simple_fisher.compute_diagonal_fisher(
    model, model.fc1, train_loader_t1, criterion, 0
)
print(fisher_info)
ewc_reg = EWC_Regularizer(
    deepcopy(model.fc1.weight.detach()), fisher_info, lambda_ewc=EWC_LAMBDA
)

# %%
# train on second task, with filter
acc_t1 = train_task(
    model,
    optim,
    criterion,
    EPOCHS,
    train_loader_t2,
    [test_loader_t1, test_loader_t2],
    1,
    ewc_reg,
)
embeddings_t1_p2, _ = embed_dataset(model, test_loader_t1)

# %%

# concate dicts
for k in acc_t1.keys():
    acc_t0[k].extend(acc_t1[k])

# %%

frame = pd.DataFrame(acc_t0)
plt.figure()
sns.lineplot(data=frame, x="epoch", y="acc", hue="task")
# scatter
sns.scatterplot(data=frame, x="epoch", y="acc", hue="task", legend=False)
# draw vertical line at task boundary
plt.axvline(x=EPOCHS - 0.5, color="black", linestyle="--")
# fix scale between .2 and .9 on y axis
plt.ylim(0, 0.9)

if SHOW:
    plt.show()

# %%

# get performance on second task at last epoch
last_epoch = np.max(frame["epoch"])
plasticity = frame.loc[
    (frame["epoch"] == last_epoch) & (frame["task"] == 1), "acc"
].values[0]
stability = frame.loc[
    (frame["epoch"] == last_epoch) & (frame["task"] == 0), "acc"
].values[0]

w_old = ewc_reg.old_weights.detach().numpy()
w_new = model.fc1.weight.detach().numpy()

w_old_filter = model.filter.numpy() @ w_old
w_old_anti_filter = model.anti_filter.numpy() @ w_old
w_new_filter = model.filter.numpy() @ w_new
w_new_anti_filter = model.anti_filter.numpy() @ w_new

# p norm 1 and 2 of difference between old and new weights
p1 = np.linalg.norm(w_old - w_new, ord=1)
p2 = np.linalg.norm(w_old - w_new, ord=2)

# range and nullspace distances
p1_null = np.linalg.norm(w_old_filter - w_new_filter, ord=1)
p2_null = np.linalg.norm(w_old_filter - w_new_filter, ord=2)
p1_range = np.linalg.norm(w_old_anti_filter - w_new_anti_filter, ord=1)
p2_range = np.linalg.norm(w_old_anti_filter - w_new_anti_filter, ord=2)


embeddings_old = embeddings_t1_p1.detach().numpy()
embeddings_new = embeddings_t1_p2.detach().numpy()

# project embeddings
embeddings_old_filter = embeddings_old @ model.filter.numpy().T
embeddings_old_anti_filter = embeddings_old @ model.anti_filter.numpy().T
embeddings_new_filter = embeddings_new @ model.filter.numpy().T
embeddings_new_anti_filter = embeddings_new @ model.anti_filter.numpy().T


# %%

p1_embedding = np.mean(np.linalg.norm(embeddings_old - embeddings_new, ord=1, axis=1))
p2_embedding = np.mean(np.linalg.norm(embeddings_old - embeddings_new, ord=2, axis=1))

p1_embedding_null = np.mean(
    np.linalg.norm(embeddings_old_filter - embeddings_new_filter, ord=1, axis=1)
)
p2_embedding_null = np.mean(
    np.linalg.norm(embeddings_old_filter - embeddings_new_filter, ord=2, axis=1)
)
p1_embedding_range = np.mean(
    np.linalg.norm(
        embeddings_old_anti_filter - embeddings_new_anti_filter, ord=1, axis=1
    )
)
p2_embedding_range = np.mean(
    np.linalg.norm(
        embeddings_old_anti_filter - embeddings_new_anti_filter, ord=2, axis=1
    )
)

# %%
# create results folder if it does not exist
if not os.path.exists(f"../results/{LOGDIR}"):
    os.makedirs(f"../results/{LOGDIR}")

# create csv and dump results
with open(
    f"../results/{LOGDIR}/results_{EWC_LAMBDA}_{LAMBDA_FILTER}_{LAMBDA_ANTI_FILTER}.csv",
    "w",
    newline="",
) as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "lambda_filter",
            "lambda_anti_filter",
            "lambda_ewc",
            "plasticity",
            "stability",
            "p1",
            "p2",
            "p1_null",
            "p2_null",
            "p1_range",
            "p2_range",
            "p1_embedding",
            "p2_embedding",
            "p1_embedding_null",
            "p2_embedding_null",
            "p1_embedding_range",
            "p2_embedding_range",
        ]
    )
    writer.writerow(
        [
            LAMBDA_FILTER,
            LAMBDA_ANTI_FILTER,
            EWC_LAMBDA,
            plasticity,
            stability,
            p1,
            p2,
            p1_null,
            p2_null,
            p1_range,
            p2_range,
            p1_embedding,
            p2_embedding,
            p1_embedding_null,
            p2_embedding_null,
            p1_embedding_range,
            p2_embedding_range,
        ]
    )
