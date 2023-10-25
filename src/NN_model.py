import numpy as np
import pandas as pd
import torch
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from fastai.vision.all import *
from fastbook import *
from sklearn.model_selection import train_test_split
matplotlib.rc('image', cmap='Greys')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import torch
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


num_moms_arrive = 5
num_moms_service = 5

def compute_sum_error(valid_dl, model, return_vector, max_err=0.05, display_bad_images=False):
    with torch.no_grad():
        bad_cases = {}
        for ind, batch in tqdm(enumerate(valid_dl)):

            xb, yb = batch
            predictions = m(model(xb[:, :]))
            #             aa = (xb[:,0]*(torch.exp(xb[:,1]))).reshape((xb.shape[0],1))
            normalizing_const = 1 - yb[:, 0]  # aa.repeat(1,predictions.shape[1])
            predictions = predictions * normalizing_const.reshape((yb.shape[0], 1))
            curr_errors = torch.sum(torch.abs((predictions - yb[:, 1:])), axis=1)
            bad_dists_inds = (curr_errors > max_err).nonzero(as_tuple=True)[0]

            prob_0 = (1 - torch.sum(predictions[:, :], axis=1)).reshape(torch.sum(predictions[:, :], axis=1).shape[0],
                                                                        1)

            preds = torch.concat((prob_0, predictions), axis=1)
            if ind == 0:
                sum_err_tens = torch.sum(torch.abs((predictions - yb[:, 1:])), axis=1)
            else:
                sum_err_tens = torch.cat((sum_err_tens, curr_errors), axis=0)
    if return_vector:
        return (torch.mean(sum_err_tens), sum_err_tens, yb, preds)
    else:
        return torch.mean(sum_err_tens)


def compute_preds(valid_dl, model, max_err=0.05, display_bad_images=False):
    import torch
    import torch.nn as nn

    m = nn.Softmax(dim=1)

    pred_list = []

    with torch.no_grad():

        for ind, batch in tqdm(enumerate(valid_dl)):

            xb, yb = batch
            predictions = m(model(xb[:, :]))
            normalizing_const = 1/torch.exp(xb[:,0])
            predictions = predictions * normalizing_const.reshape((yb.shape[0], 1))
            prob_0 = (1 - torch.sum(predictions[:, :], axis=1)).reshape(torch.sum(predictions[:, :], axis=1).shape[0],
                                                                        1)

            preds = torch.concat((prob_0, predictions), axis=1)
            if ind == 0:
                preds_torch = preds
            else:
                preds_torch = torch.cat((preds_torch, preds), axis=0)

            pred_list.append(preds)
        return (preds_torch, yb)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



def main(args):


    import torch
    import torch.nn as nn

    ## Insert your inter-arrival and service time moments here:

    ### Example:

    inter_arrival_moms = torch.tensor([3.7960, 9.3644, 36.2736, 192.3857, 1290.4828])
    service_moments = torch.tensor([3, 5.9023, 17.2487, 67.1530, 327.2156])


    m = nn.Softmax(dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # code made in pytorch3.ipynb with comments
    class Net(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(num_moms_arrive + num_moms_service - 1, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 100)
            self.fc4 = nn.Linear(100, 150)
            self.fc5 = nn.Linear(150, 200)
            self.fc6 = nn.Linear(200, 200)
            self.fc7 = nn.Linear(200, 350)
            self.fc8 = nn.Linear(350, 499)

        def forward(self, x):

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            x = self.fc8(x)
            return x

    file = os.listdir(args.model_path)[0]
    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=torch.device('cpu')))


    service_1_mom = service_moments[0]
    inter_arrival_moms = inter_arrival_moms / service_1_mom
    service_moments = service_moments / service_1_mom
    input_moms = torch.log(torch.cat((inter_arrival_moms, service_moments[1:]), axis=0))
    input_moms = input_moms.reshape((1, input_moms.shape[0]))

    with torch.no_grad():
        predictions = m(net(input_moms))
        normalizing_const = 1 / torch.exp(input_moms[:, 0])
        predictions = predictions * normalizing_const.reshape((input_moms.shape[0], 1))
        prob_0 = (1 - torch.sum(predictions[:, :], axis=1)).reshape(torch.sum(predictions[:, :], axis=1).shape[0], 1)
        preds = torch.concat((prob_0, predictions), axis=1)


    ## Number of values to present in the starionay queue lenght distribution.
    max_probs = 30
    labels = np.arange(max_probs)
    true = preds[0, :max_probs]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(7.5, 4))

    rects2 = ax.bar(x, true, width, label='NN Prediction')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('PMF', fontsize=18)
    ax.set_xlabel('Number of customers in the system', fontsize=18)
    ax.set_title('Steady-state anaylsis',
                 fontsize=21)  # 'Queue-length comparison: '+ r'$rho$' + ' = ' +str(1-yb_arr[ind_, 0])[:3])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=15)

    fig.tight_layout()
    plt.show()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='the path of the NN model', default='.\data')
    parser.add_argument('--num_moms', type=int, help='number of ph folders', default=20)
    parser.add_argument('--batch_size', type=int, help='number of ph examples in one file', default=1)
    parser.add_argument('--ph_size_max', type=int, help='number of ph folders', default = 1000)
    parser.add_argument('--data_path', type=str, help='where to save the file', default=r'C:\Users\user\workspace\data\deep_gg1')
    parser.add_argument('--max_utilization', type=float, help='What is the largest possible utilization', default = 0.999)
    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    main(args)