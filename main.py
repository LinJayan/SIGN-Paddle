# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataset import SIGNDataset, random_split, collate_fn
from net import SIGN
import argparse
from pgl.utils.data import Dataloader

from sklearn.model_selection import train_test_split
from train import train


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe', help='which dataset to use')
parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
parser.add_argument('--l0_weight', type=float, default=0.001, help='weight of the l2 regularization term') # 0.001  0.0002
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of the l2 regularization term') # 0.001  0.0002
parser.add_argument('--lr', type=float, default=0.05, help='learning rate') # 0.05  0.0015
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=500, help='the number of epochs')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), \
                            zeta (interval_min) and gama (interval_max).")
parser.add_argument('--hidden_layer', type=int, default=32, help='neural hidden layer')
parser.add_argument('--pred_edges', type=int, default=1, help='!=0: use edges in dataset, 0: predict edges \
                                                                using L_0')
parser.add_argument('--random_seed', type=int, default=2019, help='size of common item be counted')
parser.add_argument('--device', type=str, default='cpu', help='which dataset to use')
# parser.add_argument('--use_cuda',  action='store_true')
args = parser.parse_args()

signdataset = SIGNDataset("./data/", args.dataset, pred_edges=args.pred_edges)

data_num = signdataset.num_graph
num_feature = signdataset.num_feature

train_ds,val_ds,test_ds = random_split(signdataset)

train_loader = Dataloader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1,
                collate_fn=collate_fn)

test_loader = Dataloader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn)


val_loader = Dataloader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn)



print(f"""
datast: {args.dataset}
vector dim: {args.dim}
batch_size: {args.batch_size}
lr: {args.lr}
""")

datainfo = [train_loader, val_loader, test_loader, num_feature]
train(args, datainfo, [len(train_ds), len(val_ds), len(test_ds)])

