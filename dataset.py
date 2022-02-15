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
"""
This file implement the dataset for L0_SIGN model
"""

import os
import sys
import numpy as np
import json
import io
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import paddle
from pgl.utils.data import Dataset,Dataloader

import pgl
from pgl.utils.logger import log



def random_split(dataset, split_ratio=0.7, seed=2019, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    trn_split = int(split_ratio * len(dataset))
    test_split = int(0.85 * len(dataset))
    
    train_idx, test_idx, valid_idx = indices[:trn_split], indices[trn_split:test_split], indices[test_split:]

    log.info("train_set : valid_set : test_set = %d : %d: %d" %
             (len(train_idx), len(valid_idx), len(test_idx)))
    return Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx)


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        """getitem"""
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """len"""
        return len(self.indices)


class SIGNDataset(Dataset):

    """Dataset for Detecting Beneficial Feature Interactions for Recommender Systems (L0_SIGN)
    Adapted from https://github.com/ruizhang-ai/SIGN-Detecting-Beneficial-Feature-Interactions-for-Recommender-Systems/data/ml-tag/ml-tag.data.
    """

    def __init__(self,
                 data_path,
                 dataset_name,
                 pred_edges=1,
                 self_loop=True,
                 degree_as_nlabel=False):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.pred_edges = pred_edges
        self.self_loop = self_loop
        self.degree_as_nlabel = degree_as_nlabel

        self.graph_list = []
        self.glabel_list = []

        # global num
        self.num_graph = 0  # total graphs number
        self.num_feature = 0
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        self._load_data()

    def __len__(self):
        """return the number of graphs"""
        return len(self.graph_list)

    def __getitem__(self, idx):
        """getitem"""
        return self.graph_list[idx], self.glabel_list[idx]


    def read_data(self):
        # handle node and class 
        filename = os.path.join(self.data_path, self.dataset_name,
                                "%s.data" % self.dataset_name)
        log.info("loading data from %s" % filename)
        node_list = []
        label = []
        max_node_index = 0
        data_num = 0
        with open(filename, 'r') as f:
            for line in f:
                data_num += 1
                data = line.split()
                # the first element is the label of the class
                label.append(float(data[0]))
                #the rest of the elements are the nodes
                int_list = [int(data[i]) for i in range(len(data))[1:]]
                node_list.append(int_list)
                if max_node_index < max(int_list):
                    max_node_index = max(int_list)

        if not self.pred_edges:
            # pass
            edge_list = []
            sr_list = []    #sender_receiver_list, containing node index
            for nodes in node_list:
                edge_l, sr_l = self.construct_full_edge_list(nodes)
                edge_list.append(edge_l)
                # sr_list.append(sr_l)

                # print('edge_l:',edge_l)
                # print('sr_l:',sr_l)
            # edge_list = [[[],[]] for _ in range(data_num)]
            # sr_list = []
            # # handle edges
            # with open(self.edgefile, 'r') as f:
            #     for line in f:
            #         edge_info = line.split()
            #         node_index = int(edge_info[0])
            #         edge_list[node_index][0].append(int(edge_info[1]))
            #         edge_list[node_index][1].append(int(edge_info[2]))
        else:
            edge_list = []
            sr_list = []    #sender_receiver_list, containing node index
            for nodes in node_list:
                edge_l, sr_l = self.construct_full_edge_list(nodes)
                edge_list.append(edge_l)
                sr_list.append(sr_l)

        # print(label[0:10])
        # print('==========='*10)
        label = self.construct_one_hot_label(label) # 去掉
        # print(label[0:10])

        return node_list, edge_list, label, sr_list, max_node_index + 1, data_num

    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[],[]]         #first for sender, second for receiver
        sender_receiver_list = []
        for i in range(num_node):
            for j in range(num_node)[i:]:
                edge_list[0].append(i)
                edge_list[1].append(j)
                sender_receiver_list.append([nodes[i],nodes[j]])

        return edge_list, sender_receiver_list


    def construct_one_hot_label(self, label):
        """Convert an iterable of indices to one-hot encoded labels."""
        nb_classes = int(max(label)) + 1
        targets = np.array(label, dtype=np.int32).reshape(-1)
        return np.eye(nb_classes)[targets]

    def _load_data(self):
        """Loads dataset
        """
        self.node, edge, label, self.sr_list, node_num, data_num = self.read_data()

        self.num_graph = len(self.node)
        self.num_feature = node_num

        for i in tqdm(range(self.num_graph)):
            # if int((i + 1) %  100000) == 0:
            #     log.info("processing graph %s" % (i + 1))

            graph = dict()  
            edges = []  
            num_edges = 0

            node_features = np.array(self.node[i],dtype='int32').reshape(len(self.node[i]),1)
            num_nodes = len(self.node[i])

            for u,v in zip(edge[i][0],edge[i][1]):
                u_v = (u,v)
                edges.append(u_v)
            
            num_edges = len(edges)
            
            self.glabel_list.append(label[i])

            if self.pred_edges:
                sr = self.sr_list[i]    #the sender receiver list, stored in edge_attr
            else:
                sr = []

            graph['node_attr'] = node_features
            graph['num_nodes'] = num_nodes
            graph['edge_attr'] = sr
            graph['edges'] = edges
            
            assert num_edges == len(edges)

            g = pgl.Graph(
                num_nodes=graph['num_nodes'],
                edges=graph['edges'],
                node_feat={'node_attr': graph['node_attr']},
                edge_feat={'edge_attr': graph['edge_attr']})

            self.graph_list.append(g)

            # update statistics of graphs
            self.n += graph['num_nodes']
            self.m += num_edges

        msg = "Finished loading data\n"
        msg += """
                num_graph:%d
                num_feature:%d
                nodes/graph:%d
                num_edges:%d
                """ %(
                    self.num_graph,
                    self.num_feature,
                    self.n/self.num_graph,
                    self.m
                )
        log.info(msg)
            

def collate_fn(batch_data):
    graphs = []
    labels = []
    for g, l in batch_data:
        graphs.append(g)
        labels.append(l)

    labels = np.array(labels, dtype="int64")

    return graphs, labels


if __name__ == "__main__":
    signdataset = SIGNDataset("./data/", "twitter", pred_edges=0)

    train_ds,val_ds,test_ds = random_split(signdataset)

    loader = Dataloader(
        train_ds,
        batch_size=3,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn)
    cc = 0
    for batch in loader:
        g, label = batch
        # g = pgl.Graph.batch(g).tensor()
        
        print(label)
        print(g.num_graph)
        print('====='*20)
        # print(g.edges)
        print(g.graph_node_id)
        
        print(g.node_feat['node_attr'])
        print(g.edge_feat['edge_attr'])

        # for data in g:
        #     print(data.edges)
            
        
        # for data in g:
        #     print(data.node_feat['node_attr'])
           
        
        # for data in g:
        #     print(data.edge_feat['edge_attr'])

        cc += 1
        if cc == 2:
            break




