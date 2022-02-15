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
This file implement the L0_SIGN model
"""


import paddle
import paddle.nn as nn 
import paddle.nn.functional as F
from paddle.nn import Linear
import numpy as np

import pgl
from pgl.nn import pool 
from pgl.utils.logger import log

from pgl.graph import Graph



class L0_SIGN(nn.Layer):
    def __init__(self, args, n_feature):
        super(L0_SIGN, self).__init__()

        self.pred_edges = args.pred_edges
        self.n_feature = n_feature
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.l0_para = eval(args.l0_para)


        if self.pred_edges:
            self.linkpred = LinkPred(self.dim, self.hidden_layer, self.n_feature,  self.l0_para)

        self.sign = SIGN(self.dim, self.hidden_layer, self.pred_edges)
       
        self.g = nn.Linear(self.dim, 2)  #2 is the class dimention 
        self.add_sublayer("g",self.g)
        
        self.feature_emb = nn.Embedding(self.n_feature, self.dim)

        self.graph_pooling = pgl.nn.GraphPool(pool_type='MEAN')


    def forward(self, graph, is_training=True):
        # does not conduct link prediction, use all interactions
        # graph: pgl.Graph object
        # graph.node_feat['node_attr']: [bacth_size*3, 1]
        # graph.edge_feat['edge_attr']: [bact_size*6, 2]
        # graph.edges: [bact_size*6, 2]

        x, edge_index, sr = graph.node_feat['node_attr'], graph.edges, graph.edge_feat['edge_attr']
        segment_ids = graph.graph_node_id

        
        x = self.feature_emb(x)
        x = x.squeeze(1)

        graph.node_feat['node_attr'] = x
        

        if self.pred_edges:
            sr = paddle.transpose(sr, perm=[1, 0])    # [2, num_edges]
    
            s, l0_penaty = self.linkpred(sr, is_training)
            pred_edge_index, pred_edge_weight = self.construct_pred_edge(edge_index, s) 


            # 创建训练子图
            # ============== BUG ==============
            subgraph_ = pgl.Graph(
                # num_nodes=graph.num_nodes,
                node_feat={'node_attr':x},
                edges=pred_edge_index)
            
           
            updated_nodes = self.sign(subgraph_, edge_attr=pred_edge_weight)
        
            num_edges = pred_edge_weight.shape[0]
        else:
            updated_nodes = self.sign(graph)
            # updated_nodes = self.gin(graph, x)
            l0_penaty = 0
            num_edges = edge_index.shape[1]

        # l2_penaty = (updated_nodes * updated_nodes).sum()
        l2_penaty = paddle.multiply(updated_nodes, updated_nodes).sum()

        # 图平均池化
        graph_embedding = pgl.math.segment_mean(updated_nodes, segment_ids)

        out = self.g(graph_embedding)
        out = paddle.clip(out, min=0, max=1)
        return out, l0_penaty, l2_penaty, num_edges 

    def construct_pred_edge(self, fe_index, s):
        """
        fe_index: full_edge_index, [2, all_edges_batchwise]
        s: predicted edge value, [all_edges_batchwise, 1]

        construct the predicted edge set and corresponding edge weights
        """
        new_edge_index = [[],[]]
        edge_weight = []
        
        s = paddle.squeeze(s)
        
        fe_index = paddle.transpose(fe_index, perm=[1, 0]) 

        sender = paddle.unsqueeze(fe_index[0][s>0], 0)  
        receiver = paddle.unsqueeze(fe_index[1][s>0], 0)

        pred_index = paddle.concat([sender, receiver], 0)
        pred_weight = s[s>0]
        pred_index = paddle.transpose(pred_index, perm=[1, 0]) 
        

        return pred_index, pred_weight 

   


class SIGN(nn.Layer):
    """Implementation of graph attention networks (SIGN)

    This is an implementation of the paper Detecting Beneficial Feature Interactions for Recommender Systems
    (https://arxiv.org/pdf/2008.00404v6.pdf).

    Args:

        input_size: The size of the inputs. 

        hidden_size: The hidden size for sign.

    """

    def __init__(self,input_size, hidden_size, pred_edges, aggr_func="mean"):
        super(SIGN, self).__init__()
        assert aggr_func in ["sum", "mean", "max", "min"], \
                "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func
        self.pred_edges = pred_edges

        #construct pairwise modeling network

        self.lin1 = paddle.nn.Linear(input_size, hidden_size, weight_attr=nn.initializer.KaimingUniform())
        self.lin2 = paddle.nn.Linear(hidden_size, input_size, weight_attr=nn.initializer.KaimingUniform())
        self.activation = paddle.nn.ReLU()

        self.add_sublayer("lin1_g", self.lin1)
        self.add_sublayer("lin2_g", self.lin2)
        self.add_sublayer("activation", self.activation)

    def _send_func(self, src_feat, dst_feat, edge_feat=None):
        
        pairwise_analysis = self.lin1(paddle.multiply(src_feat["src"], dst_feat["dst"]))
        pairwise_analysis = self.activation(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)

       
        if edge_feat != None:
            edge_feat_ = paddle.reshape(edge_feat["e_attr"],[-1,1])
            interaction_analysis = paddle.multiply(pairwise_analysis , edge_feat_)

        else:
            interaction_analysis = pairwise_analysis

        return {'msg':interaction_analysis}

    def _recv_func(self, msg):

        return getattr(msg, self.aggr_func)(msg["msg"])
        # return msg["msg"]
        

    def forward(self, graph, edge_attr=None):
        """
        Args:
 
            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

        Return:

            If `concat=True` then return a tensor with shape (num_nodes, hidden_size),
            else return a tensor with shape (num_nodes, hidden_size * num_heads) 

        """

        self_feature = graph.node_feat['node_attr']


        msg = graph.send(
            self._send_func,
            src_feat={"src": self_feature.clone()},
            dst_feat={"dst": self_feature.clone()},
            edge_feat={"e_attr":edge_attr})
        
        output = graph.recv(reduce_func=self._recv_func, msg=msg)

        return output



class LinkPred(nn.Layer):
    def __init__(self, D_in, H, n_feature, l0_para):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(LinkPred, self).__init__()
        self.linear1 = nn.Linear(D_in, H, weight_attr=nn.initializer.KaimingUniform())
        self.add_sublayer("linear1_L",self.linear1)
        self.linear2 = nn.Linear(H, 1, weight_attr=nn.initializer.KaimingUniform())
        self.add_sublayer("linear2_L",self.linear2)
        self.relu = nn.ReLU()
        self.add_sublayer("relu_L",self.relu)
        self.dropout = nn.Dropout(p=0.5)
        self.add_sublayer("dropout",self.dropout)
       
        with paddle.no_grad():
            # self.linear1.weight.set_value(self.linear1.weight + 0.2 )
            self.linear2.weight.set_value(self.linear2.weight + 0.2 ) 

        self.temp = l0_para[0]      #temprature
        self.inter_min = l0_para[1] 
        self.inter_max = l0_para[2] 
        
        self.feature_emb_edge = nn.Embedding(n_feature, D_in, 
                                            weight_attr=paddle.ParamAttr(name='emb_weight',
                                                                    initializer=nn.initializer.Normal(mean=0.2,std=0.01)))    #D_in is the dimension size

    def forward(self, sender_receiver, is_training):
        #construct permutation input
        # print(sender_receiver)  # ok
        sender_emb = self.feature_emb_edge(sender_receiver[0,:])
        receiver_emb = self.feature_emb_edge(sender_receiver[1,:])
        # print(sender_emb)  # ok
        # print(receiver_emb)
        # _input = sender_emb * receiver_emb       #element wise product sender and receiver embeddings
        _input = paddle.multiply(sender_emb , receiver_emb) 

        # print(_input)
        #loc = _input.sum(1)
        h_relu = self.dropout(self.relu(self.linear1(_input)))
        # print(h_relu) #############
        loc = self.linear2(h_relu)  ########
        # print(loc)
        # loc = self.relu(loc_)
        if is_training:
            u = paddle.rand(loc.shape, dtype=loc.dtype) # ========= DEBUG
            u.stop_gradient = False
            logu = paddle.log2(u)
            # print("logu :",logu)
            logmu = paddle.log2(1-u)
            # print("logmu :",logmu)
            sum_log = loc + logu - logmu
            # print("sum_log :",sum_log)
            s = F.sigmoid(sum_log/self.temp)
            # s = F.relu(sum_log/self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min
            
        else:
            s = F.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min

        # print(s)
        s = paddle.clip(s, min=0, max=1)

        # s_ = paddle.ones_like(s)
        # s_.stop_gradient = False


        l0_penaty = F.sigmoid(loc - self.temp * np.log2(-self.inter_min/self.inter_max)).mean()

        return s, l0_penaty 

    def permutate_batch_wise(x, batch):
        """
        x: all feature embeddings all batch
        batch: a list containing feature belongs to which graph
        """
        return

