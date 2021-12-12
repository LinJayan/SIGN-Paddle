import numpy as np
import paddle
from net import L0_SIGN
from sklearn.metrics import roc_auc_score, accuracy_score

from paddle.optimizer import Adam

import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader

from dataset import SIGNDataset, random_split, collate_fn



def train(args, data_info, data_nums):
    train_loader = data_info[0]
    val_loader = data_info[1]
    test_loader = data_info[2]
    num_feature = data_info[3]
    
    model = L0_SIGN(args, num_feature)
    
    print(model.sublayers())
    print('----'*20)
    for item in model.named_sublayers():
        print(item)
    print('===='*20)
    print(model.parameters())
    # optimizer = torch.optim.Adagrad(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     args.lr,
    #     lr_decay=1e-5,
    #     #weight_decay=1e-5
    # )
    optimizer = paddle.optimizer.Adagrad(learning_rate=args.lr,
        parameters=model.parameters())
    # optimizer =paddle.optimizer.Adam(learning_rate=args.lr,
    #     parameters=model.parameters())
    #crit = torch.nn.BCELoss()
    crit = paddle.nn.MSELoss()

    # print([i.size() for i in filter(lambda p: not p.stop_gradient, model.parameters())])
    log.info('start training...')
    for step in range(args.n_epoch):
        # training
        loss_all = 0
        edge_all = 0
        model.train()
        for data in train_loader:
            g, label = data
            g = pgl.Graph.batch(g).tensor()
            label = paddle.to_tensor(label,dtype='float32')
            # print(g.graph_node_id)
            #return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            output, l0_penaty, l2_penaty, num_edges = model(g)
            # print('===='*8)
            # print('output:',output)
            # print('label:',label)
            
            baseloss = crit(output, label)
            l0_loss = l0_penaty * args.l0_weight 
            l2_loss = l2_penaty * args.l2_weight
            loss = baseloss + l0_loss + l2_loss 
            loss_all += g.num_graph * loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        cur_loss = loss_all / data_nums[0]

        # evaluation
        # train_auc = 0
        train_auc, train_acc, train_edges = evaluate(model, train_loader)
        val_auc,val_acc, _ = evaluate(model, val_loader)    
        test_auc, test_acc, test_edges = evaluate(model, test_loader)
        # print(step, cur_loss.numpy()[0], train_auc, val_auc, val_acc, test_auc, test_acc, test_edges)
        log.info('Epoch: {:03d}, Loss: {:.4f}, Train Auc: {:.4f},Train Acc: {:.4f},Train edges: {:07d}, Val Auc: {:.4f}, Acc: {:.4f}, Test Auc: {:.4f}, Acc: {:.4f}, Train edges: {:07d}'.
          format(step, cur_loss.numpy()[0].astype('float32'), train_auc,train_acc,train_edges, val_auc, val_acc, test_auc, test_acc, test_edges))
      


def evaluate(model, loader):
    model.eval()

    predictions = []
    labels = []
    edges_all = 0
    with paddle.no_grad():
        
        for data in loader:
           
            g, label = data
            g = pgl.Graph.batch(g).tensor()
            label = paddle.to_tensor(label,dtype='float32')
            pred, _, _, num_edges = model(g)
            # print(g.num_graph)
            # print(num_edges)
            # print('==='*15)
            pred = pred.detach().cpu().numpy()
            edges_all += num_edges
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    # print('predictions:',len(predictions))
    # print('labels:',len(labels))
    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(np.argmax(labels, 1), np.argmax(predictions, 1))
    return auc, acc, edges_all
