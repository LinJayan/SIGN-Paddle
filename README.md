## **基于Paddle的SIGN模型复现**

### **快速开始**
```
# GPU训练
export CUDA_VISIBLE_DEVICES=0
python main.py --device gpu --dataset ml-tag  # --dataset [data-path]

# CPU训练
python main.py  --dataset ml-tag # --dataset [data-path]
```

### **复现结果**

| 模型 | 数据集 | AUC | ACC |
| -------- | -------- | -------- | -------- |
| SIGN-L0     | ml-tag     | 0.940+     | 0.90+     |


### **部分训练日志**
```
[INFO] 2022-02-15 12:27:27,238 [  dataset.py:  129]:    loading data from ./data/ml-tag/ml-tag.data
100%|████████████████████████████████████████████████████████████████████████████████| 2006859/2006859 [01:00<00:00, 33296.96it/s]
[INFO] 2022-02-15 12:29:00,628 [  dataset.py:  263]:    Finished loading data
[INFO] 2022-02-15 12:29:00,628 [  dataset.py:  263]:    Finished loading data

                num_graph:2006859
                num_feature:90445
                nodes/graph:3
                num_edges:12041154
                
[INFO] 2022-02-15 12:29:02,437 [  dataset.py:   65]:    train_set : valid_set : test_set = 1404801 : 301029: 301029

datast: ml-tag
vector dim: 8
batch_size: 1024
lr: 0.05

[INFO] 2022-02-15 12:29:02,602 [    train.py:   46]:    start training...
[INFO] 2022-02-15 12:31:45,725 [    train.py:   98]:    Epoch: 000, Loss: 0.1501, Train Auc: 0.9247,Train Acc: 0.8943,Train edges: 8428799, Val Auc: 0.9151, Acc: 0.8899, Test Auc: 0.9158, Acc: 0.8910, Train edges: 1806172
[INFO] 2022-02-15 12:34:32,008 [    train.py:   98]:    Epoch: 001, Loss: 0.1077, Train Auc: 0.9429,Train Acc: 0.9026,Train edges: 8428803, Val Auc: 0.9295, Acc: 0.8950, Test Auc: 0.9301, Acc: 0.8960, Train edges: 1806173
[INFO] 2022-02-15 12:37:17,265 [    train.py:   98]:    Epoch: 002, Loss: 0.0961, Train Auc: 0.9537,Train Acc: 0.9114,Train edges: 8428804, Val Auc: 0.9358, Acc: 0.9005, Test Auc: 0.9365, Acc: 0.9009, Train edges: 1806172
[INFO] 2022-02-15 12:39:59,226 [    train.py:   98]:    Epoch: 003, Loss: 0.0883, Train Auc: 0.9616,Train Acc: 0.9183,Train edges: 8428805, Val Auc: 0.9392, Acc: 0.9034, Test Auc: 0.9401, Acc: 0.9039, Train edges: 1806174
[INFO] 2022-02-15 12:42:41,809 [    train.py:   98]:    Epoch: 004, Loss: 0.0819, Train Auc: 0.9674,Train Acc: 0.9244,Train edges: 8428804, Val Auc: 0.9415, Acc: 0.9057, Test Auc: 0.9420, Acc: 0.9065, Train edges: 1806174
[INFO] 2022-02-15 12:45:26,478 [    train.py:   98]:    Epoch: 005, Loss: 0.0764, Train Auc: 0.9717,Train Acc: 0.9304,Train edges: 8428806, Val Auc: 0.9423, Acc: 0.9076, Test Auc: 0.9431, Acc: 0.9084, Train edges: 1806174
[INFO] 2022-02-15 12:48:10,314 [    train.py:   98]:    Epoch: 006, Loss: 0.0716, Train Auc: 0.9751,Train Acc: 0.9350,Train edges: 8428805, Val Auc: 0.9424, Acc: 0.9088, Test Auc: 0.9433, Acc: 0.9098, Train edges: 1806174
[INFO] 2022-02-15 12:50:54,072 [    train.py:   98]:    Epoch: 007, Loss: 0.0676, Train Auc: 0.9779,Train Acc: 0.9391,Train edges: 8428805, Val Auc: 0.9420, Acc: 0.9100, Test Auc: 0.9427, Acc: 0.9103, Train edges: 1806174
[INFO] 2022-02-15 12:53:36,106 [    train.py:   98]:    Epoch: 008, Loss: 0.0640, Train Auc: 0.9798,Train Acc: 0.9415,Train edges: 8428806, Val Auc: 0.9411, Acc: 0.9105, Test Auc: 0.9418, Acc: 0.9109, Train edges: 1806174
[INFO] 2022-02-15 12:56:23,378 [    train.py:   98]:    Epoch: 009, Loss: 0.0610, Train Auc: 0.9816,Train Acc: 0.9447,Train edges: 8428805, Val Auc: 0.9404, Acc: 0.9110, Test Auc: 0.9412, Acc: 0.9116, Train edges: 1806174
```

