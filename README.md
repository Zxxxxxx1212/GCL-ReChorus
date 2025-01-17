# LightGCL-ReChorus
### 机器学习实验大作业

#### 项目介绍
- 以推荐系统为背景，采用[ReChorus2.0框架](https://github.com/THUwangcy/ReChorus)，复现[LightGCL模型](https://github.com/HKUDS/LightGCL)进行一系列对比实验
---

#### 代码位置
- `src/general/LightGCL.py`：LightGCL模型

#### 模型运行
```bash
pip install -r requirements.txt
python src/main.py --model_name LightGCL --dataset data/Grocery_and_Gourmet_Food
```


#### 实验结果
- 在[Grocery_and_Gourmet_Food](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)数据集上进行实验，实验结果如下：


| 模型 | HR@20/Grocery | NDCG@20/Grocery | Times/Grocery | HR@5/Grocery | NDCG@5/Grocery | Times/Grocery |
|------|---------------|-----------------|---------------|--------------|----------------|---------------|
| LightGCL | 0.5916 | 0.3093 | 216.2 | 0.3466 | 0.2382 | 217.8 |
| BPRMF | 0.5304 | 0.2801 | 72.6s | 0.3191 | 0.2195 | 158.3 |
| LightGCN | 0.6109 | 0.3260 | 221.8 s | 0.3699 | 0.2567 | 161.7 |
| LightGCL-Neighbor | 0.6111 | 0.3280 | 159.5 s | 0.3754 | 0.2594 | 157.4 |
| LIghtGCL-Neighbor-r=2 | 0.6109 | 0.3264 | 152.6 | 0.3711 | 0.2565 | 153.4 |
---
- 在[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)数据集上进行实验，实验结果如下：

| 模型 | HR@20/ML1M | NDCG@20/ML1M | Times/ML1M | HR@5/ML1M | NDCG@5/ML1M | Times/ML1M |
|------|------------|--------------|------------|-----------|-------------|------------|
| LightGCL | 0.7178 | 0.3420 | 2530.2 s | 0.3605 | 0.2404 | 2710.8 s |
| BPRMF | 0.7467 | 0.3598 | 207.1 s | 0.3779 | 0.2549 | 228.5 s |
| LightGCN | 0.7255 | 0.3555 | 1295.3 s | 0.3793 | 0.2553 | 1622.4 s |
| LightGCL-Neighbor | 0.6781 | 0.3241 | 1495.5 s | 0.3424 | 0.2283 | 1685.1 s |
| LightGCL-Neighbor-r=2 | 0.6945 | 0.3381 | 969.4 s | 0.3577 | 0.2421 | 1135.0 s |


