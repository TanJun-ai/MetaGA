
import torch
from torch import nn, optim
import numpy as np


class Dice(nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """
    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1
        ps = (x - avg) / torch.sqrt(var)  # N * 1
        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x

class MLP(nn.Module):

    def __init__(self, fc1_in_dim, fc2_in_dim, fc2_out_dim):
        super().__init__()

        layers = list()
        layers.append(nn.Linear(fc1_in_dim, fc2_in_dim))
        # layers.append(nn.BatchNorm1d(fc2_in_dim))
        layers.append(Dice())
        # layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0))
        layers.append(nn.Linear(fc2_in_dim, fc2_out_dim))
        # layers.append(nn.BatchNorm1d(fc2_out_dim))
        layers.append(Dice())
        # layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0))
        layers.append(nn.Linear(fc2_out_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

'''作为Meta-learning的基础模型'''
class BaseModel(nn.Module):

    def __init__(self, args, training_set, testing_set):
        super(BaseModel, self).__init__()
        self.args = args
        self.n_item = args.n_item  # 1872
        self.m_user = args.m_user  # 3864

        self.latent_dim = self.args.latent_dim_rec  # 64
        self.n_layers = self.args.n_layer  # 3
        self.keep_prob = self.args.keep_prob  # 0.6
        self.fc1_in_dim = args.embedding_dim * 6  # 32*6=192
        self.fc2_in_dim = args.first_fc_hidden_dim  # 64
        self.fc2_out_dim = args.second_fc_hidden_dim * 2  # 64
        self.__init_weight()
        self.mlp = MLP(self.fc1_in_dim, self.fc2_in_dim, self.fc2_out_dim)

        print("------------get_train_graph---------------")
        self.train_graph = training_set.getSparseGraph()
        print("============get_test_graph================")
        self.test_graph = testing_set.getSparseGraph()

    '''初始化模型结构和参数'''
    def __init_weight(self):
        '''对items和users的one-hot编码进行embedding，统一输出self.latent_dim=64维'''
        self.embedding_item = nn.Embedding(
            num_embeddings=self.n_item, embedding_dim=self.latent_dim)
        self.embedding_user = nn.Embedding(
            num_embeddings=self.m_user, embedding_dim=self.latent_dim)

        '''如果不使用预训练，0表示不用，则需要初始化normal_，如果使用预训练，则直接加载预训练好的参数'''
        if self.args.pretrain == 0:
            # nn.init.normal_(self.embedding_user.weight, std=0.1)
            # nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

            print('use unnormal distribution initilizer')

        print(f"lgn is already to go(dropout:{self.args.dropout})")

    '''对模型的graph层进行dropout'''
    def _dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    '''定义数据在graph层传播'''
    def computer_graph_embs(self, set_type):
        items_emb = self.embedding_item.weight.to(self.args.mdevice)
        users_emb = self.embedding_user.weight.to(self.args.mdevice)
        """信息的传播方式propagate methods for lightGCN"""
        all_emb = torch.cat([items_emb, users_emb])
        embeddings = [all_emb]
        '''对training-set和testing-set需要分别存储它们的图卷积层的表示矩阵adjacency matrix'''
        if set_type == "train-set":
            if self.args.dropout:
                g_drop = self._dropout(self.train_graph, self.args.keep_prob).to(self.args.mdevice)
            else:
                g_drop = self.train_graph.to(self.args.mdevice)
        elif set_type == "test-set":
            if self.args.dropout:
                g_drop = self._dropout(self.test_graph, self.args.keep_prob).to(self.args.mdevice)
            else:
                g_drop = self.test_graph.to(self.args.mdevice)

        '''生成3层图卷积的高阶聚合信息传播路径，3层是经过实验得出的最佳层数'''
        for layer in range(self.args.n_layer):  # 3层
            all_emb = torch.sparse.mm(g_drop, all_emb)
            embeddings.append(all_emb)

        embeddings = torch.stack(embeddings, dim=1)
        light_out = torch.mean(embeddings, dim=1)
        items, users = torch.split(light_out, [self.args.n_item, self.args.m_user])

        return items, users

    def forward(self, item_id_list, user_id_list, set_type):

        item_id_list = np.array(item_id_list.cpu())
        user_id_list = np.array(user_id_list.cpu())
        all_items, all_users = self.computer_graph_embs(set_type)

        item_emb = all_items[item_id_list].to(self.args.mdevice)  # item_emb的形状[12,64]
        user_emb = all_users[user_id_list].to(self.args.mdevice)  # user_emb的形状[12,64]

        hist_list = []
        for i in range(len(user_id_list)):
            temp_list = list(user_id_list.copy())
            temp_list.remove(user_id_list[i])
            hist_list.append(temp_list)

        hist_embedding = all_users[torch.tensor(hist_list).long()].to(self.args.mdevice)  # user_emb的形状[12,64]
        user_his_emb = self.attention_layer(hist_embedding, user_emb).to(self.args.mdevice)  # 加入item的历史记录向量
        x = torch.cat((item_emb, user_his_emb, user_emb), 1)
        return self.mlp(x)

    def getEmbedding(self, items, pos_users, neg_users, history_users, set_type):

        all_items, all_users = self.computer_graph_embs(set_type)
        items_emb = all_items[items].to(self.args.mdevice)
        pos_emb = all_users[pos_users].to(self.args.mdevice)
        neg_emb = all_users[neg_users].to(self.args.mdevice)
        hist_emb = all_users[history_users].to(self.args.mdevice)

        items_emb_ego = self.embedding_item(items).to(self.args.mdevice)
        pos_emb_ego = self.embedding_user(pos_users).to(self.args.mdevice)
        neg_emb_ego = self.embedding_user(neg_users).to(self.args.mdevice)
        return items_emb, pos_emb, neg_emb, hist_emb, items_emb_ego, pos_emb_ego, neg_emb_ego

    '''加入注意力机制'''
    def attention_layer(self, hist_emb, pos_emb):

        total_hist_emb = []

        '''len(hist_emb)=2048（三维tensor），len(hist_emb[i])=19（二维tensor），
        len(hist_emb[i][j])=64，len(pos_emb[i])=64(一维tensor)，hist_score为19个分数，即每个user_id的权重
        hist_score=tensor([-0.0058, -0.0078,  0.0040, -0.0091, -0.0125, -0.0099,  0.0011,  0.0019,
         0.0042, -0.0103, -0.0041,  0.0010, -0.0007, -0.0103, -0.0055, -0.0158, 0.0011,  0.0006,  0.0009]'''
        for i in range(len(hist_emb)):  # 对每一批进行运算，这里的每一批大小为2048

            # for j in range(len(hist_emb[i])):   # 对每一批（2048）中的每一条历史数据（包含19个user_id）进行运算
            # 计算每一条历史记录的分数，包含了里面19个user_id与候选的pos_user_id的相似度分数（使用内积的方法求得）
            hist_score = torch.mul(hist_emb[i], pos_emb[i])
            hist_score = torch.sum(hist_score, dim=1)

            '''初始化一个embedding，用来计算19个user_id的embedding加权和，
            即获得19个user_id的embedding加权向量total_emb,这个向量的大小也是64维'''
            total_emb = torch.tensor(0).to(self.args.mdevice)
            for u in range(len(hist_emb[i])):
                emb = torch.mul(hist_emb[i][u].to(self.args.mdevice), hist_score[u].to(self.args.mdevice))
                total_emb = total_emb + emb

            total_hist_emb.append(total_emb.tolist())

        user_his_emb = torch.tensor(total_hist_emb).to(self.args.mdevice)

        return user_his_emb

    '''对照公式去理解，有两个损失函数，一个是正负样本的损失函数loss，一个是L2正则项损失函数reg_loss,
    主要是在这里加入attention，用来训练得到更好的模型参数,这里的一批大小为2048，对应关系正确,传入的items, pos, neg为
    一维tensor([1518, ..., 706])，history为二维tensor([[20, ..., 2605,3461],...,])'''
    def bpr_loss(self, items, pos, neg, history, set_type):

        '''这里注意返回的pos_emb为二维tensor，hist_emb为三维tensor，
        tensor([[[-0.0041,  ..., -0.0244],...,]])里面的每一个一维list都表示一个user_id的embedding表示'''
        (items_emb, pos_emb, neg_emb, hist_emb, itemEmb0,  posEmb0, negEmb0) = self.getEmbedding(
            items.long(), pos.long(), neg.long(), history.long(), set_type)
        user_his_emb = self.attention_layer(hist_emb.to(self.args.mdevice), pos_emb.to(self.args.mdevice))  # 加入item的历史记录向量

        reg_loss = (1/2) * (itemEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(items))

        '''正样本得分'''
        pos_scores = torch.mul(items_emb, pos_emb) + torch.mul(user_his_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        '''负样本得分'''
        neg_scores = torch.mul(items_emb, neg_emb) + torch.mul(user_his_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

class BPRLossTraining:
    def __init__(self, recmodel, args):
        self.model = recmodel
        self.weight_decay = args.decay
        self.local_lr = args.local_lr
        self.opt = optim.Adam(self.model.parameters(), lr=self.local_lr, weight_decay=0)
    '''训练每一批数据，并进行参数的局部更新'''
    def batch_traning(self, items, pos, neg, history, set_type):
        loss, reg_loss = self.model.bpr_loss(items, pos, neg, history, set_type)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()




