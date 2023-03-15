

import torch
import random
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time


class LoaderDataset:
    """
    Procceeding Dataset for lastfm20 ：这个数据集分成support-set 和 query-set，
    support-set 中的item最大值为3846，user最大值为1872，
    query-set 中的item最大值为3846，user最大值为1872
    """
    def __init__(self, parameter, train_path, dataset):
        self.n_item = 0
        self.m_user = 0
        self.trainDataSize = 0

        self.save_path = parameter.data_dir + dataset
        self.args = parameter
        trainUniqueItems, trainItem, trainUser = [], [], []

        with open(train_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().strip('\n').split(' ')
                    users = [int(i) for i in l[1:]]
                    item_id = int(l[0])
                    trainUniqueItems.append(item_id)
                    trainItem.extend([item_id] * len(users))
                    trainUser.extend(users)
                    self.n_item = max(self.n_item, item_id)
                    self.m_user = max(self.m_user, max(users))
                    self.trainDataSize += len(users)

        self.trainUniqueItems = np.array(trainUniqueItems)
        self.trainItem = np.array(trainItem)
        self.trainUser = np.array(trainUser)

        print(f"{self.trainDataSize} interactions for training")

        '''手动设置item和user的最大容量，确保矩阵不会越界'''
        self.n_item = self.args.n_item
        self.m_user = self.args.m_user
        '''user 和正样本item 的交互矩阵，矩阵的元素值都为1.0，这里从0开始，即user_id为0的user其
        真实的user_id=1,(0, 1) 1.0。。。(6039, 3819)	1.0,数据类型为矩阵csr_matrix'''
        self.UserItemNet = csr_matrix((np.ones(len(self.trainItem)), (self.trainItem, self.trainUser)),
                                      shape=(self.n_item, self.m_user))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        '''self.n_user=6040，拿到每个item对应的正样本user_id的list，self.allPos=[...,[...3671 3683 3703 3735 3751 3819]]'''
        self.allPos = self.getUserPosItems(list(range(self.n_item)))

        print(f"{parameter.dataset_name} is ready to go")

        self.Graph = None

    def getUserPosItems(self, items):
        posUsers = {}
        for item in items:
            posUsers[item] = self.UserItemNet[item].nonzero()[1]
        return posUsers

    '''稀疏矩阵转成稠密矩阵'''
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    '''如果已经有了生成好的图卷积层，直接加载，如果没有，需要重新生成
    生成原理需要对照论文中的公式，这里的创新点是在其基础上增加候选user和交互历史user的相似度权重'''
    def getSparseGraph(self):
        print("loading exist adjacency matrix")
        try:
            pre_adj_mat = sp.load_npz(self.save_path + '/s_pre_adj_mat.npz')
            print("successfully loaded......")
            norm_cos_adj = pre_adj_mat
        except :
            print("generating new adjacency matrix......")
            start = time()

            adj_mat = sp.dok_matrix((self.n_item + self.m_user, self.n_item + self.m_user), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            '''adj_mat对应公式（8）中的邻接矩阵A'''
            adj_mat[:self.n_item, self.n_item:] = R
            adj_mat[self.n_item:, :self.n_item] = R.T
            adj_mat = adj_mat.todok()
            np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
            row_sum = np.array(adj_mat.sum(axis=1))  # 将adj_mat中一行的元素相加

            '''这里的权重仅仅是距离的权重，距离越远，权重越低，创新点是在这个基础上加入相似度的权重
            norm_adj=1/√|Nu||Ni|，norm_adj=norm_adj+γ，γ是用attention原理计算得到,
            norm_adj是一个(5718, 5718)的矩阵，d_mat和adj_mat也是(5718, 5718)的矩阵，在lastfm数据集上，
            R的形状为(3846, 1872)，即5718=3846+1872，row_sum形状为（5718,1）'''
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            '''形成对角矩阵，比如d_inv=[0.2,0,0.1]，会形成一个3x3的对角矩阵，对角元素分别是0.2,0,0.1，
            这里因为d_inv为size=5718的list，因此对角矩阵d_mat的形状为(5718, 5718)'''
            d_mat = sp.diags(d_inv)

            '''d_mat对应公式（8）中的矩阵D^(-1/2)，下面的操作实现公式（8）：D^(-1/2)*A*D^(-1/2)'''
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            # norm_adj = norm_adj + γ
            '''下面计算γ'''
            c_mat = adj_mat.T.dot(adj_mat)
            atten_row_sum = np.array(c_mat.sum(axis=1))
            z_inv = np.power(atten_row_sum, -0.5).flatten()
            z_inv[np.isinf(z_inv)] = 0.
            z_mat = sp.diags(z_inv)
            cos_adj = z_mat.dot(adj_mat)
            cos_adj = cos_adj.dot(z_mat)
            cos_adj = cos_adj.tocsr()

            norm_cos_adj = norm_adj + cos_adj
            # norm_cos_adj = norm_adj
            end = time()
            print(f"costing {end -start} s, saved norm_mat...")
            # sp.save_npz(self.save_path + '/s_pre_adj_mat.npz', norm_cos_adj)


        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_cos_adj).to(self.args.mdevice)
        self.Graph = self.Graph.coalesce().to(self.args.mdevice)

        return self.Graph

def UniformSample(dataset, set_type):

    '''allPos是每个item_id对应的正样本user_ids，为字典类型 3947: array([9, 10, ...])'''
    allPos = dataset.allPos  # 所有的正样本item_id

    item_num = dataset.n_item  # item的数量,这里是3948
    items = random.sample(range(0, dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）
    S = []  # 用来获取三元组<item, user+, user->

    for i, item in enumerate(items):

        posForItem = allPos[item]
        # 这个item没有对应的user_id，即item没有user的交互记录，则跳过这个item
        if len(posForItem) == 0:
            continue
        '''生成范围内不重复的随机整数，生成的个数为len(posForItem)'''
        posindex = random.sample(range(0, len(posForItem)), len(posForItem))
        '''
        item和它对应的正样本user(item:posForItem)
        1088:[53  155  156  196  419  422  510  514  756  795  846 1213 1372 1498 1528 1545 1584 1675 1691 2580]'''
        for index in range(len(posForItem)):
            '''这里是从每个item对应的正样本user_id的list中随机抽取一个正样本posuser'''
            pos_index = posindex[index]
            posuser = posForItem[pos_index]  # 从posForItem抽取一个正样本
            history_items = list(posForItem)  # history_items是抽取一个正样本后，剩下的item_id都是这个正样本的历史交互记录
            history_items.remove(posuser)
            '''np.array是具有相同长度的数组，所以要取最小数量的那个作为history_items，否则会报错误：
            IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed'''
            history_items = history_items[0:9]
            '''负样本抽取策略：从0-m_user个user_id中随机找一个id出来，如果这个user_id在正样本list中，则继续找，
            直到找到的user_id不在正样本的list中，就是item对应的负样本'''

            if set_type == "train-set":
                while True:
                    neguser = np.random.randint(0, dataset.m_user)
                    if neguser in posForItem:
                        continue
                    else:
                        break
                "返回三元组<item, user+, user->，len(posForItem)为每个item对应的交互user_id的集合"
                S.append([item, posuser, neguser] + history_items)
            else:
                while True:
                    # neguser = np.random.randint(0, dataset.m_user)
                    neguser = np.random.randint(0, 2500)      # 对movielens-1m数据集所用
                    if neguser in posForItem:
                        continue
                    else:
                        break
                S.append([item, posuser] + [1])
                S.append([item, neguser] + [0])

    Sample = np.array(S)

    return Sample
