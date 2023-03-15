
import argparse
from time import time
import numpy as np
import torch
from model import BaseModel, BPRLossTraining
from dataset_load import LoaderDataset
from metaga_training import model_training, supp_testing, query_testing
from attention_layer import model_test_training

torch.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()

# parser.add_argument('--data_dir', type=str, default='../data/lastfm-20-100-400')
parser.add_argument('--data_dir', type=str, default='../data/movielens-1m-200-800')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing-400-1600')
# parser.add_argument('--data_dir', type=str, default='../data/book_crossing-200-800')

parser.add_argument('--dataset_name', type=str, default='define')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--latent_dim_rec', type=int, default=64, help="the embedding size of lightGCN")
parser.add_argument('--n_layer', type=int, default=3, help="the layer num of lightGCN")
parser.add_argument('--keep_prob', type=float, default=0.6, help="the batch size for bpr loss training procedure")
parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not，0 is not use')
parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not，0 is not use")
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--bpr_batch', type=int, default=1024, help="the batch size for bpr loss training procedure")
parser.add_argument('--local_lr', type=float, default=0.001, help="the local model learning rate")
parser.add_argument('--meta_lr', type=float, default=0.005, help="the meta model learning rate")
parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalization")
parser.add_argument('--test_batch', type=int, default=100, help="the batch size of items for testing")
parser.add_argument('--top_k', type=int, default=10, help="the size of recommendation")
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='Embedding dimension for item and user.')
parser.add_argument('--second_fc_hidden_dim', type=int, default=32, help='Embedding dimension for item and user.')

args = parser.parse_args()
'''设置随机种子，使得生成的随机数固定不变'''
torch.manual_seed(args.seed)


'''设置运行的cuda'''
GPU = torch.cuda.is_available()
mdevice = torch.device('cuda:1' if GPU else 'cpu')
args.mdevice = mdevice
# args.n_item = 3846  # lastfm-20
# args.m_user = 1872
args.n_item = 3953  # movielens-1m
args.m_user = 6041
# args.n_item = 8000  # book_crossing
# args.m_user = 2947

item_size = 14  # 7*2

'''training-set和testing-set按照头部尾部用户区分，training-set为头部用户，
testing-set为尾部用户，他们的user_id（一个user_id对应多个item_id）是不同的；
而training-set里面的support-set和query-set是按照一条user_id：[item_id1，...]来划分的，
前20个item_id划分给support-set，后面的12个item_id划分给query-set，所以support-set和query-set
都有同样的user_id。support-set的历史点击记录可以用到query-set中。'''
print("=====================training-set-start=======================")
training_supp_path = args.data_dir + '/training/top_supp_10_pos.txt'
train_dataset = '/training'
training_supp_dataset = LoaderDataset(args, training_supp_path, train_dataset)
print("training_supp_items:" + str(training_supp_dataset.n_item))
print("training_supp_users:" + str(training_supp_dataset.m_user))
training_supp_size = len(training_supp_dataset.trainUniqueItems)   # 训练集有100条数据
print("====training_supp_size====:" + str(training_supp_size))

training_query_path = args.data_dir + '/training/top_query_7_pos.txt'
training_query_dataset = LoaderDataset(args, training_query_path, train_dataset)
print("training_query_items:" + str(training_query_dataset.n_item))
print("training_query_users:" + str(training_query_dataset.m_user))
training_query_size = len(training_query_dataset.trainUniqueItems)
print("====training_query_size====:" + str(training_query_size))
print("=====================training-set-end=======================")

print("=====================testing-set-start=======================")
testing_supp_path = args.data_dir + '/testing/tail_supp_7_pos.txt'
test_dataset = '/testing'
testing_supp_dataset = LoaderDataset(args, testing_supp_path, test_dataset)
print("testing_supp_items:" + str(testing_supp_dataset.n_item))
print("testing_supp_users:" + str(testing_supp_dataset.m_user))
testing_supp_size = len(testing_supp_dataset.trainUniqueItems)   # 测试集有400条数据
print("=====testing_supp_size=====:" + str(testing_supp_size))

testing_query_path = args.data_dir + '/testing/tail_query_7_pos.txt'
testing_query_dataset = LoaderDataset(args, testing_query_path, test_dataset)
print("testing_query_items:" + str(testing_query_dataset.n_item))
print("testing_query_users:" + str(testing_query_dataset.m_user))
testing_query_size = len(testing_query_dataset.trainUniqueItems)   # 测试集有400条数据
print("=====testing_query_size=====:" + str(testing_query_size))
print("=====================testing-set-end=======================")


Recmodel = BaseModel(args, training_supp_dataset, testing_supp_dataset)
Recmodel = Recmodel.to(mdevice)
meta_optim = torch.optim.Adam(Recmodel.parameters(), lr=args.meta_lr, weight_decay=0.0001)
bpr = BPRLossTraining(Recmodel, args)


max_ndcg10 = 0.
max_prec10 = 0.
max_ndcg8 = 0.
max_prec8 = 0.
max_ndcg5 = 0.
max_prec5 = 0.

for epoch in range(args.epochs):

    test_start = time()
    """meta-learning，寻找出比较合适的θ"""
    '''这步model_training必须有，使用BPR损失函数来寻找最优模型参数'''
    training_supp_loss = model_training(args, training_supp_dataset, Recmodel, bpr, "train-set")
    t_loss = supp_testing(args, training_query_dataset, Recmodel, meta_optim, item_size)
    train_end = time()
    print(f'EPOCH[{epoch + 1}/{args.epochs}] Loss[{t_loss:.3f}] Time[{train_end - test_start:.3f}]')


    """local-learning,寻找最合适的θ1，θ2，...，θn"""
    testing_aver_loss = model_test_training(args, training_supp_dataset, testing_supp_dataset, Recmodel, bpr)
    test_pre10, test_ndcg10, test_pre8, test_ndcg8, test_pre5, test_ndcg5 = query_testing(
        args, testing_query_dataset, Recmodel, item_size)

    test_end = time()

    if test_pre10 > max_prec10:
        max_prec10 = test_pre10
    if test_ndcg10.item() > max_ndcg10:
        max_ndcg10 = test_ndcg10.item()
    if test_pre8 > max_prec8:
        max_prec8 = test_pre8
    if test_ndcg8.item() > max_ndcg8:
        max_ndcg8 = test_ndcg8.item()
    if test_pre5 > max_prec5:
        max_prec5 = test_pre5
    if test_ndcg5.item() > max_ndcg5:
        max_ndcg5 = test_ndcg5.item()
    if epoch % 10 == 0:
        print("=============================query-testing-start=============================")
        print("TOP-10: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(test_pre10, test_ndcg10.item()))
        print("TOP-8: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(test_pre8, test_ndcg8.item()))
        print("TOP-5: query_prec:{:.4f}\t\tquery_ndcg:{:.4f}".format(test_pre5, test_ndcg5.item()))
        print("--------------------------------------------------------------------------------")
        print("TOP-10: max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec10, max_ndcg10))
        print("TOP-8: max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec8, max_ndcg8))
        print("TOP-5: max_prec:{:.4f}\t\tmax_ndcg:{:.4f}".format(max_prec5, max_ndcg5))
        print("===========================query-testing-end=================================")