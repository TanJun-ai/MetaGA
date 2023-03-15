
import torch
from utils import add_metric
from dataset_load import UniformSample
from utils import shuffle, minibatch
from torch.nn import functional as F


'''对train-set数据集进行训练'''
def model_training(args, dataset, recommend_model, loss_class, set_type):

    Recmodel = recommend_model
    Recmodel.train()
    bpr_loss = loss_class

    '''均匀抽样,得到S=[[1269 3706 4989] ...]'''
    S = UniformSample(dataset, set_type)

    items_id = torch.Tensor(S[:, 0]).long()
    pos_users_id = torch.Tensor(S[:, 1]).long()
    neg_users_id = torch.Tensor(S[:, 2]).long()
    history_users_id = torch.Tensor(S[:, 3:]).long()

    items_id = items_id.to(args.mdevice)
    pos_users_id = pos_users_id.to(args.mdevice)
    neg_users_id = neg_users_id.to(args.mdevice)
    history_users_id = history_users_id.to(args.mdevice)

    '''utils.shuffle用于随机打乱顺序，但对应关系不变'''
    items_id, pos_users_id, neg_users_id, history_users_id = shuffle(items_id, pos_users_id, neg_users_id, history_users_id)

    total_batch = len(items_id) // args.bpr_batch + 1  # args.bpr_batch=2048
    aver_loss = 0.
    '''将组合矩阵(items_id, pos_users_id, neg_items_id）分成batch_i批'''
    for (batch_i, (batch_items, batch_pos, batch_neg, batch_hist)) in \
            enumerate(minibatch(items_id, pos_users_id, neg_users_id, history_users_id, batch_size=args.bpr_batch)):
        '''获取每个批次的loss'''
        batch_loss = bpr_loss.batch_traning(batch_items, batch_pos, batch_neg, batch_hist, set_type)
        aver_loss += batch_loss

    aver_loss = aver_loss / total_batch

    return round(aver_loss, 3)

'''对train-set数据集进行测试'''
def supp_testing(args, dataset, recommend_model, meta_optim, item_size):

    Recmodel = recommend_model

    S = UniformSample(dataset, "test-set")

    items_id = torch.Tensor(S[:, 0]).long()
    users_id = torch.Tensor(S[:, 1]).long()
    labels = torch.Tensor(S[:, 2]).long()

    items_id = items_id.to(args.mdevice)
    users_id = users_id.to(args.mdevice)
    labels = labels.to(args.mdevice)

    test_data_len = len(dataset.trainUniqueItems)

    """这里的测试是按照一个item_id对应其所有交互的user_id来进行的，例如：
	37: [1240 483 149 2 658 246 95 539 628 1689 1800 53 47 179]
	会先预测item_id=37的user_id列表，然后对比其label，计算precision和ndcg"""
    train_loss = []
    for i in range(test_data_len):
        batch_items = items_id[i * item_size:(i + 1) * item_size]
        batch_users = users_id[i * item_size:(i + 1) * item_size]
        batch_labels = labels[i * item_size:(i + 1) * item_size]

        batch_items, batch_users, batch_labels = shuffle(batch_items, batch_users, batch_labels)

        testing_query_y_pred = Recmodel(batch_items.to(args.mdevice), batch_users.to(args.mdevice), "test-set")

        loss = F.mse_loss(testing_query_y_pred.to(args.mdevice), batch_labels.float().view(-1, 1).to(args.mdevice))
        train_loss.append(loss)

    t_loss = torch.stack(train_loss).mean(0)
    meta_optim.zero_grad()
    t_loss.backward()
    meta_optim.step()

    return t_loss

'''对test-set数据进行一次测试'''
def query_testing(args, dataset, recommend_model, item_size):
    Recmodel = recommend_model
    Recmodel.eval()  # 模型测试

    '''加载和处理数据'''
    S = UniformSample(dataset, "test-set")

    items_id = torch.Tensor(S[:, 0]).long()
    users_id = torch.Tensor(S[:, 1]).long()
    labels = torch.Tensor(S[:, 2]).long()

    items_id = items_id.to(args.mdevice)
    users_id = users_id.to(args.mdevice)
    labels = labels.to(args.mdevice)

    test_data_len = len(dataset.trainUniqueItems)


    total_ndcg_list10 = []
    total_prec_list10 = []
    total_ndcg_list8 = []
    total_prec_list8 = []
    total_ndcg_list5= []
    total_prec_list5 = []


    """这里的测试是按照一个item_id对应其所有交互的user_id来进行的，例如：
    37: [1240 483 149 2 658 246 95 539 628 1689 1800 53 47 179]
    会先预测item_id=37的user_id列表，然后对比其label，计算precision和ndcg"""
    for i in range(test_data_len):
        batch_items = items_id[i * item_size:(i + 1) * item_size]
        batch_users = users_id[i * item_size:(i + 1) * item_size]
        batch_labels = labels[i * item_size:(i + 1) * item_size]

        batch_items, batch_users, batch_labels = shuffle(batch_items, batch_users, batch_labels)

        testing_query_y_pred = Recmodel(batch_items, batch_users, "test-set")
        test_output_list, query_recom_list = testing_query_y_pred.view(-1).sort(descending=True)

        test_prec10, test_ndcg10 = add_metric(query_recom_list, batch_labels, topn=10)
        total_prec_list10.append(test_prec10)
        total_ndcg_list10.append(test_ndcg10)
        test_prec8, test_ndcg8 = add_metric(query_recom_list, batch_labels, topn=8)
        total_prec_list8.append(test_prec8)
        total_ndcg_list8.append(test_ndcg8)
        test_prec5, test_ndcg5 = add_metric(query_recom_list, batch_labels, topn=5)
        total_prec_list5.append(test_prec5)
        total_ndcg_list5.append(test_ndcg5)

    total_prec10 = sum(total_prec_list10) / len(total_prec_list10)
    total_ndcg10 = sum(total_ndcg_list10) / len(total_ndcg_list10)
    total_prec8 = sum(total_prec_list8) / len(total_prec_list8)
    total_ndcg8 = sum(total_ndcg_list8) / len(total_ndcg_list8)
    total_prec5 = sum(total_prec_list5) / len(total_prec_list5)
    total_ndcg5 = sum(total_ndcg_list5) / len(total_ndcg_list5)


    return total_prec10, total_ndcg10, total_prec8, total_ndcg8, total_prec5, total_ndcg5




