
import numpy as np
import random
import torch


'''输入头部user_id的list，和尾部需要增强的user_id，输出最高相似分数和与尾部user_id最相似的头部top_user_id，
这里使用的相似度计算法是欧式距离：dist=1/(1+sqrt(pos_user-hist_user)^2)'''
def attention_scores(hist_list, pos_user):
    score_list = []
    for i in range(len(hist_list)):
        dist = torch.sqrt(torch.pow(torch.tensor(pos_user-hist_list[i]), 2))
        sim = 1/(1 + dist)
        score_list.append(sim.item())
    sim_scores = torch.tensor(score_list)
    _scores, recom_indexs = sim_scores.view(-1).sort(descending=True)

    return _scores[0].item(), recom_indexs[0].item()

'''得到增强第test_item_index[j]个item对应的user_list，原来有7个users，现在有14个，增加了1倍数量，
同时需要取得对应的负样本，一共28个user，是经过了gcn的embedding表示'''
def model_test_training(args, training_supp_dataset, testing_supp_dataset, Recmodel, bpr):
    """"训练集train-set进行3层gcn后得到的embedding表示,
    train_items_gcn=torch.Size([3846, 64])，
    train_users_gcn=torch.Size([1872, 64])"""
    train_items_gcn, train_users_gcn = Recmodel.computer_graph_embs("train-set")
    '''测试集test-set进行3层gcn后得到的embedding表示'''
    test_items_gcn, test_users_gcn = Recmodel.computer_graph_embs("test-set")

    train_allPos = training_supp_dataset.allPos  # 训练集training-set所有的正样本item_id
    test_allPos = testing_supp_dataset.allPos  # 测试集testing-set所有的正样本item_id
    item_num = testing_supp_dataset.n_item  # item的数量,这里是3846
    items = random.sample(range(0, testing_supp_dataset.n_item), item_num)  # 随机生成item_num个item_id，范围是（0, dataset.n_item）

    '''获取training-set中support-set的item:{user1,user2,...}'''
    train_all_user_dict = {}
    train_item_index = []
    for i, item in enumerate(items):
        train_pos_user = train_allPos[i]
        if len(train_pos_user) == 0:
            continue
        train_all_user_dict[i] = train_pos_user
        train_item_index.append(i)
    '''获取testing-set中support-set的item:{user1,user2,...}'''
    test_all_user_dict = {}
    test_item_index = []
    for m, item in enumerate(items):
        test_pos_user = test_allPos[m]
        if len(test_pos_user) == 0:
            continue
        test_all_user_dict[m] = test_pos_user
        test_item_index.append(m)

    total_loss_list = []

    for j in range(len(test_item_index)):   # 400次循环
        test_i = test_item_index[j]     # 对第一个item_id进行操作
        test_list = test_all_user_dict[test_i]
        test_list = test_list.tolist()
        test_users_gcn_emb = test_users_gcn[test_list]
        test_users_gcn_emb = test_users_gcn_emb.tolist()

        t_score, t_index = attention_scores(train_item_index, test_i)

        '''因为test_i的数量是train_i的4倍，所以用k = j % 100表示循环4次取train_i的index.
        这里进行了尾部数据增强操作，样本数量从7个增强到14个'''
        k = j % 100     # 使用k效果更差
        train_i = train_item_index[t_index]
        train_list = train_all_user_dict[train_i]
        sim_score_list = []
        for index in range(len(test_list)):
            test_pos_user = test_list[index]
            train_list_emb = train_users_gcn[train_list]

            sim_score, sim_index = attention_scores(train_list, test_pos_user)
            sim_score_list.append(round(sim_score, 4))      # 保留小数点后2位
            test_list.append(train_list[sim_index])
            test_users_gcn_emb.append((train_list_emb[sim_index]).tolist())


        neg_user_emb_list = []
        neg_user_list = []
        '''抽取负样本，14（正）+14（负）'''
        for t in range(len(test_list)):     # 给14个user补充负样本
            while True:
                neg_user = np.random.randint(0, testing_supp_dataset.m_user)
                if neg_user in test_list:
                    continue
                else:
                    break
            '''对补充后的正样本对应的负样本也要乘以一个sim_score'''
            neg_user_list.append(neg_user)
            neg_user_emb_list.append(test_users_gcn[neg_user].tolist())

        '''一共28个user的gcn的embedding表示，前14个为正样本，后14个为负样本，并没有打乱顺序'''
        pos_user_emb_list = torch.tensor(test_users_gcn_emb)
        neg_user_emb_list = torch.tensor(neg_user_emb_list)

        hist_user_emb_list = []
        test_i_list = []
        for n in range(len(test_users_gcn_emb)):
            test_i_list.append(test_i)
            temp_emb = test_users_gcn_emb.copy()
            temp_emb.remove(temp_emb[n])

            hist_emb = temp_emb
            hist_user_emb_list.append(hist_emb)

        item_emb_list = test_items_gcn[test_i_list]
        item_emb_ego = Recmodel.embedding_item(torch.tensor(test_i_list).to(args.mdevice))
        pos_emb_ego = Recmodel.embedding_user(torch.tensor(test_list).to(args.mdevice))
        neg_emb_ego = Recmodel.embedding_user(torch.tensor(neg_user_list).to(args.mdevice))

        hist_emb_list = torch.tensor(hist_user_emb_list)
        pos_emb_list = pos_user_emb_list
        neg_emb_list = neg_user_emb_list
        user_his_emb = Recmodel.attention_layer(hist_emb_list.to(args.mdevice), pos_emb_list.to(args.mdevice))


        reg_loss = (1 / 2) * (item_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2)
                              + neg_emb_ego.norm(2).pow(2)) / float(len(items))

        '''正样本得分'''
        pos_scores = torch.mul(item_emb_list.to(args.mdevice), pos_emb_list.to(args.mdevice)) + torch.mul(user_his_emb.to(args.mdevice), pos_emb_list.to(args.mdevice))
        pos_scores = torch.sum(pos_scores, dim=1)
        '''负样本得分'''
        neg_scores = torch.mul(item_emb_list.to(args.mdevice), neg_emb_list.to(args.mdevice)) + torch.mul(user_his_emb.to(args.mdevice), neg_emb_list.to(args.mdevice))
        neg_scores = torch.sum(neg_scores, dim=1)

        _loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = reg_loss * args.decay
        _loss = _loss + reg_loss

        bpr.opt.zero_grad()
        _loss.backward(retain_graph=True)
        bpr.opt.step()
        total_loss_list.append(_loss.item())

    aver_loss = sum(total_loss_list)/len(total_loss_list)

    return aver_loss