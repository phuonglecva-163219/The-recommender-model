import json

import tensorflow as tf
import numpy as np
from model import MyModel


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    user_hist_dict = data[9]
    max_entity = data[10]
    max_relation = data[11]
    item_hist_dict = data[12]
    adj_entity_exp_score = data[13]
    model = MyModel(args, n_user, n_entity, n_relation, adj_entity, adj_relation, user_hist_dict, max_entity, item_hist_dict, adj_entity_exp_score)
    saver = tf.train.Saver()
    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item, n_user)
    user_list, train_record, test_record, item_set, k_list = topk_settings(True, train_data, test_data,
                                                                           n_item, n_user)

    print(user_list)

    if args.use_pretrained:
        with tf.Session() as sess:
            user_list, train_record, test_record, item_set, k_list = topk_settings(True, train_data, test_data,
                                                                                   n_item, n_user)
            saver.restore(sess, '/home/phuonglc/project/model.ckpt')
            # items, scores = model.get_scores(sess, {
            #     model.user_indices: [1 + max_entity + 1] * args.batch_size,
            #     model.item_indices: [i for i in range(args.batch_size)]
            # })
            # print(items)
            # print(scores)
            user = 0
            item_list = get_scores_for_user(sess, model, user, train_record, test_record, item_set, k_list, args.batch_size, max_entity)
            print(item_list)
        return

    import os.path
    if os.path.exists('../checkpoints/checkpoint_1'):
        with tf.Session() as sess:
            saver.restore(sess, '../checkpoints/model.ckpt')
            print('model is  loaded.')
            res = {}
            for user in user_list[:]:
                item_list = get_scores_for_user(sess, model, user, train_record, test_record, item_set, k_list,
                                                args.batch_size, max_entity)
                res[str(user)] = [[int(item[0]), float(item[-1])] for item in item_list]
            print(res)
            with open('predictions_lastfm.json', 'w') as f:
                json.dump(res, f, indent=2)
                print('file localtion: {}'.format(os.path.realpath(f.name)))
    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(args.n_epochs):
                # training
                np.random.shuffle(train_data)
                start = 0
                # skip the last incomplete minibatch if its size < batch size
                while start + args.batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(model, train_data, user_hist_dict, start, start + args.batch_size, max_entity, max_relation))
                    start += args.batch_size
                    if show_loss:
                        print(start, loss)

                # CTR evaluation
                train_auc, train_f1, acc1 = ctr_eval(sess, model, train_data, args.batch_size, user_hist_dict, max_entity, max_relation)
                eval_auc, eval_f1, acc2 = ctr_eval(sess, model, eval_data, args.batch_size, user_hist_dict, max_entity, max_relation)
                test_auc, test_f1, acc3 = ctr_eval(sess, model, test_data, args.batch_size, user_hist_dict, max_entity, max_relation)

                print('epoch %d    train auc: %.4f  f1: %.4f acc: %.4f   eval auc: %.4f  f1: %.4f  acc: %.4f   test auc: %.4f  f1: %.4f acc: %.4f '
                      % (step, train_auc, train_f1, acc1, eval_auc, eval_f1, acc2, test_auc, test_f1, acc3))

                # top-K evaluation
                if show_topk:
                    precision, recall = topk_eval(
                        sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size, max_entity)
                    print('precision: ', end='')
                    for i in precision:
                        print('%.4f\t' % i, end='')
                    print()
                    print('recall: ', end='')
                    for i in recall:
                        print('%.4f\t' % i, end='')
                    print('\n')

            saver.save(sess, '../checkpoints/model.ckpt')
            # res = {}
            # for user in user_list[:]:
            #     item_list = get_scores_for_user(sess, model, user, train_record, test_record, item_set, k_list,
            #                                     args.batch_size, max_entity)
            #     res[str(user)] = [[int(item[0]), float(item[-1])] for item in item_list]
            # print(len(res.keys()))
            # import os
            # filename = r"C:\Users\phuon\workspace\webapp\movie_recommendation_web\backend\data\predictions.json"
            # if os.path.exists(filename):
            #     os.remove(filename)
            #
            # print('start writing....')
            import time
            # time.sleep(5)
            # with open(filename, 'w') as f:
            #     json.dump(res, f, indent=2)
            #     print('file localtion: {}'.format(os.path.realpath(f.name)))



def topk_settings(show_topk, train_data, test_data, n_item, n_user):
    if show_topk:
        # user_num = 100
        k_list = [2, 5, 10, 20, 50]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        print("len userlist: %s" %(len(user_list)))
        # if len(user_list) > user_num:
        #     user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, user_hist_dict, start, end, max_entity, max_relation):
    user_indices = data[start:end, 0]
    user_indices = [user + max_entity + 1 for user in user_indices]
    feed_dict = {model.user_indices: user_indices,
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 }
    return feed_dict


def ctr_eval(sess, model, data, batch_size, user_hist_dict, max_entity, max_relation):
    start = 0
    auc_list = []
    f1_list = []
    acc_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1, acc = model.eval(sess, get_feed_dict(model, data, user_hist_dict, start, start + batch_size, max_entity, max_relation))
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(acc_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size, max_entity):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user + max_entity + 1] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user + max_entity + 1] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall

def get_scores_for_user(sess, model, user, train_record, test_record, item_set, k_list, batch_size, max_entity):
    test_item_list = list(item_set - train_record[user])
    item_score_map = dict()
    start = 0
    while start + batch_size <= len(test_item_list):
        items, scores = model.get_scores(sess, {model.user_indices: [user + max_entity + 1] * batch_size,
                                                model.item_indices: test_item_list[start:start + batch_size]})

        for item, score in zip(items, scores):
            item_score_map[item] = score
        start += batch_size

    # padding the last incomplete minibatch if exists
    if start < len(test_item_list):
        items, scores = model.get_scores(
            sess, {model.user_indices: [user + max_entity + 1] * batch_size,
                   model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                           batch_size - len(test_item_list) + start)})
        for item, score in zip(items, scores):
            item_score_map[item] = score

    item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
    item_sorted = [i[0] for i in item_score_pair_sorted]
    return item_score_pair_sorted[:10]

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def create_model(args, data):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    user_hist_dict = data[9]
    max_entity = data[10]
    max_relation = data[11]
    item_hist_dict = data[12]
    adj_entity_exp_score = data[13]
    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation, user_hist_dict, max_entity,
                 item_hist_dict, adj_entity_exp_score)
    return model