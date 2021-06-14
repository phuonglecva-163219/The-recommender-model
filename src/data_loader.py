import numpy as np
import os
# import matplotlib.pyplot as plt

def similarity(a_vector, u_vector):
    score = np.abs(np.corrcoef(a_vector, u_vector))
    return score[0][1]


def build_user_similarity(user_hist_dict, n_user, n_item):
    ratio = 1
    num_user_added = np.random.choice(n_user, int(n_user * ratio))
    user_similarity = [None] * len(num_user_added)
    index = 0
    for user in num_user_added:
        user_vector = [0] * n_item
        for i in user_hist_dict[user]:
            if i in range(n_item):
                user_vector[i] = 1
        user_similarity[index] = user_vector
        index += 1
    user_similarity = np.abs(np.corrcoef(user_similarity))
    user_similarity = user_similarity.tolist()
    res = [[i for i in range(len(user)) if user[i] >= 0.2] for user in user_similarity]
    count = 0
    for user in res:
        count += len(user)
    print('num user added: {}'.format(count - n_user))
    return res


def build_item_similarity(item_history_dict, n_item, n_user):
    ratio = 1
    item_indices = np.random.choice(n_item, int(n_item * ratio))
    item_users = list(item_history_dict.items())
    item_users = [item_users[i] for i in item_indices]
    map_idx_to_item = [None] * len(item_users)
    for i in range(len(item_users)):
        item = item_users[i][0]
        map_idx_to_item[i] = item

    # list_item = list(item_history_dict.keys())
    item_sim = [[]] * len(item_users)
    index = 0
    for item in map_idx_to_item:
        item_vector = [0] * n_user
        for user in item_history_dict[item]:
            if user  == -1:
                continue
            item_vector[user] = 1
        item_sim[index] = item_vector
        index += 1
    item_sim = np.abs(np.corrcoef(item_sim))
    item_sim = list(map(lambda row: np.where(row >= 0.005)[0], item_sim))
    count = 0
    for item in item_sim:
        count += len(item)
    print('num item relation added: {}'.format(count - n_item))
    return item_sim, map_idx_to_item

def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, user_hist_dict, item_hist_dict, item_history_dict = load_rating(args)
    user_similarity = build_user_similarity(user_hist_dict, n_user, n_item)
    item_similarity, map_idx_to_item = build_item_similarity(item_history_dict, n_item, n_user)
    n_entity, n_relation, adj_entity, adj_relation, max_entity, max_relation, adj_entity_exp_score = load_kg(args, user_hist_dict, n_user, n_item, user_similarity, item_similarity, map_idx_to_item)

    item_hist_dict = construct_items_score(args, item_hist_dict, n_item)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, user_hist_dict, max_entity, max_relation, item_hist_dict, adj_entity_exp_score


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    print(n_user, n_item)
    user_history_dict = dict()
    item_history_dict = dict()
    train_data, eval_data, test_data = dataset_split(rating_np, args)
    for i in range(len(train_data)):
        user = train_data[i][0]
        item = train_data[i][1]
        rating = train_data[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

            if item not in item_history_dict:
                item_history_dict[item] = []
            item_history_dict[item].append(user)
    for i in range(len(rating_np)):
        user = rating_np[i][0]
        if user not in user_history_dict:
            user_history_dict[user] = [-1]
    for i in range(len(rating_np)):
        item = rating_np[i][1]
        if item not in item_history_dict:
            item_history_dict[item] = [-1]

    item_hist_dict = {}
    for row in rating_np:
        item = row[1]
        item_hist_dict[item] = 0

    for row in train_data:
        item = row[1]
        feedback = row[2]
        if row[2] == 1:
            if item not in item_hist_dict:
                item_hist_dict[item] = 0
            item_hist_dict[item] += 1
    item_hist_dict = {item: np.exp(item_hist_dict[item]) for item in item_hist_dict}
    return n_user, n_item, train_data, eval_data, test_data, user_history_dict, item_hist_dict, item_history_dict


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    # traverse training data, only keeping the users with positive ratings


    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args, user_hist_dict, n_user, n_item, user_similarity, item_similarity, map_idx_to_item):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    # max_entity, max_relation
    max_entity = max(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    max_relation = max(set(kg_np[:, 1]))

    kg = construct_kg(kg_np, user_hist_dict, max_relation, max_entity, user_similarity, item_similarity, map_idx_to_item)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity, n_user, n_item)
    adj_entity_score = construct_entity_score(kg)

    adj_entity_exp_score = np.zeros([len(kg), 1], dtype=np.float32)
    for entity, score in adj_entity_score.items():
        adj_entity_exp_score[entity] = score

    return n_entity, n_relation, adj_entity, adj_relation, max_entity, max_relation, adj_entity_exp_score


def construct_kg(kg_np, user_hist_dict, max_relation, max_entity, user_similarity, item_similarity, map_idx_to_item):
    print('constructing knowledge graph ...')
    kg = dict()

    unknown_item = max_entity + len(user_hist_dict)
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    for user_id, list_items in user_hist_dict.items():
        like_relation = max_relation + 1
        user_relation = like_relation + 1
        user_new_id = user_id + max_entity + 1
        for item in list_items:
            if item == -1:
                item = unknown_item
            if user_new_id not in kg:
                kg[user_new_id] = []

            kg[user_new_id].append((item, like_relation))

            if item not in kg:
                kg[item] = []
            kg[item].append((user_new_id, like_relation))
        user_list = user_similarity[user_id]
        for user_high_sim in user_list:
            if user_high_sim == user_id:
                continue
            user_high_sim_new_id = user_high_sim + max_entity + 1
            kg[user_new_id].append((user_high_sim_new_id, user_relation))

            if user_high_sim_new_id not in kg:
                kg[user_high_sim_new_id] = []
            kg[user_high_sim_new_id].append((user_new_id, user_relation))

    for item in range(len(item_similarity)):
        item_id = map_idx_to_item[item]
        item_list = item_similarity[item]
        for another in item_list:
            if item_id == another:
                continue
            kg[item_id].append((another, max_relation + 3))
            kg[another].append((item_id, max_relation + 3))

    return kg


def construct_adj(args, kg, entity_num, n_user, n_item):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num + n_user + 1, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num + n_user + 1, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num + n_user):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation

def construct_items_score(args, item_hist_dict, n_item):
    adj_item_score = np.zeros([n_item, 1], dtype = np.float32)
    for item, score in item_hist_dict.items():
        adj_item_score[item] = score

    return adj_item_score

def construct_entity_score(kg):
    adj_entity_score = {item: np.exp(len(value)) for item, value in kg.items()}
    return adj_entity_score
