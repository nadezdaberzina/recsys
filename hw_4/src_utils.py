import numpy as np
import pandas as pd


def prefilter_items(data, item_features):

    test_size_weeks = 3
    data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
    
    # Уберем самые популярные товары (их и так купят)
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['share_unique_users'] = popularity['user_id'] / (data_train['user_id'].nunique())
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    not_recent = data_train[data_train['day'] < data_train['day'].max() - 365].item_id.tolist()
    data = data[~data['item_id'].isin(not_recent)]
    
    # Уберем не интересные для рекоммендаций категории (department)
    departments_to_exclude = item_features.groupby('department')['item_id'].nunique().reset_index()
    departments_to_exclude.rename(columns={'item_id' : 'number_of_items'}, inplace=True)
    departments_to_exclude = departments_to_exclude[departments_to_exclude['number_of_items'] < 100].department.tolist()
    
    items_to_exclude = item_features[item_features['department'].isin(departments_to_exclude)].item_id.tolist()
    data = data[~data['item_id'].isin(items_to_exclude)]
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data_train[data_train['quantity'] == 0] = 1
    data_train['price'] = data_train['sales_value'] / data_train['quantity']
    
    not_profitable = data_train[data_train['price'] < 40].item_id.tolist()
    data = data[~data['item_id'].isin(not_profitable)]
    
    # Уберем слишком дорогие товары

    too_expensive = data_train[data_train['price'] > 70].item_id.tolist()
    data = data[~data['item_id'].isin(too_expensive)]
    
    return data


def get_similar_items_recommendation(user, model, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    
    bought_items = data_train.loc[data_train['user_id'] == 2500]
    bought_items = bought_items.groupby(['item_id'])['quantity'].count().reset_index()
    
    bought_items.sort_values('quantity', ascending=False, inplace=True)
    
    bought_items = bought_items[bought_items['item_id'] != 999999].head(5)
    
    bought_items = bought_items['item_id'].tolist()

    res = []

    for item in bought_items :
        rec = model.similar_items(itemid_to_id[item], N=2)
        top_rec = rec[1][0]
        res.append(id_to_itemid[top_rec])
    
    return res


def get_similar_users_recommendation(user, model, N=5):

    users = model.similar_users(userid_to_id[user], N=6)
    similar_user, score = list(zip(*users))
    similar_users = similar_user[1:]
    
    own = ItemItemRecommender(K=1, num_threads=4) # K - кол-во билжайших соседей

    own.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=True)

    res = []

    for user in similar_users:
    
        rec = own.recommend(user, 
                        user_items=csr_matrix(user_item_matrix).tocsr(),   # на вход user-item matrix
                        N=1, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)
    
        res.append(id_to_itemid[rec[0][0]])
    
    return res
