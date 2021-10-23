import numpy as np
import pandas as pd


def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    hit_rate_at_k = (flags.sum() > 0) * 1
    
    return hit_rate_at_k


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    precision_at_k = flags.sum() / len(recommended_list)
    
    return precision_at_k


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    money_precision_at_k = (np.inner(flags, prices_recommended)) / prices_recommended.sum()
    
    
    return money_precision_at_k


def recall_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    recall_at_k = flags.sum() / len(bought_list)
    
    return recall_at_k


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)
    
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    money_recall_at_k = (np.inner(flags, prices_recommended)) / prices_bought.sum()
    
    return money_recall_at_k


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        
        if flags[i-1] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / k
    
    return result


def map_k(recommended_list, bought_list, k=5):
    
    sum_ap_k = 0
    for user in users:
        
        recommended_list = user['recommended_list']
        recommended_list = np.array(recommended_list)
        recommended_list = recommended_list[:k]
        
        bought_list = user['bought_list']
        bought_list = np.array(bought_list)
    
        flags = np.isin(recommended_list, bought_list)
    
        if sum(flags) == 0:
            return 0
    
        sum_ = 0
        for i in range(1, k+1):
        
            if flags[i-1] == True:
                p_k = precision_at_k(recommended_list, bought_list, k=i)
                sum_ += p_k
            
            ap_k = sum_ / k
            
        sum_ap_k += ap_k
        
    result = sum_ap_k / len(users)
            
    return result


def dcg_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    if flags[0] == True:
        sum_ += 1
        
    for i in range(2, k+1):
        
        if flags[i-1] == True:
            gain = 1 / np.log2(i)
            sum_ += gain
    
    result = sum_ / k
    
    return result


def idcg_k(k=5):
    
    sum_ = 1
        
    for i in range(2, k+1):
        
        gain = 1 / np.log2(i)
        sum_ += gain
    
    result = sum_ / k
    
    return result


def ndcg(recommended_list, bought_list, k=5):
    
    result = dcg_k(recommended_list, bought_list, k=5) / idcg_k(k=5)
    
    return result


def reciprocal_rank(recommended_list, bought_list, k=5):
    
    sum_rank = 0
    
    for user in users:
        
        recommended_list = user['recommended_list']
        recommended_list = np.array(recommended_list)
        
        bought_list = user['bought_list']
        bought_list = np.array(bought_list)
    
        flags = np.isin(recommended_list, bought_list)
        
        for i in range(1, k+1):
        
            if flags[i-1] == True:
                rank = 1 / (i)
                sum_rank += rank
        
    result = sum_rank / len(users)
            
    return result
