# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)] # initial user and index for whole dataset
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    """
    It takes the training data, the number of classes, the probability of a client choosing a class, the
    number of clients, and a seed, and returns a dictionary of clients and the indices of the training
    data they have
    
    :param y_train: the labels of the training data
    :param num_classes: number of classes in the dataset
    :param p: the probability of a client choosing a class
    :param num_users: number of clients
    :param seed: random seed
    :param alpha_dirichlet: the parameter for the dirichlet distribution, defaults to 100 (optional)
    :return: A dictionary of the users and the classes they have been assigned to.
    """
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    # pis = np.copy(Phi)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
    # print(pis-Phi)
    # print(Phi)
    # print(Psi[0])
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]
        #print(all_idxs)
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        #print(num_clients_per_class,'\n')
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())
        #print(p_dirichlet.tolist(),'\n',assignment,'\n')
        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
                #print(set(all_idxs[(assignment==client_k)]),'\n')
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
                #print(set(all_idxs[(assignment==client_k)]),'\n') 
    return dict_users