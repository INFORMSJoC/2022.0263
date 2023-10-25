import numpy as np
import sys
sys.path.append(r'C:\Users\user\workspace\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

import os

import pandas as pd
import argparse
from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt

from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
# import seaborn as sns
import random
from scipy.stats import loguniform
# from butools.fitting import *
from datetime import datetime
# from fastbook import *
import torch
import itertools
from scipy.special import factorial
import pickle as pkl


def compute_R(lam, alph, T):
    e = torch.ones((T.shape[0], 1))
    return np.array(lam * torch.inverse(lam * torch.eye(T.shape[0]) - lam * e @ alph - T))


def compute_pdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_pdf(x, s, A).flatten())

    return pdf_list

def compute_cdf_within_range(x_vals, s, A):
    pdf_list = []
    for x in x_vals:
        pdf_list.append(compute_cdf(x, s, A).flatten())

    return pdf_list

def compute_pdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return np.dot(np.dot(s, expm(A * x)), A0)


def compute_cdf(x, s, A):
    '''
    x: the value of pdf
    s: inital probs
    A: Generative matrix
    '''
    A0 = -np.dot(A, np.ones((A.shape[0], 1)))
    return 1 - np.sum(np.dot(s, expm(A * x)))


def steady_i(rho, alph, R, i):
    return (1 - rho) * alph @ matrix_power(R, i)


def create_gen_erlang_given_sizes(group_sizes, rates, probs=False):
    ph_size = np.sum(group_sizes)
    erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes)]
    final_a = np.zeros((ph_size, ph_size))
    final_s = np.zeros(ph_size)
    if type(probs) == bool:
        rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
        rands = np.random.rand(group_sizes.shape[0])
        rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
    else:
        rand_probs = probs
    for ind in range(group_sizes.shape[0]):
        final_s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        final_a[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = erlang_list[ind]

    return final_s, final_a


def create_gen_erlang_many_ph(max_ph_size = 500):
    ph_size = np.random.randint(1, max_ph_size)
    num_groups = np.random.randint(2,20)
    group_sizes = np.random.randint(1,25,num_groups)
    group_sizes_1 = (group_sizes*ph_size/np.sum(group_sizes)).astype(int)+1
    rates = ((np.ones(num_groups)*np.random.uniform(1, 1.75))**np.arange(num_groups))
    s,A = create_gen_erlang_given_sizes(group_sizes_1, rates)

    A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)


def ser_moment_n(s, A, mom):
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False

def compute_first_n_moments(s, A, n=3):
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


def create_erlang_row(rate, ind, size):
    aa = np.zeros(size)
    aa[ind] = -rate
    if ind < size - 1:
        aa[ind + 1] = rate
    return aa


def create_row_rates(row_ind, is_absorbing, in_rate, non_abrosing_out_rates, ph_size, non_absorbing):
    '''
    row_ind: the current row
    is_abosboing: true if it an absorbing state
    in_rate: the rate on the diagonal
    non_abrosing_out_rates: the matrix with non_abrosing_out_rates
    ph_size: the size of phase type
    return: the ph row_ind^th of the ph matrix
    '''

    finarr = np.zeros(ph_size)
    finarr[row_ind] = -in_rate  ## insert the rate on the diagonal with a minus sign
    if is_absorbing:  ## no further changes is requires
        return finarr
    else:
        all_indices = np.arange(ph_size)
        all_indices = all_indices[all_indices != row_ind]  ## getting the non-diagonal indices
        rate_ind = np.where(non_absorbing == row_ind)  ## finding the current row in non_abrosing_out_rates
        finarr[all_indices] = non_abrosing_out_rates[rate_ind[0][0]]
        return finarr

def generate_erlang_given_rates(rate, ph_size):
    A = np.identity(ph_size)
    A_list = [create_erlang_row(rate, ind, ph_size) for ind in range(ph_size)]
    A = np.concatenate(A_list).reshape((ph_size, ph_size))
    return A

def gives_rate(states_inds, rate, ph_size):
    '''
    states_ind: the out states indices
    rate: the total rate out
    return: the out rate array from that specific state
    '''
    final_rates = np.zeros(ph_size - 1)  ## initialize the array
    rands_weights_out_rate = np.random.rand(states_inds.shape[0])  ## Creating the weights of the out rate
    ## Computing the out rates
    final_rates[states_inds] = (rands_weights_out_rate / np.sum(rands_weights_out_rate)) * rate
    return final_rates

def give_s_A_given__fixed_size(ph_size, scale_low, scale_high):
    if ph_size > 1:
        potential_vals = np.linspace(scale_low, scale_high, 20000)
        randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
        ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
        w = np.random.rand(ph_size)
        numbers = np.arange(0, ph_size + 1)  # an array from 0 to ph_size + 1
        p0 = 0.9
        distribution = (w / np.sum(w)) * (1 - p0)  ## creating a pdf from the weights of w
        distribution = np.append(p0, distribution)
        random_variable = rv_discrete(values=(numbers, distribution))  ## constructing a python pdf
        ww = random_variable.rvs(size=1)

        ## choosing the states that are absorbing
        absorbing_states = np.sort(np.random.choice(ph_size, ww[0], replace=False))
        non_absorbing = np.setdiff1d(np.arange(ph_size), absorbing_states, assume_unique=True)

        N = ph_size - ww[0]  ## N is the number of non-absorbing states
        p = np.random.rand()  # the probability that a non absorbing state is fully transient
        mask_full_trans = np.random.choice([True, False], size=N, p=[p, 1 - p])  # True if row sum to 0
        if np.sum(mask_full_trans) == mask_full_trans.shape[0]:
            mask_full_trans = False
        ser_rates = ser_rates.flatten()

        ## Computing the total out of state rate, if absorbing, remain the same
        p_outs = np.random.rand(N)  ### this is proportional rate out
        orig_rates = ser_rates[non_absorbing]  ## saving the original rates
        new_rates = orig_rates * p_outs  ## Computing the total out rates
        out_rates = np.where(mask_full_trans, orig_rates, new_rates)  ## Only the full trans remain as the original

        ## Choosing the number of states that will have a postive rate out for every non-absorbing state

        num_trans_states = np.random.randint(1, ph_size, N)

        ## Choosing which states will go from each non-absorbing state
        trans_states_list = [np.sort(np.random.choice(ph_size - 1, num_trans_states[j], replace=False)) for j in
                             range(N)]
        # Computing out rates
        non_abrosing_out_rates = [gives_rate(trans_states, out_rates[j], ph_size) for j, trans_states in
                                  enumerate(trans_states_list)]
        ## Finalizing the matrix

        #     return trans_states_list, absorbing_states, ser_rates, non_abrosing_out_rates
        lists_rate_mat = [
            create_row_rates(row_ind, row_ind in absorbing_states, ser_rates[row_ind], non_abrosing_out_rates, ph_size,
                             non_absorbing) for row_ind in range(ph_size)]
        A = np.concatenate(lists_rate_mat).reshape((ph_size, ph_size))  ## converting all into one numpy array

        num_of_pos_initial_states = np.random.randint(1, ph_size + 1)
        non_zero_probs = np.random.dirichlet(np.random.rand(num_of_pos_initial_states), 1)
        inds_of_not_zero_probs = np.sort(np.random.choice(ph_size, num_of_pos_initial_states, replace=False))
        s = np.zeros(ph_size)
        s[inds_of_not_zero_probs] = non_zero_probs

    else:
        s = np.array([1.])
        potential_vals = np.linspace(scale_low, scale_high, 20000)
        randinds = np.random.randint(potential_vals.shape[0], size=ph_size)
        ser_rates = (potential_vals[randinds]).reshape((1, ph_size))
        A = -ser_rates

    return (s, A)

def balance_sizes(sizes):
    for ind in range(sizes.shape[0]):
        if sizes[ind] < 3:
            ind_max = np.argmax(sizes)
            if sizes[ind_max] >2 :
                sizes[ind] +=1
                sizes[ind_max] -=1
    return sizes

def recursion_group_size(group_left, curr_vector, phases_left):
    if group_left == 1:
        return np.append(phases_left, curr_vector)
    else:

        if phases_left + 1 - group_left == 1:
            curr_size = 1
        else:
            curr_size =  1+ np.random.binomial(phases_left + 1 - group_left-1, np.random.uniform(0.1,0.5))
        return recursion_group_size(group_left - 1, np.append(curr_size, curr_vector), phases_left - curr_size)

def create_mix_erlang_ph(ph_size, scale_low=1, max_scale_high=15, max_ph=500):

    if ph_size > 2:
        ph_size_gen_ph = np.random.randint(2, ph_size)

    else:
        return create_gen_erlang_many_ph(ph_size)

    erlang_max_size = np.random.randint(int(0.25 * max_ph), int(0.75 * max_ph))

    scale_high = np.random.uniform(2, max_scale_high)
    # ph_size_gen_ph = np.random.randint(5, max_ph - erlang_max_size)
    # if int(0.5*ph_size_gen_ph ) > 1:

    #     num_groups = np.random.randint(1,  int(0.5*ph_size_gen_ph) )
    # else:
    #     num_groups = 1
    num_groups = sample_num_groups(ph_size_gen_ph)


    # group_sizes = np.random.randint(1, 25, num_groups)

    group_sizes_gen_ph = recursion_group_size(num_groups, np.array([]), ph_size_gen_ph) #(group_sizes * ph_size_gen_ph / np.sum(group_sizes)).astype(int) + 1
    if np.random.rand()>0.01:
        group_sizes_gen_ph = balance_sizes(group_sizes_gen_ph)
    erlang_list_gen_ph = [give_s_A_given__fixed_size(size, scale_low, scale_high) for size in group_sizes_gen_ph.astype(int)]
    erlang_list_gen_ph_A = [lis[1] for lis in erlang_list_gen_ph]
    erlang_list_gen_ph_s = [lis[0] for lis in erlang_list_gen_ph]

    ph_size_erl = ph_size - ph_size_gen_ph #np.random.randint(5, erlang_max_size)
    # if ph_size_erl > 2:
    #     num_groups = np.random.randint(1, min(7, ph_size_erl - 1))
    # else:
    #     num_groups = 1
    num_groups = sample_num_groups(ph_size_erl)


    # group_sizes = recursion_group_size(num_groups, np.array([]), ph_size_erl).astype(int)  #np.random.randint(1, 25, num_groups)
    if np.random.rand() > 0.8:
        rates = np.random.rand(num_groups)*200   #((np.ones(num_groups) * np.random.uniform(1, 1.75)) ** np.arange(num_groups))
    else:
        rates = np.random.uniform(1, 1.75) ** (np.random.rand(num_groups) * 10)
    group_sizes_erl = recursion_group_size(num_groups, np.array([]), ph_size_erl).astype(int) # (group_sizes * ph_size_erl / np.sum(group_sizes)).astype(int) + 1
    if np.random.rand()>0.01:
        group_sizes_erl = balance_sizes(group_sizes_erl)
    erlang_list_erl = [generate_erlang_given_rates(rates[ind], ph_size_erl) for ind, ph_size_erl in
                       enumerate(group_sizes_erl)]
    group_sizes = np.append(group_sizes_gen_ph, group_sizes_erl)
    group_sizes = group_sizes.astype(int)
    rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)

    ph_list = erlang_list_gen_ph_A + erlang_list_erl

    ph_size = np.sum(group_sizes)
    A = np.zeros((int(ph_size), int(ph_size)))
    s = np.zeros(int(ph_size))
    for ind in range(group_sizes.shape[0]):
        if ind < group_sizes_gen_ph.shape[0]:
            s[int(np.sum(group_sizes[:ind])):int(np.sum(group_sizes[:ind]) + group_sizes[ind])] = rand_probs[0][ind] * \
                                                                                        erlang_list_gen_ph_s[ind]
        else:
            s[np.sum(group_sizes[:ind])] = rand_probs[0][ind]  # 1/diff_list.shape[0]
        A[np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind],
        np.sum(group_sizes[:ind]):np.sum(group_sizes[:ind]) + group_sizes[ind]] = ph_list[ind]

    fst_mom = compute_first_n_moments(s, A, 1)
    if type(fst_mom[0]) != bool:
        A = A * fst_mom[0][0]
        fst_mom = compute_first_n_moments(s, A, 1)
        return (s, A)

    else:
        return False



def create_gen_erlang_many_ph(ph_size):
    # ph_size = np.random.randint(1, max_ph_size)

    num_groups = sample_num_groups(ph_size)


    # if ph_size > 1:
    #     num_groups = np.random.randint(2,min(8,ph_size))
    # else:
    #     num_groups = 1
    group_sizes_1 = recursion_group_size(num_groups, np.array([]), ph_size).astype(int)
    if np.random.rand()>0.01:
        group_sizes_1 = balance_sizes(group_sizes_1)
    rates = np.random.uniform(1, 1.75)**(np.random.rand(num_groups)*10) # ((np.ones(num_groups)*np.random.uniform(1, 1.85))**np.arange(num_groups))
    s,A = create_gen_erlang_given_sizes(group_sizes_1, rates)

    A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)

def sample_num_groups(n, thresh =0.98):
    if np.random.rand()>thresh:
        num = 1+np.random.binomial(n-1, np.random.uniform(0.2,0.99))
    elif np.random.rand()>0.9:
        num = 1+np.random.binomial(int(n*0.1), np.random.uniform(0.3,0.87))
    else:
        if n<10:
            portion = 0.3
        else:
            portion = 0.8
        num = 1+np.random.binomial(min(10,int(n-1)*portion), np.random.uniform(0.1,0.9))
    if (num==1) & (n>1 ) &(np.random.rand()>0.4):
        num +=1
    return num

def create_Erlang_given_ph_size(ph_size):
    s = np.zeros(ph_size)
    s[0] = 1
    rate = ph_size
    A = generate_erlang_given_rates(rate, ph_size)
    # A = A*compute_first_n_moments(s, A, 1)[0][0]
    return (s,A)

def send_to_the_right_generator(num_ind, ph_size):

    if num_ind == 1: ## Any arbitrary ph
        s_A =  create_mix_erlang_ph(ph_size) # give_s_A_given_size(np.random.randint(60, max_ph_size))
    elif num_ind > 1:
        s_A = create_gen_erlang_many_ph(ph_size)
    else:
        s_A = create_Erlang_given_ph_size(ph_size)
    if type(s_A) != bool:
        try:

            s = s_A[0]
            A = s_A[1]

            return (s,A)

        except:
            print('Not able to extract s and A')


def kroneker_sum(G,H):
    size_g = G.shape[0]
    size_h = H.shape[0]
    return np.kron(G, np.identity(size_h)) + np.kron( np.identity(size_g),H)

def compute_steady(s_arrival, A_arrival, s_service, A_service, y_size=500, eps=0.000001):

    inter_arrival_expected = ser_moment_n(s_arrival, A_arrival, 1)
    inter_service_expected = ser_moment_n(s_service, A_service, 1)

    A_service0 = -np.dot(A_service, np.ones((A_service.shape[0], 1)))
    A_arrival0 = -np.dot(A_arrival, np.ones((A_arrival.shape[0], 1)))

    A0 = A_arrival
    A1 = np.kron(np.identity(A_arrival.shape[0]), A_service0)
    A = kroneker_sum(np.zeros(A_arrival.shape), np.dot(A_service0, s_service))
    B = kroneker_sum(A_arrival, A_service)
    C = kroneker_sum(np.dot(A_arrival0, s_arrival), np.zeros((A_service.shape[0], A_service.shape[0])))
    C0 = np.kron(np.dot(A_arrival0, s_arrival), s_service)

    R = QBDFundamentalMatrices(A, B, C, "R")

    rho = (ser_moment_n(s_service, A_service, 1) / ser_moment_n(s_arrival, A_arrival, 1))[0][0]

    A0T = A0.transpose()
    A1T = A1.transpose()
    C0T = C0.transpose()
    BRAT = np.array(B + np.dot(R, A)).transpose()

    eqns = np.concatenate((np.concatenate((A0T, A1T), axis=1), np.concatenate((C0T, BRAT), axis=1)), axis=0)[:-1, :]

    sys_size = \
    np.concatenate((np.concatenate((A0T, A1T), axis=1), np.concatenate((C0T, BRAT), axis=1)), axis=0)[:-1, :].shape[1]
    u0_size = A0.shape[0]
    u0_eq = np.zeros((1, sys_size))
    u0_eq[0, :u0_size] = 1

    tot_eqns = np.concatenate((eqns, u0_eq), axis=0)

    u0 = 1 - rho
    B = np.zeros(sys_size)
    B[-1] = u0

    X = np.linalg.solve(tot_eqns, B)

    steady = np.zeros(y_size)
    steady[0] = np.sum(X[:A0.shape[0]])
    steady[1] = np.sum(X[A0.shape[0]:])
    tot_sum = np.sum(X)
    for ind in tqdm(range(2, y_size-1)):
        steady[ind] = np.sum(np.dot(X[u0_size:], matrix_power(R, ind - 1)))
        if np.sum(steady) > 1 - eps:
            break

    steady = np.append(steady, 1 - np.sum(steady))


    return steady




def compute_y_moms(s,A,num_moms,max_ph_size):


    lam_vals = np.random.uniform(0.8, 0.99, 1)


    lam_y_list = []

    for lam in lam_vals:
        x = create_final_x_data(s, A, lam)

        y = compute_y_data_given_folder(x, x.shape[0] - 1, tot_prob=70, eps=0.0001)
        if type(y) == np.ndarray:
            moms = compute_first_n_moments(s, A, num_moms)

            mom_arr = np.concatenate(moms, axis=0)

            lam = x[0, x.shape[0] - 1]


            mom_arr = np.log(mom_arr)
            mom_arr = np.delete(mom_arr, 0)
            mom_arr = np.append(lam, mom_arr)

            if not np.any(np.isinf(mom_arr)):

                lam_y_list.append((mom_arr, y))

    return lam_y_list

def compute_y_data_given_folder(x, ph_size_max, tot_prob=70, eps=0.0001):

    try:
        lam = x[0, ph_size_max].item()
        A = x[:ph_size_max, :ph_size_max]
        s = x[ph_size_max, :ph_size_max].reshape((1, ph_size_max))
        expect_ser = ser_moment_n(s, A, 1)
        if expect_ser:
            rho = lam * expect_ser[0][0]

            R = compute_R(lam, s, A)

            steady_state = np.array([1 - rho])
            for i in range(1, tot_prob):
                steady_state = np.append(steady_state, np.sum(steady_i(rho, s, R, i)))

            steady_state = np.append(steady_state, 1 - np.sum(steady_state))
            return steady_state

            # if np.sum(steady_state) > 1 - eps:
            #     return steady_state
            # else:
            #     return False

    except:
        print("x is not valid")


def create_final_x_data(s, A, lam):

    lam_arr = np.zeros((A.shape[0] + 1, 1))

    s1 = s.reshape((1, s.shape[0]))
    expect_ser = ser_moment_n(s, A, 1)
    if expect_ser:

        lam_arr[0, 0] = lam


        return np.append(np.append(A, s1, axis=0), lam_arr, axis=1).astype(np.float32)

def saving_batch(x_y_data, data_path, data_sample_name, num_moms, save_x = False):
    '''

    :param x_y_data: the data is a batch of tuples: ph_input, first num_moms moments and steady-state probs
    :param data_path: the folder in which we save the data
    :param data_sample_name: the name of file
    :param num_moms: number of moments we compute
    :param save_x: should we save ph_data
    :return:
    '''

    now = datetime.now()


    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
    x_list =  []
    mom_list = []
    y_list = []

    for x_y in x_y_data:
        if type(x_y) != bool:
            if save_x:
                x_list.append(torch.from_numpy(x_y[0]))
            mom_list.append(torch.from_numpy(x_y[0]))
            y_list.append(torch.from_numpy(x_y[1]))


    if save_x: # should we want to save the x_data
        # x_list = [torch.from_numpy(x_y[0]) for x_y in x_y_data if type(x_y) != bool]
        # torch_x = torch.stack(x_list).float()
        pkl_name_xdat = 'xdat_' + data_sample_name + current_time +'size_' + '.pkl' #+ str(torch_x.shape[0]) +
        full_path_xdat = os.path.join(data_path, pkl_name_xdat)
        pkl.dump(x_list, open(full_path_xdat, 'wb'))

    # dumping moments
    # mom_list = [torch.from_numpy(x_y[1]) for x_y in x_y_data if type(x_y) != bool]
    torch_moms = torch.stack(mom_list).float()
    pkl_name_moms = 'moms_' + str(num_moms) + data_sample_name + current_time + 'size_'+ str(torch_moms.shape[0]) + '.pkl'
    full_path_moms = os.path.join(data_path, pkl_name_moms)
    pkl.dump(torch_moms, open(full_path_moms, 'wb'))


    # dumping steady_state
    # y_list = [torch.from_numpy(x_y[2]) for x_y in x_y_data if type(x_y) != bool]
    torch_y = torch.stack(y_list).float()
    pkl_name_ydat = 'ydat_' + data_sample_name + current_time +'size_'+ str(torch_y.shape[0]) + '.pkl'
    full_path_ydat = os.path.join(data_path, pkl_name_ydat)
    pkl.dump(torch_y, open(full_path_ydat, 'wb'))

# def generate_one_ph(batch_size, max_ph_size, num_moms, data_path, data_sample_name):
#
#     sample_type_arr = np.random.randint(1, 3, batch_size)
#     x_y_moms_list = [send_to_the_right_generator(val, args.ph_size,  num_moms, data_path, data_sample_name) for val in sample_type_arr]
#     x_y_moms_list = [x_y_moms for x_y_moms in x_y_moms_list if x_y_moms]
#     x_y_moms_lists =  [compute_y_moms(x_y_moms[0],x_y_moms[1], num_moms, max_ph_size) for x_y_moms  in x_y_moms_list]
#     saving_batch(list(itertools.chain(*x_y_moms_lists)), data_path, data_sample_name, num_moms)


    return 1


def sample_size_1(ph_max_size):
    root = int(ph_max_size ** 0.5)
    weight = np.random.uniform(0.5, 1)
    if np.random.rand() > 0.5:
        arrival = int(ph_max_size / (root * weight))
        service = int(root * weight)
    else:
        service = int(ph_max_size / (root * weight))
        arrival = int(root * weight)
    return (arrival,service)

def sample_size(ph_max_size):
    total = np.random.randint(2,ph_max_size+1)
    if np.random.rand()>0.5:
        arrival = np.random.randint(1,max(2,int((total+1)*np.random.rand())))
        service = int(total/arrival)
    else:
        service = np.random.randint(1,max(2,int((total+1)*np.random.rand())))
        arrival = int(total/service)
    return (arrival,service)

def saving_batch_g_g_1(torch_moms, torch_y, data_path, data_sample_name, num_moms, save_x = False):
    '''

    :param x_y_data: the data is a batch of tuples: ph_input, first num_moms moments and steady-state probs
    :param data_path: the folder in which we save the data
    :param data_sample_name: the name of file
    :param num_moms: number of moments we compute
    :param save_x: should we save ph_data
    :return:
    '''

    now = datetime.now()


    current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 10000000, 1)[0])

    pkl_name_moms = 'moms_' + str(num_moms) + data_sample_name + current_time + 'size_'+ str(torch_moms.shape[0]) + '.pkl'
    full_path_moms = os.path.join(data_path, pkl_name_moms)
    pkl.dump(torch_moms, open(full_path_moms, 'wb'))


    # dumping steady_state

    pkl_name_ydat = 'ydat_' + data_sample_name + current_time +'size_'+ str(torch_y.shape[0]) + '.pkl'
    full_path_ydat = os.path.join(data_path, pkl_name_ydat)
    pkl.dump(torch_y, open(full_path_ydat, 'wb'))


def manage_batch(batch_size, ph_size_max, num_moms, data_path, data_sample_name, max_util):
    '''
    batch_size: the batch size we save as a single tensor
    ph_size_ax: relate to the product of the arrival and service sizes
    num_moms: which we need to compute for both arrivals and services
    data_path: the folder in which it is saved
    data_sample_name: the file name of saved tensor
    '''

    'looping over a batch, creating BS samples which include 20 arrival moments, 20 service moments and 500 probs.'
    'In charge of saving batches - creating input tensor of size (BSX40) and output of size (BSX500)'

    mom_output_list = [manage_single_sample(ph_size_max, num_moms, max_util) for ind in range(batch_size)]
    mom_output_list = [pair for pair in mom_output_list if pair]

    mom_list = []
    y_list = []
    for pair in mom_output_list:
        mom_list.append(pair[0])
        y_list.append(pair[1])

    torch_moms = torch.stack(mom_list).float()
    torch_y = torch.stack(y_list).float()
    saving_batch_g_g_1(torch_moms, torch_y, data_path, data_sample_name, num_moms)


def sampling_examples(ph_size_max, num_moms, eps = 0.05):
    '''
    ph_size_max: the maximum number of batch size (product of arrival and service)
    num_moms: number of save moments
    '''
    'sampling arrival and service sizes, then sampling (s,A) for each one'
    'take the pair (ph_arrival, ph_service) and compute y - i.e., the y values for deep'
    'compute moments of both arrival and service'
    'return to manage batch (moms, y)'

    a_size, ser_size = sample_size(ph_size_max)

    ser_size = np.random.randint(250,1000)
    print(ser_size)
    a_size = 1

    flag = True
    while flag: #sample until it is valid
        arrival_result = send_to_the_right_generator(np.random.randint(1, 3), a_size)
        if arrival_result:
            s_arrival, A_arrival = arrival_result
            flag = False

    flag = True
    while flag:  # sample until it is valid
        service_result = send_to_the_right_generator(np.random.randint(1, 3), ser_size)
        if service_result:
            s_service, A_service = service_result
            flag = False


    rho = np.random.uniform(0.6,0.8)
    A_arrival = A_arrival * rho

    s_arrival = s_arrival.reshape((1, s_arrival.shape[0]))
    s_service = s_service.reshape((1, s_service.shape[0]))

    return (s_arrival, A_arrival, s_service, A_service)

def manage_single_sample(ph_size_max, num_moms, max_util,eps = 0.05):
    '''
    ph_size_max: the maximum number of batch size (product of arrival and service)
    num_moms: number of save moments
    '''
    'sampling arrival and service sizes, then sampling (s,A) for each one'
    'take the pair (ph_arrival, ph_service) and compute y - i.e., the y values for deep'
    'compute moments of both arrival and service'
    'return to manage batch (moms, y)'

    a_size, ser_size = sample_size_1(ph_size_max)

    elements = [0, 1, 2]
    probabilities = [0.1, 0.4, 0.5]
    flag = True

    while flag: #sample until it is valid

        arrival_result = send_to_the_right_generator(np.random.choice(elements, 1, p=probabilities)[0], a_size)
        if arrival_result:
            s_arrival, A_arrival = arrival_result
            flag = False

    flag = True
    while flag:  # sample until it is valid
        service_result = send_to_the_right_generator(np.random.choice(elements, 1, p=probabilities)[0], ser_size)
        if service_result:
            s_service, A_service = service_result
            flag = False


    rho = np.random.uniform(0.3,max_util)
    A_arrival = A_arrival * rho

    s_arrival = s_arrival.reshape((1, s_arrival.shape[0]))
    s_service = s_service.reshape((1, s_service.shape[0]))




    stead = compute_steady(s_arrival, A_arrival, s_service, A_service)

    if np.sum(stead) > 1-eps:  # otherwise don't use it

        arrival_moms = compute_first_n_moments(s_arrival, A_arrival, num_moms)
        service_moms = compute_first_n_moments(s_service, A_service, num_moms)

        arrival_moms = torch.log(torch.tensor(np.array(arrival_moms).flatten()))
        service_moms = torch.log(torch.tensor(np.array(service_moms).flatten()))[1:]

        moms_tesnor = torch.cat((arrival_moms,service_moms))

        return (moms_tesnor, torch.tensor(stead))


def main(args):

    print('The current path is:')
    print(os.getcwd())


    # if sys.platform == 'linux':
    #
    #     if os.getcwd() =='/gpfs/fs0/scratch/d/dkrass/eliransc/Deep_queue/code':
    #
    #         data_path = '/scratch/d/dkrass/eliransc/training/gg1_1'
    #     else:
    #         data_path = '/scratch/eliransc/training/gg1_1'
    #
    # else:
    #
    #     data_path = r'C:\Users\user\workspace\data\deep_gg1'

    data_path = args.data_path


    data_sample_name = 'batch_size_' + str(args.batch_size) + '_num_moms_' + str(
        args.num_moms) + '_num_max_size_' +'num_phases_'+str(args.ph_size_max)+'_'+ str(args.max_utilization)
    # x_vals = np.linspace(0, 1, 30)

    # Compute ph_dists


    for ind in tqdm(range(args.num_examples)):
        cur_time = int(time.time())
        seed = cur_time + len(os.listdir(data_path))+np.random.randint(1,1000)
        np.random.seed(seed)
        if ind == 0:
            manage_batch(1, args.ph_size_max, args.num_moms, data_path, data_sample_name,
                         args.max_utilization)
        else:
            manage_batch(args.batch_size, args.ph_size_max, args.num_moms, data_path, data_sample_name, args.max_utilization)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, help='number of ph folders', default=1)
    parser.add_argument('--num_moms', type=int, help='number of ph folders', default=20)
    parser.add_argument('--batch_size', type=int, help='number of ph examples in one file', default=1)
    parser.add_argument('--ph_size_max', type=int, help='number of ph folders', default = 1000)
    parser.add_argument('--data_path', type=str, help='where to save the file', default=r'C:\Users\user\workspace\data\deep_gg1')
    parser.add_argument('--max_utilization', type=float, help='What is the largest possible utilization', default = 0.999)
    args = parser.parse_args(argv)

    return args

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    main(args)
