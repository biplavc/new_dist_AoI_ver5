from collections import UserList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tqdm
import os
import pickle

import datetime
import copy
import time
from getSNR import *

now = datetime.datetime.now()




from create_graph_1 import *
import itertools
from itertools import product  
from random_scheduling import *
from greedy_scheduling import *
from mad_scheduling import *
from rr_scheduling import *
from omad_greedy_UL_scheduling import *
from pf_scheduling import *


import sys

from joblib import Parallel, delayed
import multiprocessing as mp

from parameters import *

random.seed(4)
# tf.random.set_seed(42)

print_matrix = False

def distributed_run(arguments):
  
    print(f"passed arguments are {arguments}\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    print(f"passed arguments are {arguments}")

    # pool.starmap(do_scheduling, [(arg[0], arg[1], arg[2]) for arg in arguments]) ## this enable multiprocessing but I am getting memroy allocation and other CUDA related errors with this, so now using sequential execution
    
    for j in arguments: ## no multiprocessing
        do_scheduling(j[0],j[1],j[2])
    
#############################################################

def do_scheduling(deployment, I, scheduler):
    
    STEPS = list(range(min_steps, max_steps, interval))
    
    for T in STEPS:
        print(f"\nsimulation will run for T={T} steps\n")
        global print_matrix
        
        deployment_options = ["MDS", "RP"]
        scheduler_options  = ["random", "greedy", "mad", "omad_greedy_UL", "omad_cumAoI_UL", "omad_links_UL" , "rr", "dqn", "pf"] ## "random", "greedy", "mad", "omad_greedy_UL", "omad_cumAoI_UL", "omad_links_UL" , "rr", "dqn", "pf"
        assert(deployment in deployment_options and scheduler in scheduler_options)
        # schedulers  = ["dqn" "random", "greedy", "mad", "omad_greedy_UL", "rr", "pf"]

        random.seed(4) ## this seed ensures same location of users in every case, keep both seeds
        
        if test_case:
            
            ## exp 24
            print(f"under experiment {experiment}", file = open(folder_name + "/results.txt", "a"), flush = True)

            drones_needed           = 1
            users_per_drone         = [I]
            
            
    #####################################       

            topology = "fully_connected" ## options = "ring", "fully_connected", "new_ring" , "new_ring_1_connection"
            
            if topology == "fully_connected": ## each device connected to everyone else
                adj_matrix = np.ones((I,I))
                for i in range(I):
                    adj_matrix[i][i] = 0
                    
            if topology == "ring": ## each device connected to two of the nearest ones but the edge ones (1,14,15,28) are connected to two of their LHS and RHS respectively. older
                J = I/2 ## devices per feeder, we have 2 feeders
                adj_matrix = np.zeros((I,I))
                for i in range(I):
                    
                    if i%J==0: ## first devices in the feeder, will connect to 2 devices on its right
                        adj_matrix[i][i+1] = 1
                        adj_matrix[i][i+2] = 1
                        
                    elif (i+1)%J==0: ## last devices in the feeder, will connect to 2 devices on its left
                        adj_matrix[i][i-1] = 1
                        adj_matrix[i][i-2] = 1
                        
                    else:
                        adj_matrix[i][i-1] = 1
                        adj_matrix[i][(i+1)%I] = 1
                            
            if topology == "new_ring": ## each device connected to two of the nearest ones in groups of 14. newer
                J = I/2 ## devices per feeder, we have 2 feeders
                adj_matrix = np.zeros((I,I))
                for i in range(I):
                    
                    if i==0: ## 0, 14 first devices in the feeder, will connect to 2 devices on its right
                        adj_matrix[i][1] = 1
                        adj_matrix[i][13] = 1
                        
                    elif i==13: ## last devices in the feeder, will connect to 2 devices on its left
                        adj_matrix[i][12] = 1
                        adj_matrix[i][0] = 1
                        
                    elif i==14: ## last devices in the feeder, will connect to 2 devices on its left
                        adj_matrix[i][15] = 1
                        adj_matrix[i][27] = 1
                        
                    elif i==27: ## last devices in the feeder, will connect to 2 devices on its left
                        adj_matrix[i][26] = 1
                        adj_matrix[i][14] = 1
                                            
                    else:
                        adj_matrix[i][i-1] = 1
                        adj_matrix[i][(i+1)%I] = 1
                        
            if topology == "new_ring_1_connection": ## each device connected to two of the nearest ones in groups of 14. newer
                J = I/2 ## devices per feeder, we have 2 feeders
                adj_matrix = np.zeros((I,I))
                for i in range(I):
                    
                    adj_matrix[i][(i+1)%28] = 1
                        
    ####################################
                        

            # adj_matrix              = np.array([[0, 1, 1, 0, 0], ## 5 UL 10 DL
            #                                     [0, 0, 1, 1, 0],
            #                                     [0, 0, 0, 1, 1],
            #                                     [1, 0, 0, 0, 1],
            #                                     [1, 1, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1, 0, 0], ## 5 UL 8 DL
            #                                     [0, 0, 1, 1, 0],
            #                                     [0, 0, 0, 0, 1],
            #                                     [1, 0, 0, 0, 1],
            #                                     [1, 0, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1, 1, 1], ## 5 UL 11 DL
            #                                     [1, 0, 0, 1, 1],
            #                                     [0, 0, 0, 1, 0],
            #                                     [1, 0, 0, 0, 1],
            #                                     [1, 1, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 0, 0, 0], ## 5 UL 5 DL
            #                                     [0, 0, 1, 0, 0],
            #                                     [0, 0, 0, 1, 0],
            #                                     [0, 0, 0, 0, 1],
            #                                     [1, 0, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1], ## 3 UL 6 DL
            #                                     [1, 0, 1],
            #                                     [1, 1, 0]])
            
            
            # adj_matrix              = np.array([[0, 1], ## 2 UL 2 DL
            #                                     [1, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1, 0], ## 4 UL 8 DL
            #                                     [0, 0, 1, 1],
            #                                     [1, 0, 0, 1],
            #                                     [1, 1, 0, 0]])

            
            # adj_matrix              = np.array([[0, 1, 1, 0], ## 4 UL 7 DL
            #                                     [0, 0, 1, 1],
            #                                     [1, 0, 0, 1],
            #                                     [0, 1, 0, 0]])
            
            # adj_matrix              = np.array([[0, 0, 1, 0], ## 4 UL 4 DL
            #                                     [0, 0, 0, 1],
            #                                     [1, 0, 0, 0],
            #                                     [0, 1, 0, 0]])
            
            if print_matrix == False:
                print(f"adj_matrix = \n", np.matrix(adj_matrix))
                print_matrix = True
            
            tx_rx_pairs = []
            tx_users    = []
            
            rows, columns = np.shape(adj_matrix)
            # print(f"rows = {rows}, columns = {columns}")
            
            ## relevant pair calculation starts
            
            # age at the final dest will be w.r.t only these pairs.  
            for i in range(rows):
                for ii in range(columns):
                    if adj_matrix[i,ii]==1:
                        pair = [i + 10, ii + 10] ## 10 as count is 10 from main_tf.py where user IDs start from 10
                        tx_rx_pairs.append(pair)
            
            for i in tx_rx_pairs:
                if i[0] not in tx_users:
                    tx_users.append(i[0])
                    
            assert drones_needed    ==len(users_per_drone)
            
            drones_coverage         = []
            
            count = 10 # user IDs will start from this. and this also ensured that UAV and users have different IDs. Ensure number of UAVs is less than the count
            for i in range(drones_needed):
                individual_drone_coverage = [x for x in range(count, count + users_per_drone[i])]
                print(f"individual_drone_coverage = {individual_drone_coverage}", file = open(folder_name + "/results.txt", "a"), flush = True)
                count = individual_drone_coverage[-1] + 1
                drones_coverage.append(individual_drone_coverage)
                
            user_list = []
            UAV_list = np.arange(drones_needed)
            for i in drones_coverage:
                for j in i:
                    if j!=0: ## user will not contain 0
                        user_list.append(j)
                        
            # RB_needed_UL = {x : random.choice([1,2,3]) for x in user_list}
            RB_needed_UL = {x : random.choice([1]) for x in user_list}
            RB_needed_DL = {tuple(x) : RB_needed_UL[x[0]] for x in tx_rx_pairs}


            print(f"user_list = {user_list}, UAV_list = {UAV_list}", file = open(folder_name + "/results.txt", "a"), flush = True)
            assert (max(user_list) - min(user_list))+1 == sum(users_per_drone)
            # time.sleep(10)

                        
            if periodic_generation:
                periodicity = {x:random.choice([2,3,4]) for x in user_list} # biplav {10:1,11:2,12:3,13:2,14:3} #
            else:
                periodicity = {x:1 for x in user_list}
            
            I = len(user_list) # changed to the needed value
            
            L = 1_000 # length of the area
            B = 1_000 # breadth of the area
            
            BS_location = [L/2, B/2]
            
            user_locations = {}
            
            distances = []
            
            for j in user_list:
                x = np.round(random.uniform(0, L),2)
                y = np.round(random.uniform(0, B),2)
                user_locations[j] = [x,y]
                distances.append(math.sqrt((BS_location[0]-x)**2 + (BS_location[1]-y)))
                
            print(f"BS_location = {BS_location}, user_locations = {user_locations}, max distance = {np.max(distances)}, min distance = {np.min(distances)}")
            

            if packet_loss == True:
                SNR_threshold = 3 ## dB from URLLC MCS5 QPSK

            else:
                SNR_threshold = -1_000_000
                
            packet_download_loss_thresh  = {tuple(yy) : SNR_threshold for yy in tx_rx_pairs}
            packet_upload_loss_thresh  = {yy : SNR_threshold for yy in user_list}

                
        else: ## user defined UAV and user configuration
            
            assert test_case == True, "Test Case cannot be false here" # denominator can't be 0 
                            
            # I is number of users, L length and B breadth
            x_vals = random.sample(range(1, L-1), I) # x-coordinates for users
            y_vals = random.sample(range(1, B-1), I) # y-coordinates for users
            z_vals = [0]*I

            user_coordinates = list(zip(x_vals,y_vals))

            x_grid_nos = int(L/r) + 1 # number of different values the grid takes for x axis
            y_grid_nos = int(B/r) + 1 # number of different values the grid takes for y axis

            grid_x = np.linspace(0, L, num = x_grid_nos) # generate evenly spaced x positions for grid
            grid_y = np.linspace(0, B, num = y_grid_nos) # generate evenly spaced y positions for grid
            
            grid_coordinates = list(itertools.product(grid_x , grid_y))

            drones_needed, drones_coverage = create_graph_1(user_coordinates, grid_coordinates, deployment)       
        
            user_list = [] ## this is not the same user_list as defined in the environment, this is just used to index the packet loss and sample loss
            UAV_list  = np.arange(drones_needed)
            
            for i in drones_coverage:
                for j in i:
                    if j!=0:
                        user_list.append(j)
                        
            # RB_needed_UL = {x : random.choice([1,2,3]) for x in user_list} ## biplav
            RB_needed_UL = {x : random.choice([1]) for x in user_list}
            RB_needed_DL = {x : RB_needed_UL[x[0]] for x in tx_rx_pairs}    
            
            if periodic_generation:
                periodicity = {x:random.choice([2,3,4]) for x in user_list}
            else:
                periodicity = {x:1 for x in user_list}


            if packet_loss == True:
                SNR_threshold = 3 ## dB from URLLC MCS5 QPSK

            else:
                SNR_threshold = -1_000_000
            
            packet_download_loss_thresh  = {tuple(yy) : SNR_threshold for yy in tx_rx_pairs}
            packet_upload_loss_thresh  = {yy : SNR_threshold for yy in user_list}
            

        print(f"\n\n{deployment} deployment for {I} users under {scheduler} scheduling\n", file=open(folder_name + "/results.txt", "a"), flush=True)

                
        print(f'Under test_case = {test_case}, drones_needed = {drones_needed}, UAV_list = {UAV_list}, drones_coverage = {drones_coverage}, user_list = {user_list}, periodicity = {periodicity}, RB_total_UL = {RB_total_UL}, RB_total_DL = {RB_total_DL}, total_RB_needed_UL = {np.sum(list(RB_needed_UL.values()))}, total_RB_needed_DL = {np.sum(list(RB_needed_DL.values()))}, RB_needed_UL = {RB_needed_UL}, , RB_needed_DL = {RB_needed_DL} for {deployment} deployment for {I} users under {scheduler} scheduling, packet_download_loss_thresh = {packet_download_loss_thresh}, packet_upload_loss_thresh = {packet_upload_loss_thresh}, user_list = {user_list}, UAV_list = {UAV_list} ", SNR_threshold = {SNR_threshold}\n', file=open(folder_name + "/results.txt", "a"), flush=True)  
        
        str_x = str(deployment) + " placement with " + str(I) + " users and " + str(T) + " slots needs " + str(scheduler) + " scheduler and "  + str(drones_needed) + " drones\n"
        print(f'{str_x}', file=open(folder_name + "/drones.txt", "a"), flush=True)
        
        
        if scheduler == "greedy":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            greedy_overall[I], greedy_final[I], greedy_overall_times[I] = greedy_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)  
            t2 = time.time()
            print("greedy for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(greedy_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_greedy_overall.pickle", "wb")) 
            pickle.dump(greedy_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_greedy_final.pickle", "wb"))
            pickle.dump(greedy_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "_greedy_all_actions.pickle", "wb")) 

        
        if scheduler == "random":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            random_overall[I], random_final[I], random_overall_times[I] = random_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("random for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(random_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_overall.pickle", "wb")) 
            pickle.dump(random_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_final.pickle", "wb")) 
            pickle.dump(random_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_all_actions.pickle", "wb")) 

            
        if scheduler == "omad_greedy_UL":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            omad_greedy_UL_overall[I], omad_greedy_UL_final[I], omad_greedy_UL_overall_times[I] = omad_greedy_UL_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("omad_greedy_UL for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(omad_greedy_UL_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_greedy_UL_overall.pickle", "wb")) 
            pickle.dump(omad_greedy_UL_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_greedy_UL_final.pickle", "wb")) 
            # pickle.dump(omad_greedy_UL_overall_times, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_waoi_all_actions.pickle", "wb"))
            
            
        if scheduler == "mad":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            mad_overall[I], mad_final[I], mad_overall_times[I] = mad_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()            
            print("MAD for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(mad_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_overall.pickle", "wb")) 
            pickle.dump(mad_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_final.pickle", "wb"))
            pickle.dump(mad_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_all_actions.pickle", "wb")) 
            

        if scheduler == "rr":
            t1 = time.time()
            rr_overall[I], rr_final[I], rr_overall_times[I] = rr_new_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("RR for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(rr_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_rr_overall.pickle", "wb")) 
            pickle.dump(rr_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_rr_final.pickle", "wb"))
            pickle.dump(rr_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_rr_all_actions.pickle", "wb"))
            
        if scheduler == "pf":
            t1 = time.time()
            pf_overall[I], pf_final[I], pf_overall_times[I] = pf_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("PF for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(pf_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_pf_overall.pickle", "wb")) 
            pickle.dump(pf_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_pf_final.pickle", "wb"))
            pickle.dump(pf_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_pf_all_actions.pickle", "wb"))

#############################################################
    
if __name__ == '__main__':
        

    now_str_1 = now.strftime("%Y-%m-%d %H:%M")
    folder_name = 'models/' +  now_str_1
    
    # print(f"\n\nSTATUS OF GPU : {tf.test.is_built_with_gpu_support() and {tf.test.is_gpu_available()}}\n\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    
    # print(f"\n\nSTATUS OF GPU : {tf.test.is_built_with_gpu_support() and {tf.test.is_gpu_available()}}\n\n")
    
    folder_name_MDS = folder_name + "/MDS"
    folder_name_random = folder_name + "/RP" ## RP means random placement

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        os.makedirs(folder_name_MDS)
        os.makedirs(folder_name_random) 
        
    print("execution started at ", now_str_1, file = open(folder_name + "/results.txt", "a"), flush = True)

    print("random_episodes = ", random_episodes,", RB_total_UL = ", RB_total_UL, ", RB_total_DL = ", RB_total_DL,",  min_steps = ", min_steps, ", interval = ", interval, ", max_steps = ", max_steps, ", coverage_capacity = ", coverage_capacity, ", L = ", L, ", B = ", B, ", R = ", R, ", r = ", r, "\n", file = open(folder_name + "/results.txt", "a"), flush = True)

    deployments = ["RP"] #, "RP"] #, "MDS"]
    
    schedulers  = ["random"]  
    # "random", "greedy", "mad", "omad_greedy_UL", "rr", "pf"

    limit_memory = True ## enabling this makes the code not being able to find CUDA device
    
#############################

    test_case = bool
    packet_loss = bool
    periodic_generation = bool
    
    experiment = 1 ## 1 or 5 ## biplav

    if experiment == 1:
        test_case           = True
        packet_loss         = False
        periodic_generation = False

    elif experiment == 2:
        test_case           = True
        packet_loss         = True
        periodic_generation = False
        
    elif experiment == 3:
        test_case           = False
        packet_loss         = False
        periodic_generation = False

    elif experiment == 4:
        test_case           = False
        packet_loss         = True
        periodic_generation = False

    elif experiment == 5:
        test_case           = True
        packet_loss         = False
        periodic_generation = True

    elif experiment == 6:
        test_case           = True
        packet_loss         = True
        periodic_generation = True
        
    elif experiment == 7:
        test_case           = False
        packet_loss         = False
        periodic_generation = True

    elif experiment == 8:
        test_case           = False
        packet_loss         = True
        periodic_generation = True
    

    if test_case:
        users = [14] ##biplav
    else:
        users = [8,10]

#############################
    
    arguments = list(itertools.product(deployments, users, schedulers)) ## deployment, I, scheduler
    
    dqn_overall = {}
    dqn_final = {}
    dqn_all_actions = {}
    dqn_overall_times = {} # dqn_overall_avg for each MAX_STEPS
    
    c51_overall = {}
    c51_final = {}
    c51_all_actions = {}
    c51_overall_times = {} # c51_overall_avg for each MAX_STEPS

    
    reinforce_overall = {}
    reinforce_final = {}
    reinforce_all_actions = {}
    reinforce_overall_times = {} # reinforce_overall_avg for each MAX_STEPS

    
    random_overall = {} ## sum of age at destination nodes for all of the MAX_STEPS time steps
    random_overall = {} ## sum of age at destination nodes for all of the MAX_STEPS time steps
    random_final   = {} ## sum of age at destination nodes for step =  MAX_STEPS i.e. last time step
    random_all_actions = {}
    random_overall_times = {} # random_overall_times for each MAX_STEPS

    
    greedy_overall = {}
    greedy_final   = {}
    greedy_all_actions = {}
    greedy_overall_times = {} # greedy_overall_times for each MAX_STEPS

    
    mad_overall = {}
    mad_final   = {}
    mad_all_actions = {}
    mad_overall_times = {} #  mad_overall_avg for each MAX_STEPS

    
    sac_overall = {}
    sac_final   = {}
    sac_all_actions = {}
    sac_overall_times = {} # sac_overall_avg for each MAX_STEPS

    
    lrb_overall = {}
    lrb_final   = {}
    lrb_all_actions = {}
    lrb_overall_times = {} # lrb_overall_avg for each MAX_STEPS


    mrb_overall = {}
    mrb_final   = {}
    mrb_all_actions = {}
    mrb_overall_times = {} # mrb_overall_avg for each MAX_STEPS

    
    rr_overall = {}
    rr_final   = {}
    rr_all_actions = {}
    rr_overall_times = {} # rr_overall_avg for each MAX_STEPS
    
    pf_overall = {}
    pf_final   = {}
    pf_all_actions = {}
    pf_overall_times = {} # rr_overall_avg for each MAX_STEPS
    
    
    omad_greedy_UL_overall = {} 
    omad_greedy_UL_final = {}
    omad_greedy_UL_all_actions = {}
    omad_greedy_UL_overall_times = {} # omad_greedy_UL_overall_times for each MAX_STEPS
    
    omad_cumAoI_UL_overall = {}
    omad_cumAoI_UL_final = {}
    omad_cumAoI_UL_all_actions = {}
    omad_cumAoI_UL_overall_times = {} # omad_cumAoI_UL_overall_times for each MAX_STEPS
    
    omad_links_UL_overall = {}
    omad_links_UL_final = {} 
    omad_links_UL_all_actions = {}
    omad_links_UL_overall_times = {}

    

    pool = mp.Pool(mp.cpu_count())
    print(f"pool is {pool} \n\n", file = open(folder_name + "/results.txt", "a"))
    print(f"experiment is {experiment} with test_case = {test_case}, packet_loss = {packet_loss}, periodic_generation = {periodic_generation}, RB_total_UL = {RB_total_UL}, RB_total_DL = {RB_total_DL}", file = open(folder_name + "/results.txt", "a"))
    print(f"experiment is {experiment} with test_case = {test_case}, packet_loss = {packet_loss}, periodic_generation = {periodic_generation}, RB_total_UL = {RB_total_UL}, RB_total_DL = {RB_total_DL}")

    distributed_run(arguments)
    pool.close()    
