import numpy as np
import random
from sklearn.metrics import normalized_mutual_info_score


def nmi_cp(c_label, c_pred, p_label, p_pred):
    nmi_cc = normalized_mutual_info_score(c_label, c_pred)
    nmi_pp = normalized_mutual_info_score(p_label, p_pred)
    nmicp = (nmi_cc + nmi_pp) / 2
    return nmicp

def myrand(p):
    # possibility of generating 0 or 1
    num = random.randint(-p, 99 - p)
    if num >= 0:
        return 0
    return 1


def randnet(nsize):
    avgpairsize = 100

    lenth = nsize
    corelist = np.zeros(lenth)
    netmatrix = np.zeros([lenth, lenth])
    pairnum = int(lenth / avgpairsize)

    # un-average assignment
    pairsize = [1 for x in range(pairnum)]  # initial number of nodes in each pairnum

    for i in range(0, pairnum):
        pairid_temp = random.randint(int(avgpairsize * 0.5), int(avgpairsize * 5))
        if (sum(pairsize) + pairid_temp + pairnum - i - 2) > lenth: break;
        pairsize[i] = pairid_temp

    core_pb = 50  # 50 core probability
    core_core_pb = 60  # 80 core-core connection probability
    core_per_pb = 60  # 80 core-pre connection probability
    per_per_pb = 5  # pre-pre connection probability

    pair_num_pairid_temp = 0
    rand_pairidnum = list(np.ones((lenth), dtype=int))  # pair id of each node
    leftnode = list(range(lenth))
    for pairid_temp1 in pairsize:
        pairid_templist = random.sample(leftnode, pairid_temp1)  # generate un-duplicated nodes
        leftnode = list(set(leftnode).difference(set(pairid_templist)))  # delete nodes in pairid_templist from leftnode
        for pairid_temp_p in pairid_templist:
            rand_pairidnum[pairid_temp_p] = pair_num_pairid_temp
        pair_num_pairid_temp = pair_num_pairid_temp + 1

        corelist_pairid_temp = [0 for x in range(pairid_temp1)]  # initial input --all core

        for pairid_temp2 in range(pairid_temp1):
            corelist_pairid_temp[pairid_temp2] = myrand(core_pb)  # probability of generating core
            corelist[pairid_templist[pairid_temp2]] = corelist_pairid_temp[pairid_temp2]

        for pairid_temp2 in range(pairid_temp1):
            for pairid_temp3 in range(pairid_temp1):
                if pairid_temp2 == pairid_temp3:
                    continue
                if corelist_pairid_temp[pairid_temp2] == corelist_pairid_temp[pairid_temp3]:
                    if corelist_pairid_temp[pairid_temp2] == 1:
                        pairid_temp4 = myrand(core_core_pb)  # core-core
                        netmatrix[pairid_templist[pairid_temp2]][pairid_templist[pairid_temp3]] = pairid_temp4
                        netmatrix[pairid_templist[pairid_temp3]][pairid_templist[pairid_temp2]] = pairid_temp4
                    else:
                        pairid_temp4 = myrand(per_per_pb)  # per-per
                        netmatrix[pairid_templist[pairid_temp2]][pairid_templist[pairid_temp3]] = pairid_temp4
                        netmatrix[pairid_templist[pairid_temp3]][pairid_templist[pairid_temp2]] = pairid_temp4
                else:
                    pairid_temp4 = myrand(core_per_pb)  # core-per
                    netmatrix[pairid_templist[pairid_temp2]][pairid_templist[pairid_temp3]] = pairid_temp4
                    netmatrix[pairid_templist[pairid_temp3]][pairid_templist[pairid_temp2]] = pairid_temp4

    pairid_tempi = random.sample(range(lenth), int(nsize * per_per_pb * 0.01))
    for i in pairid_tempi:
        j = random.randint(0, nsize - 1)
        netmatrix[i][j] = 1
        netmatrix[j][i] = 1

    return netmatrix, np.array(rand_pairidnum), np.array(corelist), pairnum
