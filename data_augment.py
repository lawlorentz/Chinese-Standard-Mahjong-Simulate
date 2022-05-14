# 饼万条互换 19,28,37,46互换

import numpy as np
# import json
# with open('data/count.json') as f:
#     match_samples = json.load(f)
# total_matches = len(match_samples)
obs_augment_index = [[1, 2, 3], [1, 3, 2], [
    2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
OFFSET_ACT = {
    'Pass': 0,
    'Hu': 1,
    'Play_W': 2,
    'Play_T': 11,
    'Play_B': 20,
    'Chi_W': 36,
    'Chi_T': 57,
    'Chi_B': 78,
    'Peng': 99,
    'Gang': 133,
    'AnGang': 167,
    'BuGang': 201
}




def data_augment(index):
    d = np.load('data/%d.npz' % index)
    cache = {'obs': d['obs'], 'mask': d['mask'], 'act': d['act']}
    for k in d:
        cache[k] = d[k]

    obs_augment = []
    mask_augment = []
    act_augment = []
    obs_1 = obs_2 = cache['obs'].copy()
    # act_1 = act_2 = cache['act'].copy()
    act_1 = np.zeros([cache['act'].size, 235])
    for i in range(cache['act'].size):
        act_1[i][cache['act'][i]] = 1
    act_2 = act_1.copy()
    act_0 = act_1.copy()
    mask_1 = mask_2 = cache['mask'].copy()
    for i in range(6):
        # obs
        obs_1 = cache['obs'].copy()
        for j in range(3):
            obs_1[:, :, j, :] = cache['obs'][:,
                                             :, obs_augment_index[i][j]-1, :]
        obs_2 = obs_1.copy()
        for j in range(9):
            if j == 4:
                continue
            obs_2[:, :, 0:3, j] = obs_1[:, :, 0:3, 8-j]
        obs_augment.append(obs_1)
        obs_augment.append(obs_2)
        # mask
        mask_1 = cache['mask'].copy()
        for j in range(3):
            mask_1[:, OFFSET_ACT['Chi_W']+21*j:OFFSET_ACT['Chi_T']+21*j] = cache['mask'][:, OFFSET_ACT['Chi_W'] +
                                                                                         21*(obs_augment_index[i][j]-1):OFFSET_ACT['Chi_T']+21*(obs_augment_index[i][j]-1)]
            mask_1[:, OFFSET_ACT['Play_W']+9*j:OFFSET_ACT['Play_T']+9*j] = cache['mask'][:, OFFSET_ACT['Play_W'] +
                                                                                         9*(obs_augment_index[i][j]-1):OFFSET_ACT['Play_T']+9*(obs_augment_index[i][j]-1)]
            mask_1[:, OFFSET_ACT['Peng']+9*j:OFFSET_ACT['Peng']+9*(j+1)] = cache['mask'][:, OFFSET_ACT['Peng'] +
                                                                                         9*(obs_augment_index[i][j]-1):OFFSET_ACT['Peng']+9*(obs_augment_index[i][j])]
            mask_1[:, OFFSET_ACT['Gang']+9*j:OFFSET_ACT['Gang']+9*(j+1)] = cache['mask'][:, OFFSET_ACT['Gang'] +
                                                                                         9*(obs_augment_index[i][j]-1):OFFSET_ACT['Gang']+9*(obs_augment_index[i][j])]
            mask_1[:, OFFSET_ACT['AnGang']+9*j:OFFSET_ACT['AnGang']+9*(j+1)] = cache['mask'][:, OFFSET_ACT['AnGang'] +
                                                                                             9*(obs_augment_index[i][j]-1):OFFSET_ACT['AnGang']+9*(obs_augment_index[i][j])]
            mask_1[:, OFFSET_ACT['BuGang']+9*j:OFFSET_ACT['BuGang']+9*(j+1)] = cache['mask'][:, OFFSET_ACT['BuGang'] +
                                                                                             9*(obs_augment_index[i][j]-1):OFFSET_ACT['BuGang']+9*(obs_augment_index[i][j])]
        mask_2 = mask_1.copy()
        for j in range(9):
            if j == 4:
                continue
            mask_2[:, OFFSET_ACT['Play_W'] +
                   j] = mask_1[:, OFFSET_ACT['Play_W']+8-j]
            mask_2[:, OFFSET_ACT['Play_B'] +
                   j] = mask_1[:, OFFSET_ACT['Play_B']+8-j]
            mask_2[:, OFFSET_ACT['Play_T'] +
                   j] = mask_1[:, OFFSET_ACT['Play_T']+8-j]

            mask_2[:, OFFSET_ACT['Peng']+j] = mask_1[:, OFFSET_ACT['Peng']+8-j]
            mask_2[:, OFFSET_ACT['Peng']+9 +
                   j] = mask_1[:, OFFSET_ACT['Peng']+9+8-j]
            mask_2[:, OFFSET_ACT['Peng']+18 +
                   j] = mask_1[:, OFFSET_ACT['Peng']+18+8-j]

            mask_2[:, OFFSET_ACT['Gang']+j] = mask_1[:, OFFSET_ACT['Gang']+8-j]
            mask_2[:, OFFSET_ACT['Gang']+9 +
                   j] = mask_1[:, OFFSET_ACT['Gang']+9+8-j]
            mask_2[:, OFFSET_ACT['Gang']+18 +
                   j] = mask_1[:, OFFSET_ACT['Gang']+18+8-j]

            mask_2[:, OFFSET_ACT['AnGang'] +
                   j] = mask_1[:, OFFSET_ACT['AnGang']+8-j]
            mask_2[:, OFFSET_ACT['AnGang']+9 +
                   j] = mask_1[:, OFFSET_ACT['AnGang']+9+8-j]
            mask_2[:, OFFSET_ACT['AnGang']+18 +
                   j] = mask_1[:, OFFSET_ACT['AnGang']+18+8-j]

            mask_2[:, OFFSET_ACT['BuGang'] +
                   j] = mask_1[:, OFFSET_ACT['BuGang']+8-j]
            mask_2[:, OFFSET_ACT['BuGang']+9 +
                   j] = mask_1[:, OFFSET_ACT['BuGang']+9+8-j]
            mask_2[:, OFFSET_ACT['BuGang']+18 +
                   j] = mask_1[:, OFFSET_ACT['BuGang']+18+8-j]

        for j in range(7):
            if j == 3:
                continue
            mask_2[:, OFFSET_ACT['Chi_W'] +
                   j] = mask_1[:, OFFSET_ACT['Chi_W']+6-j]
            mask_2[:, OFFSET_ACT['Chi_B'] +
                   j] = mask_1[:, OFFSET_ACT['Chi_B']+6-j]
            mask_2[:, OFFSET_ACT['Chi_T'] +
                   j] = mask_1[:, OFFSET_ACT['Chi_T']+6-j]
        mask_augment.append(mask_1)
        mask_augment.append(mask_2)

        # act
        act_1 = np.zeros([cache['act'].size, 235])
        for i_ in range(cache['act'].size):
            act_1[i_][cache['act'][i_]] = 1
        for j in range(3):
            # print(act_1[:, OFFSET_ACT['Chi_W']+21*j:OFFSET_ACT['Chi_T']+21*j])
            # print(OFFSET_ACT['Chi_W'] +21*(obs_augment_index[i][j]-1))
            # print(OFFSET_ACT['Chi_T']+21*(obs_augment_index[i][j]-1))
            # print(act_0[:, OFFSET_ACT['Chi_W'] +21*(obs_augment_index[i][j]-1):OFFSET_ACT['Chi_T']+21*(obs_augment_index[i][j]-1)])
            act_1[:, OFFSET_ACT['Chi_W']+21*j:OFFSET_ACT['Chi_T']+21*j] = act_0[:, OFFSET_ACT['Chi_W'] +
                                                                                21*(obs_augment_index[i][j]-1):OFFSET_ACT['Chi_T']+21*(obs_augment_index[i][j]-1)]
            act_1[:, OFFSET_ACT['Play_W']+9*j:OFFSET_ACT['Play_T']+9*j] = act_0[:, OFFSET_ACT['Play_W'] +
                                                                                9*(obs_augment_index[i][j]-1):OFFSET_ACT['Play_T']+9*(obs_augment_index[i][j]-1)]
            act_1[:, OFFSET_ACT['Peng']+9*j:OFFSET_ACT['Peng']+9*(j+1)] = act_0[:, OFFSET_ACT['Peng'] +
                                                                                9*(obs_augment_index[i][j]-1):OFFSET_ACT['Peng']+9*(obs_augment_index[i][j])]
            act_1[:, OFFSET_ACT['Gang']+9*j:OFFSET_ACT['Gang']+9*(j+1)] = act_0[:, OFFSET_ACT['Gang'] +
                                                                                9*(obs_augment_index[i][j]-1):OFFSET_ACT['Gang']+9*(obs_augment_index[i][j])]
            act_1[:, OFFSET_ACT['AnGang']+9*j:OFFSET_ACT['AnGang']+9*(j+1)] = act_0[:, OFFSET_ACT['AnGang'] +
                                                                                    9*(obs_augment_index[i][j]-1):OFFSET_ACT['AnGang']+9*(obs_augment_index[i][j])]
            act_1[:, OFFSET_ACT['BuGang']+9*j:OFFSET_ACT['BuGang']+9*(j+1)] = act_0[:, OFFSET_ACT['BuGang'] +
                                                                                    9*(obs_augment_index[i][j]-1):OFFSET_ACT['BuGang']+9*(obs_augment_index[i][j])]
        act_2 = act_1.copy()
        for j in range(9):
            if j == 4:
                continue
            act_2[:, OFFSET_ACT['Play_W']+j] = act_1[:, OFFSET_ACT['Play_W']+8-j]
            act_2[:, OFFSET_ACT['Play_B']+j] = act_1[:, OFFSET_ACT['Play_B']+8-j]
            act_2[:, OFFSET_ACT['Play_T']+j] = act_1[:, OFFSET_ACT['Play_T']+8-j]

            act_2[:, OFFSET_ACT['Peng']+j] = act_1[:, OFFSET_ACT['Peng']+8-j]
            act_2[:, OFFSET_ACT['Peng']+9+j] = act_1[:, OFFSET_ACT['Peng']+9+8-j]
            act_2[:, OFFSET_ACT['Peng']+18 +
                  j] = act_1[:, OFFSET_ACT['Peng']+18+8-j]

            act_2[:, OFFSET_ACT['Gang']+j] = act_1[:, OFFSET_ACT['Gang']+8-j]
            act_2[:, OFFSET_ACT['Gang']+9+j] = act_1[:, OFFSET_ACT['Gang']+9+8-j]
            act_2[:, OFFSET_ACT['Gang']+18 +
                  j] = act_1[:, OFFSET_ACT['Gang']+18+8-j]

            act_2[:, OFFSET_ACT['AnGang']+j] = act_1[:, OFFSET_ACT['AnGang']+8-j]
            act_2[:, OFFSET_ACT['AnGang']+9 +
                  j] = act_1[:, OFFSET_ACT['AnGang']+9+8-j]
            act_2[:, OFFSET_ACT['AnGang']+18 +
                  j] = act_1[:, OFFSET_ACT['AnGang']+18+8-j]

            act_2[:, OFFSET_ACT['BuGang']+j] = act_1[:, OFFSET_ACT['BuGang']+8-j]
            act_2[:, OFFSET_ACT['BuGang']+9 +
                  j] = act_1[:, OFFSET_ACT['BuGang']+9+8-j]
            act_2[:, OFFSET_ACT['BuGang']+18 +
                  j] = act_1[:, OFFSET_ACT['BuGang']+18+8-j]

        for j in range(7):
            if j == 3:
                continue
            act_2[:, OFFSET_ACT['Chi_W']+j] = act_1[:, OFFSET_ACT['Chi_W']+6-j]
            act_2[:, OFFSET_ACT['Chi_B']+j] = act_1[:, OFFSET_ACT['Chi_B']+6-j]
            act_2[:, OFFSET_ACT['Chi_T']+j] = act_1[:, OFFSET_ACT['Chi_T']+6-j]
        act_augment.append(act_1)
        act_augment.append(act_2)

    obs_ = np.concatenate(tuple(obs_augment), axis=0)
    mask_ = np.concatenate(tuple(mask_augment), axis=0)
    act_one_hot = np.concatenate(tuple(act_augment), axis=0)
    act_ = np.zeros([act_one_hot.shape[0]]).astype(np.int32)
    for i in range(act_one_hot.shape[0]):
        act_[i] = np.argmax(act_one_hot[i])

    np.savez('data/%d_augmented.npz' % index,
             obs=obs_,
             mask=mask_,
             act=act_)
    print('data %d augmented and saved' % index)


# for i in range(total_matches):
#     data_augment(i)
