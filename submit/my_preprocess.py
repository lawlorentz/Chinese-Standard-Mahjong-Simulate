from my_feature import FeatureAgent
import numpy as np
import json

obs = [[] for i in range(4)]
actions = [[] for i in range(4)]
matchid = -1

l = []


def filterData():
    global obs
    global actions
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1:  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions


def saveData():
    assert [len(x) for x in obs] == [len(x)
                                     for x in actions], 'obs actions not matching!'
    l.append(sum([len(x) for x in obs]))
    np.savez('data/%d.npz' % matchid,
             obs=np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8), 
             mask=np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8), 
             act=np.array([x for i in range(4) for x in actions[i]]))
    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


'''
need to revise
'''
with open('data/data.txt', encoding='UTF-8') as f:
    line = f.readline()
    total_play = [0, 0, 0, 0]
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            total_play = [0, 0, 0, 0]  # 初始化总共打出的牌的张数，用于28*4层
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            print('Processing match %d %s...' % (matchid, t[1]))
        elif t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':  # 发牌时要使剩余的牌相应的-1
                for j in range(3, len(t)):
                    c = 146
                    while agents[0].obs[c][agents[0].OFFSET_TILE[t[j]]] == 0 and c > 143:
                        c -= 1
                    if c == 143:
                        if agents[0].obs[c][agents[0].OFFSET_TILE[t[j]]] == 1:
                            for i in range(4):
                                agents[i].obs[c][agents[i].OFFSET_TILE[t[j]]] = 0
                    elif c == 144 or c == 145 or c == 146:
                        for i in range(4):
                            agents[i].obs[c][agents[i].OFFSET_TILE[t[j]]] = 0
                agents[p].request2obs(' '.join(t[2:]))

            elif t[2] == 'Draw':
                # 摸牌时要使剩余的牌--
                c = 146
                while agents[0].obs[c][agents[0].OFFSET_TILE[t[3]]] == 0 and c > 143:
                    c -= 1
                if c == 143:
                    if agents[0].obs[c][agents[0].OFFSET_TILE[t[3]]] == 1:
                        for i in range(4):
                            agents[i].obs[c][agents[i].OFFSET_TILE[t[3]]] = 0
                elif c == 144 or c == 145 or c == 146:
                    for i in range(4):
                        agents[i].obs[c][agents[i].OFFSET_TILE[t[3]]] = 0

                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):  # 此处我们让当前的出牌的人的当前输出层的相应牌为1，然后将输出层+1，继续下去
                    curTile = t[3]  #
                    agents[i].obs[31+total_play[p]
                                  ][agents[i].OFFSET_TILE[curTile]] = 1
                total_play[p] += 1    #
                agents[p].request2obs(line)
                for i in range(4):   #
                    if i != p:    #
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)    #
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action(
                    'Chi %s %s' % (curTile, t[3])))
                tile = t[3]
                color = tile[0]
                num = int(tile[1])
                # 表示记录每一个人每次吃的中间那张牌，也就是chi 后面的那个字符串对应的牌的数字
                j = 0
                while agents[0].obs[6+p*4+j][agents[i].OFFSET_TILE[t[3]]] == 1 and j < 3:
                    j += 1
                for i in range(4):
                    agents[i].obs[6+p*4+j][agents[i].OFFSET_TILE[t[3]]] = 1
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(
                            'Player %d Chi %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                # 在相应的人的表示碰的层将相应的牌++
                agents[p].obs[22+p][agents[p].OFFSET_TILE[t[3]]] = 1
                if p > last_p:
                    for i in range(last_p+1, p-1, 1):
                        total_play[i] += 1
                else:
                    for i in range(last_p+1, p+4-1, 1):
                        total_play[i % 4] += 1

                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(
                            'Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                # 在相应的人的表示杠的层将相应的牌++
                agents[p].obs[26+p][agents[p].OFFSET_TILE[t[3]]] = 1
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(
                    agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(
                            'Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(
                    agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(
                            'Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs(
                            'Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action(
                                'Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else:
                        break
            last_p = p
        elif t[0] == 'Score':
            filterData()
            saveData()
        line = f.readline()

with open('data/count.json', 'w') as f:
    json.dump(l, f)
