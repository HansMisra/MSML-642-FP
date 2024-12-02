from gymnasium.envs.registration import register, registry
from my_bipedal_walker import BipedalWalkerWithObject

register(
    id='BipedalWalkerWithObject-v0',
    entry_point='my_bipedal_walker.BipedalWalkerWithObject',
    max_episode_steps=1600,
    reward_threshold=300.0,
)

import numpy as np
import random
import gym
import math
from collections import defaultdict, deque
import matplotlib.pyplot as graph

ENV = "BipedalWalkerWithObject-v0"

qTable = defaultdict( lambda: np.zeros((10, 10, 10, 10)))

EPISODES = 20000
GAMMA =  0.99
ALPHA = 0.01
HIGHSCORE = -200

stateBounds = [(-math.pi, math.pi),
           (-5,5),
           (-5,5),
           (-5,5),
           (-math.pi,math.pi),
           (-5,5),
           (-math.pi, math.pi),
           (-5,5),
           (0,5),
           (-math.pi, math.pi),
           (-5, 5),
           (-math.pi, math.pi),
           (-5, 5),
           (0, 5),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1, 1),
           (-1.5, 1.5),
           (-1, 1)]


actionBounds = (-1, 1)


def updateQTable (Qtable, state, action, reward, nextState=None):
    global ALPHA
    global GAMMA

    current = Qtable[state][action]  
    qNext = np.max(Qtable[nextState]) if nextState is not None else 0
    target = reward + (GAMMA * qNext)
    new_value = current + (ALPHA * (target - current))
    return new_value

def getNextAction(qTable, epsilon, state):

    if random.random() < epsilon:

        action = ()
        for i in range (0, 4):
            action += (random.randint(0, 9),)

    else:

        action = np.unravel_index(np.argmax(qTable[state]), qTable[state].shape)

    return action

def discretizeState(state):

    discreteState = []
    if type(state) == tuple:
        state = state[0]
    #print(state)

    for i in range(len(state)):
        #print(state[i])
        #print(i)
        index = int((state[i]-stateBounds[i][0]) / (stateBounds[i][1]-stateBounds[i][0]))*19
        discreteState.append(index)
    
    return tuple(discreteState)


def convertNextAction(nextAction):
    action = []

    for i in range(len(nextAction)):

        nextVal = nextAction[i] / 9 * 2 - 1

        action.append(nextVal)
    
    return tuple(action)

def plotEpisode(myGraph, xval, yval, epScore, plotLine, i):

    xval.append(i)
    yval.append(epScore)

    plotLine.set_xdata(xval)
    plotLine.set_ydata(yval)
    myGraph.savefig("./plot")



def runAlgorithmStep(env, i, doRender):

    global HIGHSCORE
    global qTable
    env.reset()

    render_this_episode = (i % 1000 == 0)
    if render_this_episode:
        env.render_mode = "human"
    else:
        env.render_mode = None

        if render_this_episode:
            env.render()

    if i % 1000 == 0:
        if(doRender):
            env.render()

    print("Episode #: ", i)

    state = discretizeState(env.reset()[:])
    total_reward=  0
    epsilon = 1 - i/EPISODES

    while True:
        
        nextAction = convertNextAction(getNextAction(qTable, epsilon, state))
        nextActionDiscretized = getNextAction(qTable, epsilon, state)
        #print(env.step(nextAction))
        nextState, reward, done, info, extra = env.step(nextAction)
        #print(nextState)
        nextState = discretizeState(nextState)
        total_reward += reward
        qTable[state][nextActionDiscretized] = updateQTable(qTable, state, nextActionDiscretized, reward, nextState)
        state = nextState
        if done:
                break
    
    if total_reward > HIGHSCORE:

        HIGHSCORE = total_reward

    return total_reward
    
def main():

    global HIGHSCORE
    global qTable
    
    visualize = input("Visualize? [y/n]\n")
    if visualize == 'y':
        doRender = True
    else:
        doRender = False

    #env = gym.make(ENV)
    env = BipedalWalkerWithObject()
    #qTable = defaultdict( lambda: np.zeros((10, 10, 10, 10)))

    myGraph = graph.figure()
    xval, yval = [], []
    mySubPlot = myGraph.add_subplot()
    graph.xlabel("Episode #")
    graph.ylabel("Score")
    graph.title("Scores vs Episode")
    plotLine, = mySubPlot.plot(xval, yval)
    mySubPlot.set_xlim([0, EPISODES])
    mySubPlot.set_ylim([-220, 0])


    for i in range(1, EPISODES + 1):

        epScore = runAlgorithmStep(env, i, doRender)
        print("Episode finished. Now plotting..")
        plotEpisode(myGraph, xval, yval, epScore, plotLine, i)
    
    print("All episodes finished. Highest score achieved: " + str(HIGHSCORE))

  
main()
