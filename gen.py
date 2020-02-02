import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------- Functions ------------------------------------------- #
def plotCandidate(candidate, fitness):
    plt.cla()
    plt.title("Fitness = " + str(fitness))

    #plt.axis([0, 15, 0, 1])
    plt.bar(allPilotFreqs, candidate)
    plt.ylabel('Power [mW]')
    plt.xlabel('Frequency [MHz]')

    plt.draw()
    plt.pause(0.01)

def mutateCandidate(candidate):
    for i in range(NUM_OF_MUTATIONS):
        randomNum = random.randrange(0, 10)
        randIndex1 = random.randrange(NUM_OF_PILOTS)
        randIndex2 = random.randrange(NUM_OF_PILOTS)

        if randomNum < 5:
            temp = candidate[randIndex1]
            candidate[randIndex1] = candidate[randIndex2]
            candidate[randIndex2] = temp
        else:
            diff = candidate[randIndex1] - random.uniform(0, candidate[randIndex1])
            candidate[randIndex1] -= diff
            candidate[randIndex2] += diff

# ------------------------------------------- Main ------------------------------------------------ #
# Constants
NUM_OF_PILOTS_IN_FREQ_RANGE = 128
NUM_OF_MUTATIONS            = 4
FREQ_RANGE_WIDTH            = 400   # MHz
CENTER_FREQS                = [1400, 2800] # MHz

NUM_OF_FREQ_RANGES = len(CENTER_FREQS)
NUM_OF_PILOTS = NUM_OF_PILOTS_IN_FREQ_RANGE * NUM_OF_FREQ_RANGES

# Find all pilot frequencies
allPilotFreqs = np.empty([0])
for i in range(NUM_OF_FREQ_RANGES):
    freqRange = np.arange(CENTER_FREQS[i] - FREQ_RANGE_WIDTH//2, CENTER_FREQS[i] + FREQ_RANGE_WIDTH//2, FREQ_RANGE_WIDTH/NUM_OF_PILOTS_IN_FREQ_RANGE)
    allPilotFreqs = np.concatenate([allPilotFreqs,freqRange])

# Create a random candidate
candidate = np.random.uniform(0,100,NUM_OF_PILOTS)
candidateSum = sum(candidate)
candidate = [i/candidateSum for i in candidate]

# Create a figure window
plt.figure(figsize=(40, 40))

# Run the evolution
for i in range(20):
    mutateCandidate(candidate)
    plotCandidate(candidate, 1)