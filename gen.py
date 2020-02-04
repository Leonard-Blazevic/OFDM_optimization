import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import matlab.engine 

# ------------------------------------------- Functions ------------------------------------------- #
def plotCandidate(candidate, fitness, correlationVector):

    plt.subplot(2, 1, 1)
    plt.cla()
    plt.title("Fitness = " + str(fitness))
    plt.plot(deltaT_list, correlationVector)
    plt.ylabel('Correlation')
    plt.xlabel('Time')

    plt.subplot(2, 1, 2)
    plt.cla()
    plt.bar(allPilotFreqs, candidate)
    plt.ylabel('Power [mW]')
    plt.xlabel('Frequency [MHz]')

    plt.draw()
    plt.pause(0.01)



def calcFitness(population, fitnessArray):

    print("Calculating fitness...")
    fitnessArray[:] = [matlabEngine.zzb(float(SNR), matlab_freqBins, matlab.double(population[i]), float(OBSERVATION_PERIOD), nargout=1) for i in range(POPULATION_SIZE)] 



def findNewBest(fitnessArray, population, allTimeBestCandidate, oldBestCandidate):
    
    generationBestCandidate = min(fitnessArray)
    maxFitnessIndex = fitnessArray.index(generationBestCandidate)


    print("All-time best = " + str(allTimeBestCandidate) + " Generation best = " + str(generationBestCandidate))

    if(generationBestCandidate < allTimeBestCandidate):
        oldBestCandidate[:] = population[maxFitnessIndex][:]
        acf_vec_output = matlabEngine.acf(deltaT_vector, matlab_freqBins, matlab.double(population[maxFitnessIndex]), nargout=1)
        acf_vec_output = [acf_vec_output[0][i].real for i in range(len(acf_vec_output[0]))]
        plotCandidate(population[maxFitnessIndex], generationBestCandidate, acf_vec_output)

        return generationBestCandidate

    return allTimeBestCandidate



def mutateCandidate(candidate, numOfMutations):

    for i in range(numOfMutations):
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
# OFDM constants
NUM_OF_PILOTS_IN_FREQ_RANGE = 128
NUM_OF_MUTATIONS            = 4
FREQ_RANGE_WIDTH            = 400   # MHz
CENTER_FREQS                = [1400, 2800] # MHz

NUM_OF_FREQ_RANGES = len(CENTER_FREQS)
NUM_OF_PILOTS = NUM_OF_PILOTS_IN_FREQ_RANGE * NUM_OF_FREQ_RANGES

CARRIER_FREQUENCY           = FREQ_RANGE_WIDTH / NUM_OF_PILOTS_IN_FREQ_RANGE
CARRIER_PERIOD              = 1/CARRIER_FREQUENCY
SNR                         = 0.01
OBSERVATION_PERIOD          = CARRIER_PERIOD/8
deltaT_list                 = np.array(np.arange(-CARRIER_PERIOD/2, CARRIER_PERIOD/2 + CARRIER_PERIOD/500, CARRIER_PERIOD/500)).tolist()
deltaT_vector               = matlab.double(deltaT_list)

# Genetic algorithm constants
POPULATION_SIZE             = 100
BEST_PARENTS_FACTOR         = 10
MAX_NUM_OF_MUTATION_SWAPS   = 15

# Find all pilot frequencies
allPilotFreqs = np.empty([0])
for i in range(NUM_OF_FREQ_RANGES):
    freqRange = np.arange(CENTER_FREQS[i] - FREQ_RANGE_WIDTH//2, CENTER_FREQS[i] + FREQ_RANGE_WIDTH//2, FREQ_RANGE_WIDTH/NUM_OF_PILOTS_IN_FREQ_RANGE)
    allPilotFreqs = np.concatenate([allPilotFreqs,freqRange])

matlab_freqBins = matlab.double(allPilotFreqs.tolist())

# Create a population of random candidates
population = [np.random.uniform(0,100,NUM_OF_PILOTS) for candidate in range(POPULATION_SIZE)]
sums = [sum(population[j]) for j in range(POPULATION_SIZE)]
population = [[population[j][i]/sums[j] for i in range(NUM_OF_PILOTS)] for j in range(POPULATION_SIZE)]
newPopulation = [[0 for col in range(NUM_OF_PILOTS)] for row in range(POPULATION_SIZE)]

# Create an empty fitness array
fitnessArray = [0 for val in range(POPULATION_SIZE)]

# Create a placeholder for the best value
bestCandidate = [0 for val in range(NUM_OF_PILOTS)]
bestCandidateFitness = 100000000000
bestCandidateIndex = [0 for val in range(1)] # We define it as an array to be able to modify it inside the function

# Create a figure window
plt.figure(figsize=(40, 40))

# Start matlab engine
matlabEngine = matlab.engine.start_matlab()

# Run the evolution
for i in range(100):

    calcFitness(population, fitnessArray)
    bestCandidateFitness = findNewBest(fitnessArray, population, bestCandidateFitness, bestCandidate)
    bestCandidatesIndexes = np.argsort(fitnessArray)[0:(POPULATION_SIZE//BEST_PARENTS_FACTOR)]

    #createNewPopulation()

    for j in range(len(population)):
        mutateCandidate(population[j], random.randrange(0, MAX_NUM_OF_MUTATION_SWAPS))

    #population[:] = newPopulation[:]