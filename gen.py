import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import matlab.engine
import time

# ------------------------------------------- Constants ------------------------------------------- #
# OFDM constants
NUM_OF_PILOTS_IN_FREQ_RANGE = 128
FREQ_RANGE_WIDTH            = 0.4e9         # Hz
CENTER_FREQS                = [1e9, 2e9]    # Hz
NUMBER_OF_POINTS            = 10000         # Number of points for plotting the correlation function
SNR                         = 25            # Signal to noise ratio of the channel

NUM_OF_FREQ_RANGES          = len(CENTER_FREQS)
NUM_OF_PILOTS               = NUM_OF_PILOTS_IN_FREQ_RANGE * NUM_OF_FREQ_RANGES
CARRIER_FREQUENCY           = FREQ_RANGE_WIDTH / NUM_OF_PILOTS_IN_FREQ_RANGE
CARRIER_PERIOD              = 1/CARRIER_FREQUENCY
OBSERVATION_PERIOD          = CARRIER_PERIOD/10
deltaT_list                 = np.array(np.arange(-OBSERVATION_PERIOD/2, OBSERVATION_PERIOD/2 + OBSERVATION_PERIOD/NUMBER_OF_POINTS, OBSERVATION_PERIOD/NUMBER_OF_POINTS)).tolist()
deltaT_vector               = matlab.double(deltaT_list)

# Genetic algorithm constants
NUMBER_OF_GENERATIONS       = 10000
POPULATION_SIZE             = 50
BEST_PARENTS_FACTOR         = 5
MAX_NUM_OF_MUTATION_SWAPS   = 64
MUTATION_TRESHOLD           = 15


# ------------------------------------------- Functions ------------------------------------------- #
def plotCandidate(candidate, fitness, correlationVector):

    maxDistanceError = math.sqrt(fitness) * 3e8

    if initialError[0] == 0:
        initialError[0] = maxDistanceError
    
    improvementPercentage = round(((initialError[0] - maxDistanceError) / initialError[0]) * 100, 3)

    # Plot the correlation function
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.axis([1.1 * min(deltaT_list), 1.1 * max(deltaT_list), 0, 1.1 * max(correlationVector)])
    plt.title("Generation " + str(genCnt[0]) + "\nZZB = " + str(fitness) + "        Max distance error = " + str(round(maxDistanceError * 1000, 3)) + " mm        Improvement = " + str(improvementPercentage) + " %\n")
    plt.plot(deltaT_list, correlationVector)
    plt.ylabel('Correlation')
    plt.xlabel('Time')

    # Plot the frequency ranges
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.axis([CENTER_FREQS[0] - FREQ_RANGE_WIDTH/2 - FREQ_RANGE_WIDTH/10, CENTER_FREQS[NUM_OF_FREQ_RANGES - 1] + FREQ_RANGE_WIDTH/2 + FREQ_RANGE_WIDTH/10, 0, 5/NUM_OF_PILOTS])
    plt.stem(allPilotFreqs, candidate, bottom=0, use_line_collection=True)
    plt.ylabel('Power [mW]')
    plt.xlabel('Frequency [Hz]')

    # Show the graph
    plt.draw()
    plt.pause(0.01)



def calcFitness(population, fitnessArray):

    fitnessArray[:] = [matlabEngine.zzb(float(SNR), matlab_freqBins, matlab.double(population[i]), float(OBSERVATION_PERIOD), nargout=1) for i in range(POPULATION_SIZE)] 



def findNewBest(fitnessArray, population, allTimeBestCandidateFitness, allTimeBestCandidate):

    generationBestCandidateFitness = min(fitnessArray)
    maxFitnessIndex = fitnessArray.index(generationBestCandidateFitness)

    if(generationBestCandidateFitness < allTimeBestCandidateFitness):
        allTimeBestCandidate[:] = population[maxFitnessIndex][:]
        acf_vec_output = matlabEngine.acf(deltaT_vector, matlab_freqBins, matlab.double(population[maxFitnessIndex]), nargout=1)
        acf_vec_output = [acf_vec_output[0][i].real for i in range(len(acf_vec_output[0]))]
        plotCandidate(population[maxFitnessIndex], generationBestCandidateFitness, acf_vec_output)

        return generationBestCandidateFitness

    return allTimeBestCandidateFitness



def mutateCandidate(candidate, numOfMutations):

    for i in range(numOfMutations):
        randIndex1 = random.randrange(NUM_OF_PILOTS)
        randIndex2 = random.randrange(NUM_OF_PILOTS)

        diff = random.uniform(0, candidate[randIndex1])

        candidate[randIndex1] -= diff
        candidate[randIndex2] += diff



def crossoverCandidates(parent1, parent2, newPopulationIndex):
    # Generate 2 random indexes
    randIndex1 = random.randrange(0, NUM_OF_PILOTS)
    randIndex2 = random.randrange(0, NUM_OF_PILOTS)

    # If randIndex1 > randIndex2 swap them
    if randIndex1 > randIndex2:
        randIndex1,randIndex2 = randIndex2,randIndex1

    # Take pilot power values [randIndex1:randIndex2] from parent 1
    newPopulation[newPopulationIndex][randIndex1:randIndex2] = parent1[randIndex1:randIndex2]

    # Find all indexes where power values are still zero
    zeroIndexes = np.where(np.array(newPopulation[newPopulationIndex]) == 0)[0]

    # Fill those places with power values from parent 2 (while respecting the upper bound for total used power)
    for i in range(len(zeroIndexes)):
        if (sum(newPopulation[newPopulationIndex]) + parent2[zeroIndexes[i]]) <= 1:
            newPopulation[newPopulationIndex][zeroIndexes[i]] = parent2[zeroIndexes[i]]
        else:
            newPopulation[newPopulationIndex][zeroIndexes[i]] = 1 - sum(newPopulation[newPopulationIndex])
            break

    # After we filled all of the places, if the total used power is lower then 1, we find
    # 10 random pilots and distribute the remaining available power between them
    if sum(newPopulation[newPopulationIndex]) < 1:
        difference = 1 - sum(newPopulation[newPopulationIndex])
        for t in range(10):
            randIndex = random.randrange(0, NUM_OF_PILOTS)
            newPopulation[newPopulationIndex][randIndex] += difference/10



def createNewPopulation():
    newPopulation[:] = [[0 for col in range(NUM_OF_PILOTS)] for row in range(POPULATION_SIZE)]

    for k in range(BEST_PARENTS_FACTOR//2):
        cntPopulation = k*(POPULATION_SIZE//BEST_PARENTS_FACTOR)*2
        random.shuffle(bestCandidatesIndexes)

        for j in range((POPULATION_SIZE//BEST_PARENTS_FACTOR) - 1):
            crossoverCandidates(population[bestCandidatesIndexes[j]], population[bestCandidatesIndexes[j + 1]], cntPopulation)
            crossoverCandidates(population[bestCandidatesIndexes[j]], population[bestCandidatesIndexes[j + 1]], cntPopulation + 1)
            cntPopulation += 2

        crossoverCandidates(population[bestCandidatesIndexes[(POPULATION_SIZE//BEST_PARENTS_FACTOR) - 1]], population[bestCandidatesIndexes[0]], cntPopulation)
        crossoverCandidates(population[bestCandidatesIndexes[(POPULATION_SIZE//BEST_PARENTS_FACTOR) - 1]], population[bestCandidatesIndexes[0]], cntPopulation + 1)
        cntPopulation += 2



# ------------------------------------------- Main ------------------------------------------------ #
# Find all pilot frequencies
allPilotFreqs = np.empty([0])
for i in range(NUM_OF_FREQ_RANGES):
    freqRange = np.arange(CENTER_FREQS[i] - FREQ_RANGE_WIDTH//2, CENTER_FREQS[i] + FREQ_RANGE_WIDTH//2, FREQ_RANGE_WIDTH/NUM_OF_PILOTS_IN_FREQ_RANGE)
    allPilotFreqs = np.concatenate([allPilotFreqs,freqRange])

matlab_freqBins = matlab.double(allPilotFreqs.tolist())

# Create a population of random candidates
# population = [np.random.uniform(0,100,NUM_OF_PILOTS) for candidate in range(POPULATION_SIZE)]
# sums = [sum(population[j]) for j in range(POPULATION_SIZE)]
# population = [[population[j][i]/sums[j] for i in range(NUM_OF_PILOTS)] for j in range(POPULATION_SIZE)]

population = [[1/NUM_OF_PILOTS for i in range(NUM_OF_PILOTS)] for j in range(POPULATION_SIZE)]
newPopulation = [[0 for col in range(NUM_OF_PILOTS)] for row in range(POPULATION_SIZE)]

# Create an empty fitness array
fitnessArray = [0 for val in range(POPULATION_SIZE)]

# Create a placeholder for the best value
bestCandidate = [0 for val in range(NUM_OF_PILOTS)]
bestCandidateFitness = 100000000000
bestCandidateIndex = [0 for val in range(1)] # We define it as an array to be able to modify it inside the function

# Create a figure window
plt.figure(figsize=(40, 40))

# Initial solution error
initialError = [0]

# Start matlab engine
matlabEngine = matlab.engine.start_matlab()

# Generation counter
genCnt = [0]

# Run the evolution
for genCnt[0] in range(NUMBER_OF_GENERATIONS):
    calcFitness(population, fitnessArray)
    bestCandidateFitness = findNewBest(fitnessArray, population, bestCandidateFitness, bestCandidate)
    bestCandidatesIndexes = np.argsort(fitnessArray)[0:(POPULATION_SIZE//BEST_PARENTS_FACTOR)]
    createNewPopulation()

    for j in range(POPULATION_SIZE):
        randNumber = random.randrange(0, 100)
        if randNumber > (100 - MUTATION_TRESHOLD):
            mutateCandidate(newPopulation[j], random.randrange(0, MAX_NUM_OF_MUTATION_SWAPS))

    population[:] = newPopulation[:]

plt.savefig("BestFoundCandidate_gen.png")