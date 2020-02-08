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

# Simmulated annealing algorithm constants
INITIAL_TEMP                = 100
FINAL_TEMP                  = 0.00001
MAX_NUM_OF_MUTATION_SWAPS   = 4
COOLING_CONST               = 0.999


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
    plt.title("Temperature " + str(currentTemperature) + "\nZZB = " + str(fitness) + "        Max distance error = " + str(round(maxDistanceError * 1000, 3)) + " mm        Improvement = " + str(improvementPercentage) + " %\n")
    plt.plot(deltaT_list, correlationVector)
    plt.ylabel('Correlation')
    plt.xlabel('Time')

    # Plot the frequency ranges
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.axis([CENTER_FREQS[0] - FREQ_RANGE_WIDTH/2 - FREQ_RANGE_WIDTH/10, CENTER_FREQS[NUM_OF_FREQ_RANGES - 1] + FREQ_RANGE_WIDTH/2 + FREQ_RANGE_WIDTH/10, 0, 8/NUM_OF_PILOTS])
    plt.stem(allPilotFreqs, candidate, bottom=0, use_line_collection=True)
    plt.ylabel('Power [mW]')
    plt.xlabel('Frequency [Hz]')

    # Show the graph
    plt.draw()
    plt.pause(0.01)



def calcFitness(candidate):

    return matlabEngine.zzb(float(SNR), matlab_freqBins, matlab.double(candidate), float(OBSERVATION_PERIOD), nargout=1)



def mutateCandidate(candidate, numOfMutations):

    for i in range(numOfMutations):
        randIndex1 = random.randrange(NUM_OF_PILOTS)
        randIndex2 = random.randrange(NUM_OF_PILOTS)

        diff = random.uniform(0, candidate[randIndex1])

        candidate[randIndex1] -= diff
        candidate[randIndex2] += diff



# ------------------------------------------- Main ------------------------------------------------ #
# Find all pilot frequencies
allPilotFreqs = np.empty([0])
for i in range(NUM_OF_FREQ_RANGES):
    freqRange = np.arange(CENTER_FREQS[i] - FREQ_RANGE_WIDTH//2, CENTER_FREQS[i] + FREQ_RANGE_WIDTH//2, FREQ_RANGE_WIDTH/NUM_OF_PILOTS_IN_FREQ_RANGE)
    allPilotFreqs = np.concatenate([allPilotFreqs,freqRange])

matlab_freqBins = matlab.double(allPilotFreqs.tolist())

# Create an initial solution
currentCandidate = [1/NUM_OF_PILOTS for i in range(NUM_OF_PILOTS)]
newCandidate = [0 for i in range(NUM_OF_PILOTS)]
bestCandidate = [0 for i in range(NUM_OF_PILOTS)]
bestCandidateFitness = 100000

# Create an empty fitness array
currentCandidateFitness = 0
newCandidateFitness = 0

# Temperature
currentTemperature = INITIAL_TEMP

# Create a figure window
plt.figure(figsize=(40, 40))

# Initial solution error
initialError = [0]

# Start matlab engine
matlabEngine = matlab.engine.start_matlab()


while currentTemperature > FINAL_TEMP:
    newCandidate[:] = currentCandidate[:]
    mutateCandidate(newCandidate, random.randrange(0, MAX_NUM_OF_MUTATION_SWAPS))

    currentCandidateFitness = calcFitness(currentCandidate)
    newCandidateFitness = calcFitness(newCandidate)

    if (newCandidateFitness < currentCandidateFitness):
        currentCandidate[:] = newCandidate[:]
        currentCandidateFitness = newCandidateFitness

        acf_vec_output = matlabEngine.acf(deltaT_vector, matlab_freqBins, matlab.double(currentCandidate), nargout=1)
        acf_vec_output = [acf_vec_output[0][i].real for i in range(len(acf_vec_output[0]))]

        plotCandidate(currentCandidate, currentCandidateFitness, acf_vec_output)
    else:
        p = math.exp(-(newCandidateFitness - currentCandidateFitness) / currentTemperature)
        randomTreshold = random.randrange(20, 100) / 100
        if p > randomTreshold:
            currentCandidate[:] = newCandidate[:]
            currentCandidateFitness = newCandidateFitness

            acf_vec_output = matlabEngine.acf(deltaT_vector, matlab_freqBins, matlab.double(currentCandidate), nargout=1)
            acf_vec_output = [acf_vec_output[0][i].real for i in range(len(acf_vec_output[0]))]

            plotCandidate(currentCandidate, currentCandidateFitness, acf_vec_output)
    
    if currentCandidateFitness < bestCandidateFitness:
        bestCandidate[:] = currentCandidate[:]
        bestCandidateFitness = currentCandidateFitness

    currentTemperature *= COOLING_CONST

# Save the best candidate found
acf_vec_output = matlabEngine.acf(deltaT_vector, matlab_freqBins, matlab.double(bestCandidate), nargout=1)
acf_vec_output = [acf_vec_output[0][i].real for i in range(len(acf_vec_output[0]))]
plotCandidate(bestCandidate, bestCandidateFitness, acf_vec_output)
plt.savefig("BestFoundCandidate_sim_anne.png")