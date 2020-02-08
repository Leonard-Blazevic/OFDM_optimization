from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY, INTEGER, maximize, OptimizationStatus

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import matlab.engine
import scipy.integrate as integrate

# ------------------------------------------- Constants ------------------------------------------- #
TOTAL_POWER                 = 1
MIN_POWER_CHUNK             = 0.0000001
NUM_OF_PILOTS_IN_FREQ_RANGE = 64
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

# ------------------------------------------- Functions ------------------------------------------- #
def plotCandidate(candidate, fitness, correlationVector):

    # Plot the correlation function
    plt.subplot(2, 1, 1)
    plt.cla()
    plt.axis([1.1 * min(deltaT_list), 1.1 * max(deltaT_list), 0, 1.1 * max(correlationVector)])
    plt.title("CRLB = " + str(fitness) + "\n")
    plt.plot(deltaT_list, correlationVector)
    plt.ylabel('Correlation')
    plt.xlabel('Time')

    # Plot the frequency ranges
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.axis([CENTER_FREQS[0] - FREQ_RANGE_WIDTH/2 - FREQ_RANGE_WIDTH/10, CENTER_FREQS[NUM_OF_FREQ_RANGES - 1] + FREQ_RANGE_WIDTH/2 + FREQ_RANGE_WIDTH/10, 0, 1])
    plt.stem(allPilotFreqs, candidate, bottom=0, use_line_collection=True)
    plt.ylabel('Power [mW]')
    plt.xlabel('Frequency [Hz]')

    # Show the graph
    plt.draw()
    plt.pause(0.01)

def cramerRaoLowerBoundInverse():
        return (8 * (math.pi)**2 * xsum(x[i] * allPilotFreqs[i] for i in pilotIndexes) * SNR)


# ------------------------------------------- Main ------------------------------------------------ #
# Start the matlab engine
matlabEngine = matlab.engine.start_matlab()

# Find all pilot frequencies
allPilotFreqs = np.empty([0])
for i in range(NUM_OF_FREQ_RANGES):
    freqRange = np.arange(CENTER_FREQS[i] - FREQ_RANGE_WIDTH//2, CENTER_FREQS[i] + FREQ_RANGE_WIDTH//2, FREQ_RANGE_WIDTH/NUM_OF_PILOTS_IN_FREQ_RANGE)
    allPilotFreqs = np.concatenate([allPilotFreqs,freqRange])

matlab_freqBins = matlab.double(allPilotFreqs.tolist())

pilotIndexes = set(range(NUM_OF_PILOTS))

# Inititalze the model
model = Model()

# Add the model variables
# Each variable is an INTEGER that represents how many MIN_POWER_CHUNKs we assign to each pilot
x = [model.add_var(var_type=INTEGER, lb=0, ub=10000000) for j in range(NUM_OF_PILOTS)]

# Assign an objective function to the model
# We want to minimize the Cramer-Rao lower bound, but since it contains 1 / optimizationVariable
# We maximize 1 / cramerRaoLowerBoundInverse
model.objective = maximize(cramerRaoLowerBoundInverse())

# Constraint => Sum of all powers has to be equal to the total available power
model += xsum((x[j] * MIN_POWER_CHUNK) for j in range(NUM_OF_PILOTS)) == TOTAL_POWER

# Create a figure window
plt.figure(figsize=(40, 40))

# Run the optimization
status = model.optimize(max_seconds=60)

if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    # Put the resulting power values in a list
    resultPilotPowers = [var.x * MIN_POWER_CHUNK for var in model.vars]

    # Calculate the autocorrelation function of the resulting candidate
    acf_vec_output = matlabEngine.acf(deltaT_vector, matlab_freqBins, matlab.double(resultPilotPowers), nargout=1)
    acf_vec_output = [acf_vec_output[0][i].real for i in range(len(acf_vec_output[0]))]
    plotCandidate(resultPilotPowers, model.objective_value, acf_vec_output)

    # Save the graph
    plt.savefig("BestFoundCandidate_lin.png")

    # Print the solution
    print('-------------------------------------------- SOLUTION --------------------------------------------')
    print('Fitness : ' + str(model.objective_value))
    print('Pilots : ' + str(resultPilotPowers))

elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))