# __OFDM signal optimization for distance estimation in AWGN channel__ 

## **Backgrund** ##
Distance estimation is a term that refers to the problem of estimating the distance between a transmitter and receiver element only by observing quantitative measures of the received signal at the receiving end.

One of the problems in designing a localization system as described above is constructing a localization signal that would provide high distance estimation accuracy. The signal of choice for the design of a such system is an OFDM in a single frequency band. In the next section, we will briefly explain how to value the quality of a localization signal employed in such a distance estimation scheme.

The OFDM signal is determined by:
- N --> the number of subcarriers used
- n --> the subcarrier index
- S --> the complex-valued baseband signal corresponding to the n-th subcarrier
- f --> the subcarrier frequency spacing

The aim of this project is to try to use metaheuristics to optimize the above described signal by finding the close-to-optimal subcarrier layout.

## **Genetic Algorithm approach** ##
This approach implemented in gen.py script starts by randomly creating a population of OFDM signal candidates with random subcarrier layout within the pre-selected constraints:
- Number of frequency ranges and their central frequencies
- Frequency range width
- Number of subcarriers per frequency range
- SNR
- Total available power

After generating the initial population the algorithm executes the following steps:
- Find the best candidates from the current population based on the fitness function
- Use those candidates to create a new population using crossover
- Mutate sa certain number of candidates in the new population
- Repeat for the desired number of generations

The genetic algortihm can be adjusted by changing the following configuration parameters at the top of the gen.py script:
- NUMBER_OF_GENERATIONS&emsp;&emsp;&emsp;&emsp;= 10000 --> Number cycles performed
- POPULATION_SIZE&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; = 50&emsp;&emsp; --> Number of candidates in each population
- BEST_PARENTS_FACTOR&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; = 5&emsp;&emsp;&emsp;--> Number of best parents to be used for the new generation
- MAX_NUM_OF_MUTATION_SWAPS&emsp; = 64&emsp;&emsp; --> Number of possible changes during one mutation
- MUTATION_TRESHOLD&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; = 15&emsp;&emsp; --> Chance of mutation in percentages

For the fitness function we chose the Ziv-Zakai Bound which is able to capture the performance quality of the distance estimation schemes because it takes into account the signal’s autocorrelation function’s properties. It is a lower bound on the variance of the distance estimation, and its mathematical expression looks like the following:

![Alt text](ziv-zakai-bound.png?raw=true "Ziv-Zakai Bound")

Where T is the observation interval, Q(x) is the q-function (the probability that a standard normal random variable takes a value larger than x standard deviations), and Re{ACF} is the real part of the autocorrelation function. The Ziv-Zakai bound could be used to differentiate two OFDM signals with a different set of parameters according to their distance estimation performance for the given SNR.

## **Simulated Annealing approach** ##
This approach implemented in sim_anne.py script starts by randomly creating a candidate for OFDM signal candidates with random subcarrier layout within the pre-selected constraints which are the same as in the case of the genetic algortihm. After creating the initial candidate the algorithm performs the following steps:
- Mutate the current candidate to create the new candidate
- Calculate the fitness of both candidates (same fitness function as for the genetic algorithm)
- If the new candidate is better then the current one, accept the new one
- If the new one is worse than the current one accept is based on the probability p ~ current temperature
- Decrease the current temperature
- Repeat until the current temperature reaches the target temperature

The Simulated Annealing algortihm can be adjusted by changing the following configuration parameters at the top of the sim_anne.py script:
- INITIAL_TEMP&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; = 10000&emsp;&emsp; --> Number of candidates in each population
- FINAL_TEMP&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; = 0.00001&emsp;--> Number of best parents to be used for the new generation
- MAX_NUM_OF_MUTATION_SWAPS&emsp;= 64&emsp;&emsp;&emsp;&emsp;--> Number of possible changes during one mutation
- COOLING_CONST&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;= 0.999&emsp;&emsp; --> Chance of mutation in percentages