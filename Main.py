import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import operator

from collections import deque

from City import City
from Fitness import Fitness

from collections import deque

cityList = []

def minimal_tsp():
    return { "COMMENT"          : ""
           , "DIMENSION"        : None
           , "TYPE"             : None
           , "EDGE_WEIGHT_TYPE" : None
           , "CITIES"           : []}

def scan_keywords(tsp,tspfile):
    for line in tspfile:
        words   = deque(line.split())
        keyword = words.popleft().strip(": ")

        if keyword == "COMMENT":
            tsp["COMMENT"] += " ".join(words).strip(": ")
        elif keyword == "NAME":
            tsp["NAME"] = " ".join(words).strip(": ")
        elif keyword == "TYPE":
            tsp["TYPE"] = " ".join(words).strip(": ")
        elif keyword == "DIMENSION":
            tsp["DIMENSION"] = int(" ".join(words).strip(": "))
        elif keyword == "EDGE_WEIGHT_TYPE":
            tsp["EDGE_WEIGHT_TYPE"] = " ".join(words).strip(": ")
        elif keyword == "NODE_COORD_SECTION":
            break

def read_int(words):
    return int(words.popleft())

def read_city(words, cityList):
    x = float(words.popleft())
    y = float(words.popleft())
    cityList.append(City(x=int(x), y=int(y)))
    
def read_numbered_city_line(desired_number, words, cityList):
    city_number = read_int(words)
    if city_number == desired_number:
        return read_city(words,cityList)
    else:
        print("Missing or mislabeld city: expected {0}".format(desired_number))

def read_cities(tsp,tspfile):
    for n in range(1, tsp["DIMENSION"] + 1):
        line  = tspfile.readline()
        words = deque(line.split())
        if tsp["EDGE_WEIGHT_TYPE"] == "EUC_2D":
            tsp["CITIES"].append(read_numbered_city_line(n, words, cityList))
        elif tsp["EDGE_WEIGHT_TYPE"] == "GEO":
            tsp["CITIES"].append(read_numbered_city_line(n, words, cityList))
        else:
            print("Unsupported coordinate type: " + tsp["EDGE_WEIGHT_TYPE"])
            
def read_tsp_file(path):
    tspfile = open(path,'r')
    tsp     = minimal_tsp()
    scan_keywords(tsp,tspfile)
    read_cities(tsp,tspfile)
    tspfile.close()

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

# TODO for na plikach tsp
read_tsp_file('C:/Users/Krzysiek/Desktop/Studia/Algorytmy ewolucyjne/gr202.tsp')

route = geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

print(route)

# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)