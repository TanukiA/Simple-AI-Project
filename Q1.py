from random import randint
import random
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Crossover method 1: generate children to fill the length after subtracting the number of parents from total population.
# For each crossover, the male and female indexes are randomly picked from the range of all parents' length.
# If this male and female index picked are not the same one, then they can be used as the current index where parents are chosen. 
# A child is obtained by concatenating the back part of male and front part of female where the splitting occur at exactly 
# middle of the list of parents. 
def crossover1(parents, pop):
	parents_length = len(parents)
	desired_length = len(pop) - parents_length 
	children = []
	while len(children) < desired_length:
		male = randint(0, parents_length-1)
		female = randint(0, parents_length-1)
		if male != female:
			male = parents[male]
			female = parents[female]
			half = int(len(male) / 2)
			child = male[:half] + female[half:]
			children.append(child)

	return children


# Crossover method 2: generate children to fill the length after subtracting the number of parents from total population.
# For each crossover, the male and female indexes are randomly picked from the range of all parents' length.
# If this male and female index picked are not the same one, then they can be used as the current index where parents are chosen.
# To form a child, unlike method 1, it takes a random integer from 1 to the length of children, and use it as a position to split 
# parents, instead of everytime splitting into half.
def crossover2(parents, pop):
	parents_length = len(parents)
	desired_length = len(pop) - parents_length 
	children = []
	while len(children) < desired_length:
		male = randint(0, parents_length-1)
		female = randint(0, parents_length-1)
		if male != female:
			male = parents[male]
			female = parents[female]
			pos = randint(1, len(male)-1)
			child = male[:pos] + female[pos:]
			children.append(child)

	return children

# Crossover method 3: generate children to fill the length after subtracting the number of parents from total population.
# For each individual involved, a random number from 0 to 3 is generated (4 parameters of vacation plan).
# For each individual, the male and female indexes are randomly picked from the range of all parents' length.
# If this male and female index picked are not the same one, then they can be used as the current index where parents are chosen.
# After that, if the random index assigned to current individual is less than 2 (either 0 or 1), then swapping of genes happen.
# Hence, the probability for this crossover to happen is 50%. 
# A child is obtained by concatenating the back part of male and front part of female where the splitting occur at exactly 
# middle of the list of parents.
def crossover3(parents, pop): 
	
	parents_length = len(parents)
	desired_length = len(pop) - parents_length 
	indexes = []
	children = []

	# generate random index for each individual's crossover
	for i in range(desired_length):
		indexes.append(randint(0,3))

	for i in range(len(indexes)):
		male = randint(0, parents_length-1)
		female = randint(0, parents_length-1)
		if male != female:
			male = parents[male]
			female = parents[female]
			# swap gene of parent 1 & parent 2 if the random index is 0 or 1
			if indexes[i] < 2:
				male[indexes[i]], female[indexes[i]] = female[indexes[i]], male[indexes[i]]
				half = int(len(male) / 2)
				child = male[:half] + female[half:]
				children.append(child)

	return children

# Mutation method 1: Mutation only happen for individual when the mutation probability passed (mutate variable)
# is greater than the randomly generated floating number between 0 and 1. 
# The new value of mutation is randomly generated within the value range of individual. It mutates by randomly choosing
# a gene from that particular individual.
def mutation1(parents, mutate):
	for individual in parents:
		if mutate > random.random():
			pos_to_mutate = randint(0, len(individual)-1)
			# this mutation is not ideal, because it restricts the range of possible values,
			# but the function is unaware of the min/max values used to create the individuals
			individual[pos_to_mutate] = randint(min(individual), max(individual))
	return parents

# Mutation method 2: Mutation only happen for individual when the mutation probability passed (mutate variable)
# is greater than the randomly generated floating number between 0 and 1. 
# 2 unique random numbers are generated from the range of individual. Then, the individual is mutated by
# swapping 2 of its genes according to the 2 random numbers used as the indexes. 
def mutation2(parents, mutate):
	for individual in parents:
		if mutate > random.random():
			pts = random.sample(range(0, len(individual)), 2)
			temp = individual[pts[0]]
			individual[pts[0]] = individual[pts[1]]
			individual[pts[1]] = temp
	return parents
	
# Mutation method 3: Mutation only happen for individual when the mutation probability passed (mutate variable)
# is greater than the randomly generated floating number between 0 and 1. 
# However, unlike method 1, the new value of mutation is generated within the value range of randomly chosen parents (instead of
# restricted to own individual). Min value and Max value may be picked from different parents. It then mutates by randomly 
# choosing a gene from current individual.
def mutation3(parents, mutate):
	p_index = randint(0, len(parents)-1)
	min_value = min(parents[p_index])
	max_value = max(parents[p_index])
	for individual in parents:
		if mutate > random.random():
			pos_to_mutate = randint(0, len(individual)-1)
			individual[pos_to_mutate] = randint(min_value, max_value)
	return parents

# Generate initial chromosome to reach total budget
# Generate for 4 parameters (tourist spot fee, transport fee, food price & hotel fee) 
# where the sum is equal to total budget
def individual(n, min, total):
	dividers = sorted(random.sample(range(1, total), n - 1))
	arr = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
	return arr

# Generate chromosomes for entire population
def population(pop_size, n, min, total):
	return [individual(n, min, total) for x in range(pop_size)]

# A function that aims to calculate the score value for each individual.
# In this case, a lower score indicates better solution, because the score is incremented to
# show how an individual varies from the desired solution. Difference between desired parameter
# and individual's value is added to the scores. If the value is close to the desired parameter,
# less score will be accumulated. 
def fitness_func(individual, food_price, trans_fee, total_spot_fee, hotel_fee):
	score = 0
	# difference between tourist spots fee & parameter set
	score += abs(total_spot_fee - individual[0])
	# difference between total transport fee & parameter set
	score += abs(trans_fee - individual[1])
	# difference between total food price & parameter set
	score += abs(food_price - individual[2])
	# difference between total hotel fee & parameter set
	score += abs(hotel_fee - individual[3])

	# When the fee exceeds the desired parameter or is 0, extra incrementation occurs
	if (individual[0] > total_spot_fee) or (individual[0] == 0):
		score += abs(total_spot_fee - individual[0])
	if (individual[1] > trans_fee) or (individual[1] == 0):
		score += abs(trans_fee - individual[1])
	if (individual[2] > food_price) or (individual[2] == 0):
		score += abs(food_price - individual[2])
	if (individual[3] > hotel_fee) or (individual[3] == 0):
		score += abs(hotel_fee - individual[3])

	return score * 0.01

# Method 1 for selection, crossover & mutation
def evolve_method1(pop, total_spot_fee, trans_fee, food_price, hotel_fee, retain=0.1, random_select=0.04, mutate=0.01):
	scored = [(fitness_func(x, food_price, trans_fee, total_spot_fee, hotel_fee), x) for x in pop]
	# Sort the population, so that the individuals with lower score value are arranged at the top
	# Individuals with lower scores are better
	scored = [x[1] for x in sorted(scored)]
	# How many top % parents to be required
	retain_length = int(len(scored)*retain)
	# Get the list of array of individuals as parents  
	parents = scored[0:retain_length] 
	
	# randomly add other individuals to promote genetic diversity
	for individual in scored[retain_length:]: 
		if random_select > random.random():
			parents.append(individual)

	# mutate some individuals
	parents = mutation1(parents, mutate)
	
	# crossover parents to create children
	children = crossover1(parents, pop)

	parents.extend(children)
	return parents

# Method 2 for selection, crossover & mutation
def evolve_method2(pop, total_spot_fee, trans_fee, food_price, hotel_fee, retain=0.2, random_select=0.04, mutate=0.04):
	scored = [(fitness_func(x, food_price, trans_fee, total_spot_fee, hotel_fee), x) for x in pop]
	# Sort the population, so that the individuals with lower score value are arranged at the top
	# Individuals with lower scores are better
	scored = [x[1] for x in sorted(scored)]
	# How many top % parents to be required
	retain_length = int(len(scored)*retain)
	# Get the list of array of individuals as parents
	parents = scored[0:retain_length]

	# randomly add other individuals to promote genetic diversity
	for individual in scored[retain_length:]: 
		if random_select > random.random():
			parents.append(individual)

	# swap mutation
	parents = mutation2(parents, mutate)

	# crossover
	children = crossover2(parents, pop)
	parents.extend(children)
	return parents

# Method 3 for selection, crossover & mutation
def evolve_method3(pop, total_spot_fee, trans_fee, food_price, hotel_fee, retain=0.3, random_select=0.04, mutate=0.07):
	scored = [(fitness_func(x, food_price, trans_fee, total_spot_fee, hotel_fee), x) for x in pop]
	# Sort the population, so that the individuals with lower score value are arranged at the top
	# Individuals with lower scores are better
	scored = [x[1] for x in sorted(scored)]
	# How many top % parents to be required
	retain_length = int(len(scored)*retain)
	# Get the list of array of individuals as parents 
	parents = scored[0:retain_length] 

	# randomly add other individuals to promote genetic diversity
	for individual in scored[retain_length:]: 
		if random_select > random.random():
			parents.append(individual)

	# mutation
	parents = mutation3(parents, mutate)

	# crossover
	children = crossover3(parents, pop)
	parents.extend(children)
	return parents

# calculate the vacation planner to be displayed 
def cal_planner(final, duration, spot_num, trans_freq):
	hotel = final[3] / (duration-1)
	tourist_spot = final[0] / spot_num
	food = final[2] / duration / 3
	trans_fee = final[1] / duration / trans_freq

	return int(trans_freq), int(spot_num), int(hotel), int(tourist_spot), int(food), int(trans_fee)

# calculate average fitness of a population
def avg_fitness(scores):
	return sum(scores) / len(scores)

st.title('Vacation Planner')
total = st.number_input('Your total budget (RM):', min_value=500, value=5000)
duration = st.slider('Vacation duration (day):', 2, 20, 5)
hotel_rate = st.slider('Hotel star rating:', 1, 5, 2)
food_per_meal = st.slider('Food budget per meal (RM):', 10, 80, 20)
spot_num = st.slider('Number of tourist spot to visit:', 1, 30, 8)
spot_fee = st.slider('Fee for each tourist spot (RM):', 10, 600, 200)
trans_per_trip = st.slider('Transport fee per trip (RM):', 5, 300, 80)
trans_freq = st.slider('Transport frequency per day:', 1, 10, 5)

n_generation = 700
avgFitness_history = []
bestValue_history = []
bestFitness_history = []
avgFitness_m1 = []
avgFitness_m2 = []
avgFitness_m3 = []
time_taken = []
method_avg_fitness = []
generation_num = []
for i in range(700):
	generation_num.append(i+1)

total_spot_fee = spot_fee * spot_num
trans_fee = trans_per_trip * trans_freq * duration
food_price = food_per_meal * 3 * duration
# hotel rating: [1 star = 100, 2 star = 150, 3 star = 250, 4 star = 300, 5 star = 350] 
hotel_stars = [100, 150, 250, 300, 350]
hotel_fee = hotel_stars[hotel_rate-1] * (duration-1)

# population size = 200
pop = population(200, 4, 1, total)

# repeat for 3 methods
for i in range(3):
	start = time.time()
	# repeats for 700 times
	for j in range(n_generation):
	
		if i == 0:
			# GA method 1
			new_pop = evolve_method1(pop, total_spot_fee, trans_fee, food_price, hotel_fee)
		elif i == 1:
			# GA method 2
			new_pop = evolve_method2(pop, total_spot_fee, trans_fee, food_price, hotel_fee)
		elif i == 2:
			# GA method 3
			new_pop = evolve_method3(pop, total_spot_fee, trans_fee, food_price, hotel_fee)

		scored = [(fitness_func(x, food_price, trans_fee, total_spot_fee, hotel_fee), x) for x in new_pop]
		# get the sorted population 
		new_pop = [x[1] for x in sorted(scored)]
		# get the sorted scores 
		scores = [x[0] for x in sorted(scored)] 
		value = avg_fitness(scores)
		# append average fitness of a population for all
		avgFitness_history.append(value)
		if i == 0:
			# append average fitness of a population for GA metod 1
			avgFitness_m1.append(value)
		elif i == 1:
			# append average fitness of a population for GA metod 2
			avgFitness_m2.append(value)
		elif i == 2:
			# append average fitness of a population for GA metod 3
			avgFitness_m3.append(value)

		# best value of a population
		bestValue_history.append(new_pop[0])
		# best fitness of a population
		bestFitness_history.append(scores[0])

	end = time.time() 
	time_taken.append(end-start)

	# iterate to find the best fitness value among all populations
	best_fitness = bestFitness_history[0]
	best_index = 0
	for x in range(len(bestFitness_history)):
		if bestFitness_history[x] < best_fitness:
			best_index = x
			best_fitness = bestFitness_history[x]

	st.write('Best fitness value of method ', i+1, ': ', best_fitness)
	method_avg_fitness.append(avg_fitness(avgFitness_history))
	best = bestValue_history[best_index]

	trans_freq, spot_num, hotel, tourist_spot, food, trans_fee = cal_planner(best, duration, spot_num, trans_freq)

	st.write('Vacation Planner:')
	st.write(pd.DataFrame({'Parameter': ['Money on-hand', 'Vacation duration', 'Hotel star rating', 
	'Tourist spots', 'One tourist spot', 'Food price', 'Transportation fees', 'Transport frequency'], 
	'Value': [total, duration, hotel, spot_num, tourist_spot, food, trans_fee, trans_freq]}))

	# Plot graph to show the number of generation and average fitness value of a certain method 
	plt.plot(generation_num, avgFitness_history, label='GA Method')
	plt.title('Genetic Algorithm Method Performance', fontsize=12)
	plt.xlabel('No. of Generation', fontsize=12)
	plt.ylabel('Average Fitness Value', fontsize=12)
	plt.grid(True)
	plt.legend()
	st.pyplot(plt)
	plt.cla() 

	# clear the history lists before going to next method
	avgFitness_history.clear()
	bestValue_history.clear()
	bestFitness_history.clear()

# Plot the comparison of all 3 GA methods in one graph
plt.plot(generation_num, avgFitness_m1, label='GA Method 1', color='red')
plt.plot(generation_num, avgFitness_m2, label='GA Method 2', color='green')
plt.plot(generation_num, avgFitness_m3, label='GA Method 3', color='blue')
plt.title('Comparison of 3 Genetic Algorithm Methods', fontsize=12)
plt.xlabel('No. of Generation', fontsize=12)
plt.ylabel('Average Fitness Value', fontsize=12)
plt.grid(True)
plt.legend()
st.pyplot(plt)

st.write('')
st.write(pd.DataFrame({'Method': ['Method 1', 'Method 2', 'Method 3'], 
'Average Fitness Value': [method_avg_fitness[0], method_avg_fitness[1], method_avg_fitness[2]], 
'Time Taken (s)': [time_taken[0], time_taken[1], time_taken[2]]}))
st.write('**Lower fitness value indicates BETTER result**')