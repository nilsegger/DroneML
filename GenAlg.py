pygad_fitnesses = []
ga_fitnesses = []
num_rep = 5
for i in range(num_rep):

    function_inputs = [4, -2, 3.5, 5, -11, -4.7]  # Function inputs.
    desired_output = 42  # Function output.


    def fitness_func(ga_instance, solution, solution_idx):
        output = numpy.sum(solution * function_inputs)
        fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
        return fitness


    num_generations = 100  # Number of generations.
    num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.

    sol_per_pop = 20  # Number of solutions in the population.
    num_genes = len(function_inputs)

    last_fitness = 0


    def on_generation(ga_instance):
        global last_fitness
        fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        last_fitness = fitness
        # Record the fitness score
        pygad_fitnesses.append(fitness)


    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           fitness_func=fitness_func,
                           on_generation=on_generation)

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    prediction = numpy.sum(numpy.array(function_inputs) * solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))
    print("\nStart our GA: \n")


    def fitness_func2(individual, sol_idx):
        output = numpy.sum(numpy.array(function_inputs) * numpy.array(individual))
        fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
        return fitness


    def on_generation2(ga):
        fitness = ga.fitness(max(ga.population, key=ga.fitness))
        ga_fitnesses.append(fitness)


    # Define the GA
    ga = GeneticAlgorithm(
        population_size=20,
        chromosome_length=len(function_inputs),
        mutation_rate=0.1,
        generations=100,
        fitness_func=fitness_func2,
        selection_method='roulette',
        crossover_method='single',
        elitism_size=2,
        mutation_method='random_uniform',
        weight_range=(-4, 4)
    )

    # Run the GA
    best_solution, best_fitness = ga.run(callback=on_generation2)

    print("Best solution: ", best_solution)
    print("Best solution's fitness: ", best_fitness)
    prediction = numpy.sum(numpy.array(function_inputs) * best_solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
    print("\n End our Ga\n")

pygad_avg_fitnesses = numpy.mean(pygad_fitnesses, axis=0)  # Average over num_rep
ga_avg_fitnesses = numpy.mean(ga_fitnesses, axis=0)  # Average over num_rep

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(pygad_avg_fitnesses, label='PyGAD')
plt.plot(ga_avg_fitnesses, label='GA')
plt.xlabel('Generation')
plt.ylabel('Average Fitness Score')
plt.title('Average Fitness Scores Over Generations for PyGAD and GA')
plt.legend()
plt.grid()
plt.show()