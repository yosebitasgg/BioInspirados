# Genetic Algorithm Functions

# Hyperparameter ranges
batch_size_list = [8, 12, 16, 20, 24, 80, 200, 240]
epoch_list = [8, 200, 500, 527, 652, 860, 1000]
learning_rate_list = [0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001]

def generate_population(population_size):
    """Generate initial population of chromosomes"""
    population = []
    for _ in range(population_size):
        chromosome = {
            "lr": np.random.choice(learning_rate_list),
            "batch_sz": np.random.choice(batch_size_list),
            "ep": np.random.choice(epoch_list)
        }
        population.append(chromosome)
    return population

def selection(population_fitness):
    """Roulette Wheel Selection"""
    # Convert fitness to selection probability (lower is better, so invert)
    fitness_array = np.array(population_fitness)
    # Invert fitness (lower error is better)
    inverted_fitness = 1.0 / (fitness_array + 1e-10)
    # Normalize to probabilities
    probabilities = inverted_fitness / inverted_fitness.sum()

    # Select two parents
    parent_indices = np.random.choice(len(population_fitness), size=2, replace=False, p=probabilities)
    return parent_indices

def crossover(parent1, parent2):
    """Uniform crossover between two parents"""
    child1 = {}
    child2 = {}

    for key in parent1.keys():
        if np.random.rand() < 0.5:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]

    return [child1, child2]

def mutation(chromosome, mutation_rate=0.3):
    """Mutate chromosome with given probability"""
    mutated = chromosome.copy()

    if np.random.rand() < mutation_rate:
        mutated["lr"] = np.random.choice(learning_rate_list)

    if np.random.rand() < mutation_rate:
        mutated["batch_sz"] = np.random.choice(batch_size_list)

    if np.random.rand() < mutation_rate:
        mutated["ep"] = np.random.choice(epoch_list)

    return mutated

print("Genetic Algorithm functions defined successfully!")
