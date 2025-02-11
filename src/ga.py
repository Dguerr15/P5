import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        
        # Default fitness coefficients
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            decorationPercentage=0.4,
            leniency=0.4,
            jumps=0.4,
            jumpVariance=0.4,
            linearity=-0.5,
            solvability=2.0  # Increase weight for solvability
        )
        
        # Penalties for unbeatable levels, impossible jumps, and blocking objects
        penalties = 0
        
        # Penalize unsolvable levels
        if measurements['solvability'] == 0:
            penalties -= 1000  # Heavy penalty for unsolvable levels
        
        # Penalize impossible jumps (gaps larger than 4 tiles)
        level = self.to_level()
        max_jump_distance = 4  # Mario's maximum jump distance
        for y in range(height):
            for x in range(width - 1):
                if level[y][x] == '-' and level[y][x + 1] == '-':
                    gap_size = 1
                    while x + gap_size < width and level[y][x + gap_size] == '-':
                        gap_size += 1
                    if gap_size > max_jump_distance:
                        penalties -= 10 * (gap_size - max_jump_distance)  # Penalize large gaps
        
        # Penalize blocking objects (pipes that are too tall)
        for y in range(height):
            for x in range(width):
                if level[y][x] == '|' or level[y][x] == 'T':  # Pipe segments
                    pipe_height = 0
                    while y + pipe_height < height and (level[y + pipe_height][x] == '|' or level[y + pipe_height][x] == 'T'):
                        pipe_height += 1
                    if pipe_height > 3:  # Pipes taller than 3 tiles are unclimbable
                        penalties -= 10 * (pipe_height - 3)  # Penalize tall pipes
        
        # Calculate final fitness
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m], coefficients)) + penalties
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, mutation_rate=0.01):
        for y in range(height):
            for x in range(1, width - 1):  # Avoid mutating the border columns
                if random.random() < mutation_rate:
                    # Ensure that pipes are not placed in the air
                    if self.genome[y][x] == '|' or self.genome[y][x] == 'T':
                        if y == height - 1 or self.genome[y + 1][x] != 'X':
                            continue  # Skip mutation if the pipe would be floating
                    self.genome[y][x] = random.choice(options)
        return self

    # Create zero or more children from self and other
    def generate_children(self, other):
        # Choose a random crossover point (by row or column)
        crossover_point = random.randint(1, width - 1)  # Crossover by column
        child1_genome = []
        child2_genome = []
        for y in range(height):
            # Child 1: Take left part from self and right part from other
            child1_genome.append(self.genome[y][:crossover_point] + other.genome[y][crossover_point:])
            # Child 2: Take left part from other and right part from self
            child2_genome.append(other.genome[y][:crossover_point] + self.genome[y][crossover_point:])
        
        # Create new individuals from the child genomes
        child1 = Individual_Grid(child1_genome)
        child2 = Individual_Grid(child2_genome)
        
        return child1, child2


    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        
        # Default fitness coefficients
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0  # Increase weight for solvability
        )
        
        # Penalties for unbeatable levels, impossible jumps, and blocking objects
        penalties = 0
        
        # Penalize unsolvable levels
        if measurements['solvability'] == 0:
            penalties -= 1000  # Heavy penalty for unsolvable levels
        
        # Penalize impossible jumps (gaps larger than 4 tiles)
        level = self.to_level()
        max_jump_distance = 4  # Mario's maximum jump distance
        for y in range(height):
            for x in range(width - 1):
                if level[y][x] == '-' and level[y][x + 1] == '-':
                    gap_size = 1
                    while x + gap_size < width and level[y][x + gap_size] == '-':
                        gap_size += 1
                    if gap_size > max_jump_distance:
                        penalties -= 10 * (gap_size - max_jump_distance)  # Penalize large gaps
        
        # Penalize blocking objects (e.g., pipes that are too tall)
        for y in range(height):
            for x in range(width):
                if level[y][x] == '|' or level[y][x] == 'T':  # Pipe segments
                    pipe_height = 0
                    while y + pipe_height < height and (level[y + pipe_height][x] == '|' or level[y + pipe_height][x] == 'T'):
                        pipe_height += 1
                    if pipe_height > 4:  # Pipes taller than 4 tiles are unclimbable
                        penalties -= 10 * (pipe_height - 4)  # Penalize tall pipes
        
        # Penalize right-to-left stairs and reward elevation changes and smaller stairs
        elevation_changes = 0
        stair_penalties = 0
        stair_rewards = 0
        for de in self.genome:
            if de[1] == "6_stairs":  # Stairs design element
                dx = de[3]  # Direction: -1 (right to left) or 1 (left to right)
                h = de[2]  # Height of the stairs
                if dx == -1:
                    stair_penalties -= 10  # Penalize right-to-left stairs
                if h > 4:
                    stair_penalties -= 5 * (h - 4)  # Penalize large stairs
                else:
                    stair_rewards += 5  # Reward smaller stairs
                elevation_changes += 1  # Count elevation changes
        
        # Reward levels with more elevation changes
        elevation_reward = elevation_changes * 2
        
        # Calculate final fitness
        self._fitness = (
            sum(map(lambda m: coefficients[m] * measurements[m], coefficients))
            + penalties
            + stair_penalties
            + stair_rewards
            + elevation_reward
        )
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, genome):
        if random.random() < 0.1 and len(genome) > 0:
            to_change = random.randint(0, len(genome) - 1)
            de = genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            # Add mutation logic based on de_type
            # ...
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            genome.pop(to_change)
            heapq.heappush(genome, new_de)
        return genome
        #     genome.pop(to_change)
        #     heapq.heappush(genome, new_de)
        # return genome


    def generate_children(self, other):
        if len(self.genome) == 0 or len(other.genome) == 0:
            # Handle the case where one of the parents has an empty genome
            return Individual_DE(self.genome), Individual_DE(other.genome)

        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part

        # Perform mutation and return individual children
        mutated_ga = self.mutate(ga)
        mutated_gb = self.mutate(gb)
        return Individual_DE(mutated_ga), Individual_DE(mutated_gb)
    
    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([0, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_DE

def tournament_selection(population, fitnesses, tournament_size=5):
    # Randomly select tournament_size individuals and return the best one
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    return max(tournament, key=lambda x: x[1])[0]

def generate_successors(population, fitnesses, elite_size=2):
    # Sort population by fitness in descending order
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    new_population = []

    # Elitist selection: Keep the top elite_size individuals
    for i in range(elite_size):
        new_population.append(sorted_population[i][0])

    # Perform crossover and mutation to fill the rest of the new population
    while len(new_population) < len(population):
        # Use tournament selection to select parents
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)

        # Generate children through crossover
        child1, child2 = parent1.generate_children(parent2)

        new_population.append(child1)
        if len(new_population) < len(population):
            new_population.append(child2)

    return new_population

def ga():
    # Ensure the levels directory exists
    if not os.path.exists("levels"):
        os.makedirs("levels")

    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                # Calculate fitness values for the current population
                fitnesses = [ind.fitness() for ind in population]
                next_population = generate_successors(population, fitnesses)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population

if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
