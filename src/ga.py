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
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria. Is it good? Who knows?
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
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m], coefficients))
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
                    # Ensure objects are supported or within 3 spaces of other blocks
                    if self.genome[y][x] in ["X", "?", "B", "o"]:
                        # Check if the tile below is solid (X, ?, B, T, |)
                        if y < height - 1 and self.genome[y + 1][x] not in ["X", "?", "B", "T", "|"]:
                            # Check if the block is within 3 spaces of another block horizontally
                            too_far = True
                            for dx in range(-3, 4):  # Check within 3 tiles horizontally
                                if x + dx >= 0 and x + dx < width:
                                    if self.genome[y][x + dx] in ["X", "?", "B", "T", "|"]:
                                        too_far = False
                                        break
                            if too_far:
                                continue  # Skip mutation if the block is too far from other blocks
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
        # Create base level with ground
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width  # Initial solid ground - will stay solid
        g[14][0] = "m"  # Mario's start position
        g[7][-1] = "v"  # Flagpole
        for col in range(8, 14):
            g[col][-1] = "f"  # Flag
        for col in range(14, 16):
            g[col][-1] = "X"  # Solid ground

        # Create ground elevation changes starting from row 14 (one above solid ground)
        current_height = 14  # Start at one above ground level
        x = 1  # Start after Mario's position
        while x < width - 20:  # Leave space for end flag
            if random.random() < 0.3:  # 30% chance for elevation change
                change = random.choice([-1, 1])  # Go up or down
                new_height = clip(12, current_height + change, 14)  # Limit height change, keeping one row above ground
                
                # Create smooth transition
                length = random.randint(4, 8)  # Length of the elevated/lowered section
                if change == -1:  # Going down
                    for i in range(length):
                        if x + i < width - 20:
                            g[new_height][x + i] = "X"  # New ground level
                            g[new_height + 1][x + i] = "-"  # Clear above
                else:  # Going up
                    for i in range(length):
                        if x + i < width - 20:
                            for y in range(new_height, 15):  # Stop before bottom row
                                g[y][x + i] = "X"  # Fill in below
                
                x += length
                current_height = new_height
            else:
                # Maintain current height
                g[current_height][x] = "X"
                x += 1

        # Add platforms and floating blocks
        for _ in range(35):
            x = random.randint(1, width - 4)
            # Find ground level at this position
            base_height = 14  # Start checking from row 14
            for y in range(14, 0, -1):
                if g[y][x] == "X":
                    base_height = y
                    break
            
            # Place platform at reasonable height above ground
            platform_y = random.randint(max(5, base_height - 4), base_height - 2)
            platform_length = random.randint(2, 4)
            platform_type = random.choice(["X", "B"])
            
            # Check for space
            space_clear = True
            for check_y in range(max(0, platform_y-1), min(height-1, platform_y+2)):  # Avoid bottom row
                for check_x in range(max(1, x-1), min(width-1, x+platform_length+1)):
                    if g[check_y][check_x] != "-":
                        space_clear = False
                        break
            
            if space_clear:
                for i in range(platform_length):
                    if x + i < width - 1:
                        g[platform_y][x + i] = platform_type

        # Add pipes with reduced height
        for _ in range(10):
            x = random.randint(1, width - 20)
            # Find ground level at this position
            for y in range(14, -1, -1):
                if g[y+1][x] == "X":  # Look for ground below current position
                    pipe_height = random.randint(0, 2)  # Reduced pipe height range
                    # Ensure we have enough space for the pipe
                    if y - pipe_height >= 0:
                        # Add pipe segments from bottom up
                        for py in range(y, y - pipe_height, -1):
                            g[py][x] = "|"
                        g[y - pipe_height][x] = "T"  # Add pipe top
                    break

        # Add decorative elements
        for _ in range(50):
            x = random.randint(1, width - 2)
            for y in range(13, 0, -1):  # Stop before bottom row
                if g[y+1][x] in ["X", "B"]:  # Check for support below
                    if g[y][x] == "-":  # Only place in empty space
                        if random.random() < 0.4:  # Question blocks and coins
                            element = random.choice(["?", "M", "o"])
                            g[y][x] = element
                        elif random.random() < 0.3 and g[y][x] == "-":  # Enemies on solid ground
                            g[y][x] = "E"
                    break

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
            penalties -= 2000  # Heavy penalty for unsolvable levels

        # Enhanced clutter penalties
        element_count = len(self.genome)
        if element_count > 25:  # Reduced threshold
            penalties -= 30 * (element_count - 25)  # Increased penalty

        # Penalize closely packed elements
        element_positions = [(de[0], de[1]) for de in self.genome]
        for i, (x1, type1) in enumerate(element_positions):
            for x2, type2 in element_positions[i+1:]:
                if abs(x1 - x2) < 3 and type1 == type2:  # Elements of same type too close
                    penalties -= 15

        # Penalize closely packed elements
        element_positions = [(de[0], de[1]) for de in self.genome]
        for i, (x1, type1) in enumerate(element_positions):
            for x2, type2 in element_positions[i+1:]:
                if abs(x1 - x2) < 3 and type1 == type2:  # Elements of same type too close
                    penalties -= 15
                    
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
        
        # Penalize tall pipes (height > 4)
        for de in self.genome:
            if de[1] == "7_pipe":  # Pipe design element
                h = de[2]  # Height of the pipe
                if h > 4:
                    penalties -= 10 * (h - 4)  # Penalize tall pipes
        
        # Penalize high stairs (height > 4)
        for de in self.genome:
            if de[1] == "6_stairs":  # Stairs design element
                h = de[2]  # Height of the stairs
                if h > 4:
                    penalties -= 10 * (h - 4)  # Penalize high stairs
        
        # Penalize clutter (too many design elements)
        if len(self.genome) > 50:  # Adjust this threshold as needed
            penalties -= 10 * (len(self.genome) - 50)  # Penalize levels with too many elements
        
        # Reward elevation changes
        elevation_changes = 0
        for de in self.genome:
            if de[1] in ["1_platform", "6_stairs"]:  # Platforms and stairs
                elevation_changes += 1
        elevation_reward = elevation_changes  # Reward levels with more elevation changes
        
        # Calculate final fitness
        self._fitness = (
            sum(map(lambda m: coefficients[m] * measurements[m], coefficients))
            + penalties
            + elevation_reward
        )
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, genome):
        # Chance to remove an element (new!)
        if random.random() < 0.15 and len(genome) > 0:
            to_remove = random.randint(0, len(genome) - 1)
            genome.pop(to_remove)
            return genome

        # Regular mutation with improved constraints
        if random.random() < 0.1 and len(genome) > 0:
            to_change = random.randint(0, len(genome) - 1)
            de = genome[to_change]
            x = de[0]
            de_type = de[1]
            
            # Check for nearby elements of same type
            nearby_similar = False
            for other_de in genome:
                if other_de != de and other_de[1] == de_type and abs(other_de[0] - x) < 3:
                    nearby_similar = True
                    break
            
            if nearby_similar:
                # If too crowded, increase position variance to spread things out
                x = offset_by_upto(x, width / 4, min=1, max=width - 2)
            
            # Type-specific mutations with improved constraints
            if de_type == "7_pipe":
                h = offset_by_upto(de[2], 1, min=0, max=3)  # Shorter pipes
                new_de = (x, de_type, h)
            
            elif de_type == "6_stairs":
                h = offset_by_upto(de[2], 1, min=2, max=3)  # More consistent stairs
                new_de = (x, de_type, h, 1)
            
            elif de_type == "1_platform":
                w = offset_by_upto(de[2], 1, min=3, max=5)  # More consistent platforms
                h = offset_by_upto(de[3], 1, min=2, max=4)
                new_de = (x, de_type, w, h, "X")
            
            else:  # Other elements (enemies, coins, blocks)
                new_de = de  # Keep as is if not a major structure
            
            genome.pop(to_change)
            heapq.heappush(genome, new_de)
        
        return genome


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
            
            # Ensure ground is solid except for intentional holes
            for x in range(width):
                base[height-1][x] = "X"
            
            # Process holes first
            holes = [de for de in self.genome if de[1] == "0_hole"]
            for de in holes:
                x = de[0]
                w = de[2]
                for x2 in range(w):
                    if 1 <= x + x2 < width - 2:  # Prevent holes at edges
                        base[height-1][x + x2] = "-"
            
            # Process other elements
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                x = de[0]
                de_type = de[1]
                
                if de_type == "7_pipe":
                    h = de[2]
                    if x < width - 1:  # Check if pipe fits
                        base[height - h - 2][x] = "T"
                        for y in range(height - h - 1, height - 1):
                            base[y][x] = "|"
                
                elif de_type == "6_stairs":
                    h = de[2]
                    # Build stairs from left to right, increasing in height
                    for x2 in range(h):
                        x_pos = x + x2
                        if 1 <= x_pos < width - 1:  # Stay within bounds
                            # Fill from ground up to current height
                            for y in range(x2 + 1):  # x2 + 1 ensures height increases with x
                                y_pos = height - 2 - y  # Start from just above ground
                                if 4 <= y_pos < height - 1:  # Keep stairs in reasonable range
                                    base[y_pos][x_pos] = "X"
                
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]
                    y_pos = height - h - 2
                    if 4 <= y_pos < height - 2:  # Keep platforms in reasonable range
                        for x2 in range(w):
                            if 1 <= x + x2 < width - 1:
                                base[y_pos][x + x2] = madeof
                
                elif de_type == "2_enemy":
                    if 1 <= x < width - 1 and base[height-2][x] != "|":  # Don't place enemies on pipes
                        base[height-2][x] = "E"
                
                elif de_type in ["3_coin", "4_block", "5_qblock"]:
                    y = de[2]
                    if 4 <= y < height - 2 and 1 <= x < width - 1:  # Keep in reasonable range
                        value = "o" if de_type == "3_coin" else \
                            "B" if de_type == "4_block" and de[3] else "X" if de_type == "4_block" else \
                            "M" if de_type == "5_qblock" and de[3] else "?"
                        base[y][x] = value
            
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # Generate fewer elements to reduce clutter
        elt_count = random.randint(8, 48)
        g = []
        
        # Add elevation changes through platforms and stairs first
        elevation_count = random.randint(8, 12)  # Increased from previous version
        for _ in range(elevation_count):
            x = random.randint(1, width - 15)  # Give more room for elements
            if random.random() < 0.6:  # 60% chance for stairs vs platforms
                # Stairs - always going right (dx = 1)
                height = random.randint(2, 3)  # Manageable height
                g.append((x, "6_stairs", height, 1))  # Force stairs to go left to right
            else:
                # Platform
                width_platform = random.randint(3, 7)
                height = random.randint(2, 4)  # Varied heights
                g.append((x, "1_platform", width_platform, height, "X"))  # More solid platforms
        
        # Add some holes (but not too many)
        hole_count = random.randint(2, 4)
        for _ in range(hole_count):
            x = random.randint(1, width - 10)
            hole_width = random.randint(2, 3)  # Smaller holes
            g.append((x, "0_hole", hole_width))
        
        # Add shorter pipes
        pipe_count = random.randint(3, 5)
        for _ in range(pipe_count):
            x = random.randint(1, width - 10)
            height = random.randint(0, 3)  # Reduced height range
            g.append((x, "7_pipe", height))
        
        # Add decorative elements
        decoration_count = random.randint(15, 20)  # Increased decorations
        for _ in range(decoration_count):
            x = random.randint(1, width - 10)
            y = random.randint(8, 12)
            if random.random() < 0.6:
                g.append((x, "3_coin", y))  # More coins
            else:
                g.append((x, "5_qblock", y, random.choice([True, False])))
        
        # Add enemies on ground and platforms
        enemy_count = random.randint(4, 7)
        for _ in range(enemy_count):
            x = random.randint(1, width - 10)
            g.append((x, "2_enemy"))
        
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
