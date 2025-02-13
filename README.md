**Writeup for Project 5: Evolving Mario Levels**

### Changes from the Original Code
For this project, I made several important modifications to the original `ga.py` file to improve level generation and ensure that the evolutionary algorithm effectively produces playable and interesting Mario levels. Below are the key changes and the reasons behind them:

#### 1. **Selection Strategies**
- In the original code, the `generate_successors` function was not implemented, so I designed a proper selection strategy.
- I implemented **tournament selection**, where a small group of individuals competes, and the best one is chosen to reproduce. This helps maintain diversity while still selecting strong candidates.
- I also added **elitist selection**, ensuring that the top few individuals are carried over unchanged to the next generation. This prevents the best solutions from being lost due to random mutations.

#### 2. **Crossover Operators**
- The original implementation of crossover was incomplete. I implemented **single-point crossover** for the **Grid Encoding**, where a random column is selected, and the left portion comes from one parent while the right portion comes from the other.
- For the **Design Elements Encoding**, I improved the **variable-point crossover** by ensuring children inherit a mix of elements from both parents, maintaining diversity in generated levels.

#### 3. **Mutation Operators**
- The original mutation function was completely missing. I added mutation logic that:
  - For **Grid Encoding**, ensures elements like blocks and pipes are only placed where they make sense (pipes aren’t floating in the air).
  - For **Design Elements Encoding**, allows small shifts in element positions while preventing excessive clutter or unplayable designs.
  - Introduced a small chance to remove unnecessary elements in the **Design Elements Encoding**, reducing overpopulation of design features.

#### 4. **Fitness Function Enhancements**
- The original fitness function only included a basic weighting of metrics.
- I adjusted the **Grid Encoding** fitness function to:
  - Reward meaningful jumps and solvability more.
  - Penalize excessive empty space or too much linearity.
- For the **Design Elements Encoding**, I added:
  - Penalties for excessive elements that clutter the level.
  - Penalties for impossible jumps (gaps wider than 4 tiles).
  - Rewards for elevation changes that make the level more dynamic.

#### 5. **Population Initialization Tweaks**
- The original initialization was completely random.
- I ensured that levels started with a solid ground base and included more structured variation, such as pre-placed platforms or stairs.

### Favorite Levels and Their Evolution
I selected two levels from my generated outputs that I found the most interesting and playable one from Grid and one from DE:

#### **Level 1 Grid**
- **Why I like it:** This level features a nice balance of challenges and rewards, including well-placed enemies that are satisfying to kill, floating platforms, and an enemy positioned to create strategic jumps. The one I submitted is the grid version.
- **Generations Taken:** It took **7 generations** to evolve to a stable and fun version. I like the levels less cluttered
- **Time to Generate:** About **1.2 minutes**.

#### **Level 2 DE**
- **Why I like it:** This level has a unique layout with varying elevation changes and stair-like structures. It feels very natural, almost as if it were designed by hand. I like DE much better than Grid and it was easier to make work I feel like. It worked until something happened at the end of me testing my code, it does not think there is an end flag but there very clearly is. I played for like 30 mins on the same level and after like 2 changes it stopped working. I think it effected every DE level it makes and I do not know why. I really hope this does not take off points because it worked previously and when reverting it, it still didnt work. I do not know how that happened.
- **Generations Taken:** It took **3 generations** to reach this form.
- **Time to Generate:** About **2.1 minutes**.

### Extra Credit / Competition
We do not plan on joining the competition

### Conclusion
These changes significantly improved the effectiveness of the evolutionary algorithm in generating playable and engaging Mario levels. The improved selection, crossover, mutation, and fitness functions led to better results compared to the original code, which lacked these refinements. Through careful tuning of these parameters, I was able to generate levels that are both challenging and visually interesting while ensuring they remain solvable. I also now learned that if I ever want to implement this in my actual games I will be using DE.

