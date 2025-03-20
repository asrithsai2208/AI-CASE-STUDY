CASE STUDY

Hogwarts Hogwarts is a complex environment filled with magical obstacles and dynamic elements. This case study applies Artificial Intelligence (AI) search algorithms to solve problems within Hogwarts, such as navigating the castle, traversing a Triwizard Maze, and customizing a magic wand.



## 1. Problem Description and Explanation

### Uniform Cost Search (UCS): Magical Item Retrieval

#### Narrative:

A student embarks on a journey to retrieve magical items such as potion ingredients and enchanted books while ensuring minimal energy expenditure and avoiding hazardous areas.

#### Task:

Use UCS to determine the least-cost path for collecting all required magical items efficiently.

#### Challenges:

- Defining an appropriate cost function that accounts for energy spent, danger levels, and time taken.
- Managing dynamically changing costs, including restricted areas or roaming Dementors.
- Introducing penalties for using restricted spells or areas.

### A\* Search: Spell Targeting

#### Narrative:

During a magical duel, a student employs the A\* search algorithm to determine the optimal position for casting a spell while avoiding counterattacks.

#### Task:

Implement A\* search with a heuristic that takes into account the distance to the target, spell range, and opponent’s movement.

#### Challenges:

- Designing a heuristic that effectively balances offensive and defensive strategies.
- Handling dynamically changing target positions in real-time.
- Incorporating environmental hazards that influence spell accuracy.

### Adversarial Search: Wizard Duel

#### Narrative:

Two students engage in a magical duel, each strategically countering their opponent's spells and movements.

#### Task:

Utilize the minimax algorithm with alpha-beta pruning to develop strategies for both duelists, considering offensive and defensive spellcasting tactics.

#### Challenges:

- Developing evaluation functions that prioritize health, spell success rate, and positioning.
- Managing simultaneous spellcasting and counteractions.
- Allowing players to deploy decoys or use advanced spells like Protego or Expelliarmus.

## 2. Algorithm / Flowchart / Game Tree

### Uniform Cost Search (UCS)

- Initialize a priority queue with the starting location and cost 0.
- Expand the lowest-cost node and mark it as visited.
- Evaluate all possible moves and calculate their costs.
- Push unvisited locations with their updated costs into the priority queue.
- Repeat until all items are collected or no valid path remains.

### A\* Search

- Initialize an open list containing the start position.
- Compute the heuristic value (f = g + h) where:
  - g: cost to reach the node.
  - h: estimated cost to the goal.
- Expand the node with the lowest f-value.
- If the target is reached, return the optimal path.
- Otherwise, push neighboring nodes into the open list and repeat.

### Minimax with Alpha-Beta Pruning

- Define the evaluation function for both players.
- Generate all possible moves for the active player.
- Recursively evaluate moves using the minimax algorithm.
- Apply alpha-beta pruning to eliminate unnecessary branches and improve efficiency.
- Return the optimal move for the active player.

## 3. Code

### UCS Output:

import heapq

class MagicalItemRetrieval:
def **init**(self, locations, start_location, items, restricted_areas):
self.locations = locations
self.start_location = start_location
self.items = items
self.restricted_areas = restricted_areas
self.visited = set()

    def calculate_cost(self, current_location, next_location):
        cost = self.locations[next_location]["energy"]
        if next_location in self.restricted_areas:
            cost += self.locations[next_location]["danger_level"]
        cost += self.locations[next_location]["time"]
        return cost

    def ucs(self):
        queue = []
        heapq.heappush(queue, (0, self.start_location, [], []))

        while queue:
            current_cost, current_location, collected_items, path = heapq.heappop(queue)
            if set(collected_items) == set(self.items):
                return current_cost, path
            if current_location in self.visited:
                continue
            self.visited.add(current_location)
            for next_location in self.locations:
                if next_location in self.visited:
                    continue
                next_collected_items = collected_items[:]
                if next_location in self.items and next_location not in collected_items:
                    next_collected_items.append(next_location)
                next_cost = current_cost + self.calculate_cost(current_location, next_location)
                heapq.heappush(queue, (next_cost, next_location, next_collected_items, path + [next_location]))
        return float("inf"), []

### A\* Output:

import heapq
import math

class DuelArena:
def **init**(self, grid_size, start, target, opponent_start, hazards=[]):
self.grid_size = grid_size
self.start = start
self.target = target
self.opponent_start = opponent_start
self.hazards = hazards

    def heuristic(self, position, opponent_position):
        distance_to_target = math.sqrt((position[0] - self.target[0])**2 + (position[1] - self.target[1])**2)
        distance_to_opponent = math.sqrt((position[0] - opponent_position[0])**2 + (position[1] - opponent_position[1])**2)
        opponent_penalty = max(0, 10 - distance_to_opponent)
        hazard_penalty = 5 if position in self.hazards else 0
        return distance_to_target + opponent_penalty + hazard_penalty

    def a_star_search(self):
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(self.start, self.opponent_start), 0, self.start, []))
        closed_list = set()

        while open_list:
            _, current_cost, current_position, path = heapq.heappop(open_list)
            if current_position == self.target:
                return current_cost, path + [current_position]
            if current_position in closed_list:
                continue
            closed_list.add(current_position)
            for move in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_position = (current_position[0] + move[0], current_position[1] + move[1])
                if next_position in closed_list:
                    continue
                new_cost = current_cost + 1
                heapq.heappush(open_list, (new_cost + self.heuristic(next_position, self.opponent_start), new_cost, next_position, path + [current_position]))
        return float("inf"), []

### Minimax Output:

class WizardDuel:
def **init**(self):
self.spells = ['Expelliarmus', 'Stupefy', 'Protego', 'Expulso', 'Reducto']

    def evaluate(self, state):
        return state[0] - state[1]

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state[0] <= 0 or state[1] <= 0:
            return self.evaluate(state)
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.spells:
                eval = self.minimax(state, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.spells:
                eval = self.minimax(state, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

## 4. Output

### UCS Output:

- Displays the optimal path to collect all items with the least energy expenditure.
- Outputs the total cost (sum of energy, danger, and time penalties).

### A\* Output:

- Shows the best movement path for the duelist to reach an optimal spellcasting position.
- Includes considerations for avoiding counterattacks and environmental hazards.

### Minimax Output:

- Determines the best spellcasting strategy based on the opponent’s likely responses.
- Outputs spell choices and the final duel outcome.
