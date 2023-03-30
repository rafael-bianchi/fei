#%%
import logging
import heapq
import random

#%%
class missionary_cannibal_problem:
    def __init__(self, initial_state=(0,0,3,3, 'R'), current_state=None, max_move: int = 2):
        # 0 left missionaries
        # 1 left cannibals
        # 2 right missionaries
        # 3 right cannibals
        # 4 boat position
        self.initial_state = initial_state
        self.current_state = current_state if current_state is not None else initial_state
        self.max_move = max_move

    def generate_possible_next_states(self):
        states = []
        if not self.is_end_state():
            for i in range(self.max_move + 1):
                for j in range(self.max_move + 1):
                    if i + j == 0 or i + j > self.max_move:
                        continue
                    
                    if self.current_state[4] == 'R':
                        if self.current_state[2] >= i and self.current_state[3] >= j:
                            yield missionary_cannibal_problem(initial_state=self.initial_state, current_state=(self.current_state[0] + i, self.current_state[1] + j, self.current_state[2] - i, self.current_state[3] - j, 'L'), max_move=self.max_move)
                    else:
                        if self.current_state[0] >= i and self.current_state[1] >= j:
                            yield missionary_cannibal_problem(initial_state=self.initial_state, current_state=(self.current_state[0] - i, self.current_state[1] - j, self.current_state[2] + i, self.current_state[3] + j, 'R'), max_move=self.max_move)

    def is_loss_state(self):
        assert self.current_state is not None

        return (self.current_state[0] != 0 and self.current_state[0] < self.current_state[1]) or \
               (self.current_state[2] != 0 and self.current_state[2] < self.current_state[3])

    def is_win_state(self):
        assert self.current_state is not None

        return self.current_state[0] == self.initial_state[2] and \
                self.current_state[1] == self.initial_state[3]

    def is_end_state(self):
        assert self.current_state is not None

        return self.is_loss_state() or self.is_win_state()
    
    def __hash__(self):
        return hash(self.current_state)

    def __str__(self):
        return f'{self.current_state[0]}M{self.current_state[1]}C {("🛶" if self.current_state[4] == "L" else "")}_____{("🛶" if self.current_state[4] == "R" else "")} M{self.current_state[2]}C{self.current_state[3]}'

    def __eq__(self, __value: object):
        if not isinstance(__value, missionary_cannibal_problem):
            return False
        
        return self.current_state == __value.current_state
    
    def __lt__(self, other):
        return self.current_state[0] + other.current_state[1] < other.current_state[2] + other.current_state[3]
    
#%%   
def heuristic(state: missionary_cannibal_problem):
    return abs(state.current_state[0] - state.initial_state[2]) + \
           abs(state.current_state[1] - state.initial_state[3])

def cost(path):
    return len(path)

def breadth_first_search(graph: missionary_cannibal_problem):
    if graph.is_win_state():
        return [graph]

    queue = [[graph]]
    visited = set()
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in visited:
            neighbours = [state for state in node.generate_possible_next_states()]
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour.is_win_state():
                    return new_path
            visited.add(node)
    
    return []

def depth_first_search(graph : missionary_cannibal_problem):
    if graph.is_win_state():
        return [graph]

    stack = [[graph]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node not in visited:
            neighbours =  node.generate_possible_next_states()
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                stack.append(new_path)
                if neighbour.is_win_state():
                    return new_path
            visited.add(node)
    
    return []
#%%
def greedy_best_first_search(graph: missionary_cannibal_problem):
    if graph.is_win_state():
        return [graph]

    visited = set()
    priority_queue = [(heuristic(graph), [graph])]
    heapq.heapify(priority_queue)

    while priority_queue:
        _, path = heapq.heappop(priority_queue)
        node = path[-1]

        if node not in visited:
            neighbours = [state for state in node.generate_possible_next_states()]
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                if neighbour.is_win_state():
                    return new_path

                heapq.heappush(priority_queue, (heuristic(neighbour), new_path))
                visited.add(node)

    return []

def a_star_search(graph : missionary_cannibal_problem):
    if graph.is_win_state():
        return [graph]

    visited = set()
    priority_queue = [(heuristic(graph) + cost([graph]), [graph])]
    heapq.heapify(priority_queue)

    while priority_queue:
        _, path = heapq.heappop(priority_queue)
        node = path[-1]

        if node not in visited:
            neighbours = [state for state in node.generate_possible_next_states()]
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                if neighbour.is_win_state():
                    return new_path

                heapq.heappush(priority_queue, (heuristic(neighbour) + cost(new_path), new_path))
                visited.add(node)

    return []

#%%

def main():
    
    print('######## Breadth First Search ########')
    count = 1
    initial_state = (0,0,3,3, 'R')
    for c in breadth_first_search(missionary_cannibal_problem(initial_state, max_move=2)):
        print(f'{count}-{c}')
        count +=1


    print('######## Depth First Search ########')
    count = 1
    initial_state = (0,0,3,3, 'R')
    for c in depth_first_search(missionary_cannibal_problem(initial_state, max_move=2)):
        print(f'{count}-{c}')
        count +=1

    print('######## Greedy Best First Search ########')
    count = 1
    initial_state = (0,0,3,3, 'R')
    for c in greedy_best_first_search(missionary_cannibal_problem(initial_state, max_move=2)):
        print(f'{count}-{c}')
        count +=1

    print('######## A* Search ########')
    count = 1
    initial_state = (0,0,3,3, 'R')
    for c in a_star_search(missionary_cannibal_problem(initial_state, max_move=2)):
        print(f'{count}-{c}')
        count +=1


if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    main()

# %%
