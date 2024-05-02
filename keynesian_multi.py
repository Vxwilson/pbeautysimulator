import random, math
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt

"""
Multi-round Keynesian Beauty Contest Simulation

This code is put up in a short time frame and is not optimized for performance for the assignment MGC2230.
Further enhancement may be made to improve the readability and performance of the code. Bugs may exist in the code.
"""

class Actor:
    
    def __init__(self, k_level, learning_rate=1, strategy=None, multiplier=2/3):
        self.k_level = k_level
        self.learning_rate = learning_rate
        self.init_strategy(strategy)
        self.guess_history = []
        self.previous_error = None
        self.game_winning_guess_history = []
        self.multiplier = multiplier

    def init_strategy(self, strategy):
        if self.k_level == 0:
            self.strategy = "random"
            return
        if strategy is not None:
            self.strategy = strategy
            return

    def guess(self):

        previous_win = self.game_winning_guess_history[-1] if len(self.game_winning_guess_history) > 0 else None
        if self.k_level > 0:
            if previous_win is None:  # first round, guess based on k-level
                if self.k_level < 1:
                    self.guess_mean =  50 * (1 - self.k_level) + (50 * self.multiplier) * self.k_level + random.uniform(-5, 5)
                else:
                    # at each increasing level of k level, the actor will guess the average of the other actors' guesses
                    self.guess_mean = 50 * (self.multiplier) ** (self.k_level-1) + random.uniform(-5, 5)
            else:
                if self.strategy == "last_mean":
                    self.guess_mean = previous_win / self.multiplier
                elif self.strategy == "moving_average":
                    possible_length = min(3, len(self.game_winning_guess_history))
                    self.guess_mean = sum(self.game_winning_guess_history[-possible_length:]) / possible_length / self.multiplier
                elif self.strategy == "last_mean_two_thirds":
                    self.guess_mean = previous_win
                elif self.strategy == "last_mean_offset":
                    self.guess_mean = previous_win / self.multiplier
                elif self.strategy == "gradient_descent":
                    self.guess_mean = (self.guess_history[-1] + self.previous_error * self.learning_rate)
                elif self.strategy == "k_level_predicting":
                    possible_length = min(1, len(self.game_winning_guess_history)) # change to get the last 2 or 3
                    moving_average = sum(self.game_winning_guess_history[-possible_length:]) / possible_length
                    
                    last_k = math.log(moving_average / 50, self.multiplier) + 1
                    new_k = last_k + 0.6

                    self.guess_mean = 50 * (self.multiplier) ** (new_k-1)

        if self.strategy == "random":
            guess = random.randint(0, 100)
            self.guess_history.append(guess)
            return guess
        else:
            offset = self.guess_mean * random.uniform(-0.1, 0.1)
            self.guess_mean += offset

            guess = max(0, self.guess_mean * self.multiplier)
            self.guess_history.append(guess)
            return guess

    def update(self, difference_in_guess, winning_guess):
        self.previous_error = difference_in_guess
        self.game_winning_guess_history.append(winning_guess)

def simulate_multiround(num_actors, num_rounds, k_levels=[0, 1, 2], strategies=[], learning_rate=1, multiplier = 2/3):
    actors = populate_actors(num_actors, k_levels, strategies, learning_rate, multiplier)
    distributions = [None for _ in range(num_rounds)]
    average_guess_history = [None for _ in range(num_rounds)]

    for round in range(num_rounds):
        # calculate distribution of strategies 
        strategy_distribution = {}
        for actor in actors:
            if actor.strategy in strategy_distribution:
                strategy_distribution[actor.strategy] += 1
            else:
                strategy_distribution[actor.strategy] = 1
        # sort by string
        strategy_distribution = dict(sorted(strategy_distribution.items()))
        distributions[round] = strategy_distribution

        guesses = [actor.guess() for actor in actors]
        average = sum(guesses) / len(guesses)

        winner = min(actors, key=lambda actor: abs(actor.guess_history[-1] - average * multiplier))
        winner_guess = guesses[actors.index(winner)]
        
        winning_guess = average * multiplier
        average_guess_history[round] = winning_guess

        for i, actor in enumerate(actors):
            actor.update(guesses[i] - winning_guess, winning_guess)

        # eliminate 10 worst actors and add 10 new actors
        sorted_actors = sorted(actors.copy(), key=lambda actor: abs(actor.guess_history[-1] - winning_guess))
        actors = eliminate_and_duplicate_actors(sorted_actors, k_levels, strategies).copy()

        # some way to print out the winner strategy type, and the average guess
        # print(f"Round {round}: Winner is {winner.strategy} with guess {winner_guess:.2f} (winning: {winning_guess:.3f}) and rationality {winner.k_level:.2f}")
        
    # use matplotlib to plot the distribution of strategies as plots over time
    graph_dist_over_time(distributions, strategies, average_guess_history)

def populate_actors(num_actors_per_level, k_levels, strategies, learning_rate=0.2, multiplier=2/3):
    actors = []
    if 0 in k_levels:
        # append k-level * number of strategies random actors
        for _ in range(num_actors_per_level * (len(k_levels) - 1)):
            actors.append(Actor(0, strategy="random", learning_rate=learning_rate, multiplier = multiplier))
            # print(f'appended actor with strategy {"random"} and k-level 0')

    for strategy in strategies:
        for k in k_levels:
            if strategy == "random":
                for _ in range(num_actors_per_level):
                    actors.append(Actor(0, strategy=strategy, learning_rate=learning_rate, multiplier = multiplier))
                continue
            if k == 0:
                continue
            for _ in range(num_actors_per_level):
                actors.append(Actor(k, strategy=strategy, learning_rate=learning_rate, multiplier = multiplier))
    # print(f"Total actors: {len(actors)}"
    return actors

def eliminate_and_duplicate_actors(sorted_actors, k_levels, strategies):
    number_to_eliminate = (len(k_levels)-(1 if 0 in k_levels else 0)) * len(strategies)
    sorted_actors = sorted_actors[: -number_to_eliminate].copy()

    # new_actors = duplicate_best_of_actors(sorted_actors, number_to_eliminate)
    new_actors = []
    for i in range(number_to_eliminate):
        new_actors.append(deepcopy(sorted_actors[i]))
    sorted_actors += new_actors

    return sorted_actors

def graph_dist_over_time(distributions, strategies, average_guess_history):
    df = pd.DataFrame(distributions)

    # divide plot into 2 side by side
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})

    # set plot size
    plt.gcf().set_size_inches(10, 5)

    # df.plot(kind='area', stacked=True)
    df.plot(kind='area', stacked=True, ax=ax[0])

    ax[0].set_title('Strategy Distribution over Time')
    ax[0].set_xlabel('Round')
    ax[0].set_ylabel('Number of Actors')

    # # Put a legend to the right of the current axis
    ax[0].legend(bbox_to_anchor=(0.4, 0.2))

    # plot 2 side by side with winner guess, single line plot
    ax[1].plot(average_guess_history)
    ax[1].set(xlabel='Round', ylabel='Winning Guess',
           title='Winning Guess over Time')
    ax[1].grid()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_multiround(10, 25, k_levels=[0, 1, 2, 3, 4], strategies=[ "random", "last_mean", "last_mean_two_thirds", "gradient_descent", "k_level_predicting"], multiplier=2/3, learning_rate=0.2)
    # simulate_multiround(10, 30, k_levels=[ 1, 2, 3, 4], strategies=[  "last_mean", "last_mean_two_thirds", "gradient_descent", "k_level_predicting"], multiplier=2/3)
    #                                                         "random", "random", "random","random",  "random","random", "random", "random",
    #                                                         "moving_average","moving_average","moving_average","moving_average","moving_average","moving_average"], multiplier=2/3)
    # simulate_multiround(10, 500, k_levels=[ 1, 2, 3], strategies=[ "last_mean", "last_mean_two_thirds"], multiplier=2/3)

