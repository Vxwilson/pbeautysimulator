import numpy as np
import matplotlib.pyplot as plt

"""
Single Round Keynesian Beauty Contest simulation
"""

class Actor:
    def __init__(self, rationality):
        self.rationality = rationality

    def guess(self):
        # Implement the logic for the actor's guess here
       
        if self.rationality == 0:
            return np.random.uniform(0, 100)
        
        if self.rationality < 1:
            return 50 * (1 - self.rationality) + 33 * self.rationality 
        
        # at each increasing level of rationality, the actor will guess the average of the other actors' guesses
        return 50 * (2/3) ** (self.rationality) * np.random.uniform(0.8, 1.2)
    
def keynesian_beauty_contest(average_k_level, actor_count=50):

    random_offset = min(2, average_k_level)

    actors = [Actor(rationality=average_k_level + np.random.uniform(-1 * random_offset, random_offset)) for _ in range(actor_count)]

    # Each actor makes a guess
    guesses = [actor.guess() for actor in actors]

    # The average of all guesses
    average_guess = np.mean(guesses)

    # The winner is the actor whose guess is closest to 2/3 of the average guess
    winner = min(actors, key=lambda actor: abs(actor.guess() - 2/3 * average_guess))
    winner_guess = guesses[actors.index(winner)]

    # print(f"Winner is {winner.__class__.__name__} with guess {winner_guess:.2f} and rationality {winner.get_rationality():.2f}")
    return winner, average_guess, winner.rationality, winner_guess

def plot(rationality=None, iteration_count=100):
    # rationality; 0, 0.5, 1, ..., 20
    if rationality is None:
        rationality =  np.arange(0, 20.1, 0.5)

    # using array of rationality, calculate the average rationality of the winner for each rationality level
    average_rationality = []
    final_average_guess = []
    final_average_winner_guess = []

    for rationality_level in rationality:
        print(f'Calculating for rationality level {rationality_level}...')
        winner_rationality = 0
        average_guess = 0
        average_winner_guess = 0

        for _ in range(iteration_count):
            result = keynesian_beauty_contest(rationality_level)
            winner_rationality += result[2]
            average_guess += result[1]
            average_winner_guess += result[3]

        average_rationality.append(winner_rationality/iteration_count)
        final_average_guess.append(average_guess/iteration_count)
        final_average_winner_guess.append(average_winner_guess/iteration_count)

    # plot the average rationality of the winner for each rationality level
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    axes[0].plot(rationality, average_rationality)
    axes[0].set(xlabel="Average rationality (k-level)", ylabel="Winner k-level")
    # plot x=y
    axes[0].plot(rationality, rationality, 'r--')
    # legend for first plot for blue and red lines
    axes[0].legend(['Average winner k-level', 'y=x'])

    # second plot, plot the average guess for each rationality level
    axes[1].plot(rationality, final_average_guess)
    axes[1].set(xlabel="Average rationality level (k-level)", ylabel="Average guess")
    # plot y=0
    axes[1].plot(rationality, np.zeros(len(rationality)), 'g--')
    # plot on top y = 50 * (2/3) ^ (k-1)
    axes[1].plot(rationality, 50 * (2/3) ** (rationality), 'r--')
    # legend for second plot for blue, red, green lines
    axes[1].legend(['Average guess', 'y=0', 'y=50 * (2/3) ^ (k)'])

    plt.show()

if __name__ == "__main__":
    plot(iteration_count=100)
