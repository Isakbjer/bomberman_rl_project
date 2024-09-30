import re
import matplotlib.pyplot as plt
import os

# Define the relative path to the log file
log_file = os.path.join("agent_code", "my_tabQ_agent_v1", "logs", "my_tabQ_agent_v1.log")

# Initialize counters and data storage
total_games = 0
wins = 0
survival_rate = []
coins_collected = []
enemies_killed = []

# Regular expressions to match log events
coin_pattern = re.compile(r"INFO: Coin collected!")
kill_pattern = re.compile(r"INFO: Opponent killed!")
death_pattern = re.compile(r"INFO: Agent died.*")
win_pattern = re.compile(r"INFO: End of round: Agent survived")
end_of_round_pattern = re.compile(r"INFO: End of round:")

# Read the log file and extract information
with open(log_file, 'r') as f:
    game_coins = 0
    game_kills = 0
    survived = False

    for line in f:
        # Detect if the agent survived (won the game)
        if win_pattern.search(line):
            wins += 1
            survived = True

        # Detect if the agent died
        if death_pattern.search(line):
            survived = False

        # Count coins collected
        if coin_pattern.search(line):
            game_coins += 1

        # Count opponents killed
        if kill_pattern.search(line):
            game_kills += 1

        # Detect end of game (when the agent dies or survives)
        if end_of_round_pattern.search(line):
            total_games += 1
            survival_rate.append(1 if survived else 0)
            coins_collected.append(game_coins)
            enemies_killed.append(game_kills)

            # Reset counters for the next game
            game_coins = 0
            game_kills = 0
            survived = False

# Calculate metrics
win_rate = wins / total_games if total_games > 0 else 0
avg_coins = sum(coins_collected) / total_games if total_games > 0 else 0
avg_kills = sum(enemies_killed) / total_games if total_games > 0 else 0
avg_survival_rate = sum(survival_rate) / total_games if total_games > 0 else 0

# Print the results
print(f"Total Games: {total_games}")
print(f"Average Coins Collected: {avg_coins:.2f}")
print(f"Average Enemies Killed: {avg_kills:.2f}")

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot average coins collected over time
plt.subplot(2, 1, 1)
plt.plot(coins_collected, label="Coins Collected")
plt.title('Coins Collected Per Game')
plt.xlabel('Game')
plt.ylabel('Coins')

# Plot average enemies killed over time
plt.subplot(2, 1, 2)
plt.plot(enemies_killed, label="Enemies Killed")
plt.title('Enemies Killed Per Game')
plt.xlabel('Game')
plt.ylabel('Kills')

"""
Doesnt work from this file as we dont have win event, so use the log_analysis for this
# Plot win rate over time
plt.subplot(2, 2, 4)
plt.plot([wins / (i+1) for i in range(total_games)], label="Win Rate")
plt.title('Win Rate Over Time')
plt.xlabel('Game')
plt.ylabel('Win Rate')

# Plot win rate and survival rate over time
plt.subplot(2, 2, 1)
plt.plot(survival_rate, label="Survival")
plt.title('Survival Rate Over Time')
plt.xlabel('Game')
plt.ylabel('Survival (1 = survived, 0 = died)')
"""
plt.tight_layout()
plt.show()