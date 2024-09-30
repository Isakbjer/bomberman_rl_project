import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to the log file within the logs folder
log_file_path = os.path.join("logs", "game.log")

# Regular expressions to detect specific events
end_of_round_pattern = re.compile(r"WRAPPING UP ROUND")
coin_collected_pattern = re.compile(r"picked up coin")
death_pattern = re.compile(r"blown up|killed by")
step_pattern = re.compile(r"STARTING STEP (\d+)")

def analyze_log(file_path):
    agent_stats = defaultdict(lambda: {
        'total_coins': 0,
        'round_coins': 0,
        'wins': 0,
        'survived_rounds': 0,
        'total_rounds': 0,
        'deaths': 0,
        'alive': True
    })

    survival_times = []
    total_rounds_played = 0

    # Read the log file line by line
    with open(file_path, 'r') as log_file:
        for line in log_file:
            # Check for survival time (steps)
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))

            # Check if the round is wrapping up
            if end_of_round_pattern.search(line):
                # Calculate survival time
                alive_agents = [agent for agent, stats in agent_stats.items() if stats['alive']]
                for agent in alive_agents:
                    survival_times.append(current_step)

                total_rounds_played += 1

                # Reset for next round
                for agent, stats in agent_stats.items():
                    stats['total_rounds'] += 1
                    stats['round_coins'] = 0
                    stats['alive'] = True

                continue

            # Check if an agent died
            if death_pattern.search(line):
                agent_name = re.search(r"Agent <(.+?)>", line).group(1)
                if agent_stats[agent_name]['alive']:  
                    agent_stats[agent_name]['deaths'] += 1
                    agent_stats[agent_name]['alive'] = False 

    return agent_stats, survival_times, total_rounds_played

# Run analysis
if __name__ == "__main__":
    agent_stats, survival_times, total_rounds_played = analyze_log(log_file_path)

    # Calculate and print statistics
    for agent, stats in agent_stats.items():
        win_rate = (stats['wins'] / total_rounds_played) * 100 if total_rounds_played > 0 else 0
        survival_rate = (stats['survived_rounds'] / total_rounds_played) * 100 if total_rounds_played > 0 else 0
        avg_survival_time = sum(survival_times) / total_rounds_played if total_rounds_played > 0 else 0

        print(f"Agent: {agent}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Survival Rate: {survival_rate:.2f}%")
        print(f"  Average Survival Time: {avg_survival_time:.2f} steps")
        print("------------------------------")

    # Plot the metrics
    plt.figure(figsize=(10, 6))

    # Plot survival times over time
    plt.subplot(3, 1, 1)
    plt.plot(range(len(survival_times)), survival_times, label="Survival Time")
    plt.title('Survival Time Over Time')
    plt.xlabel('Game')
    plt.ylabel('Survival Time (steps)')

    # Plot survival rate
    survival_rates = [stats['survived_rounds'] / total_rounds_played * 100 for agent, stats in agent_stats.items()]
    plt.subplot(3, 1, 2)
    plt.plot(range(len(survival_rates)), survival_rates, label="Survival Rate")
    plt.title('Survival Rate Over Time')
    plt.xlabel('Game')
    plt.ylabel('Survival Rate (%)')

    # Plot win rate
    win_rates = [stats['wins'] / total_rounds_played * 100 for agent, stats in agent_stats.items()]
    plt.subplot(3, 1, 3)
    plt.plot(range(len(win_rates)), win_rates, label="Win Rate")
    plt.title('Win Rate Over Time')
    plt.xlabel('Game')
    plt.ylabel('Win Rate (%)')

    plt.tight_layout()
    plt.show()
