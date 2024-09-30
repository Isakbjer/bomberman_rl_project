import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict

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

    survival_times = defaultdict(list)  # Tracks survival times for all agents

    # Read the log file line by line
    with open(file_path, 'r') as log_file:
        current_steps = 0
        for line in log_file:
            # Track steps to calculate survival time
            step_match = step_pattern.search(line)
            if step_match:
                current_steps = int(step_match.group(1))

            # Check if the round is wrapping up
            if end_of_round_pattern.search(line):
                # Determine the winners of the round
                alive_agents = [agent for agent, stats in agent_stats.items() if stats['alive']]

                if alive_agents:
                    if any(agent_stats[agent]['round_coins'] > 0 for agent in alive_agents):
                        max_coins = max(agent_stats[agent]['round_coins'] for agent in alive_agents)
                        round_winners = [agent for agent in alive_agents if agent_stats[agent]['round_coins'] == max_coins]
                        for winner in round_winners:
                            agent_stats[winner]['wins'] += 1
                    else:
                        # If no coins were collected, treat all alive agents equally as winners
                        for agent in alive_agents:
                            agent_stats[agent]['wins'] += 1

                # Update stats for all agents and reset for the next round
                for agent, stats in agent_stats.items():
                    if stats['alive']:
                        stats['survived_rounds'] += 1
                    stats['total_rounds'] += 1
                    stats['round_coins'] = 0  # Reset only the coins for the current round
                    survival_times[agent].append(current_steps)  # Track survival time
                    stats['alive'] = True  # Reset alive status for the next round

                continue  # Skip further checks for this line

            # Check if an agent died (but count only once per death event)
            if death_pattern.search(line):
                agent_name = re.search(r"Agent <(.+?)>", line).group(1)
                if agent_stats[agent_name]['alive']:  # Only count death once per round
                    agent_stats[agent_name]['deaths'] += 1
                    agent_stats[agent_name]['alive'] = False  # Mark as dead

            # Check for coin collection
            if coin_collected_pattern.search(line):
                agent_name = re.search(r"Agent <(.+?)>", line).group(1)
                agent_stats[agent_name]['total_coins'] += 1  # Track total coins collected
                agent_stats[agent_name]['round_coins'] += 1  # Track coins for the current round

    return agent_stats, survival_times

# Run analysis
if __name__ == "__main__":
  agent_stats, survival_times = analyze_log(log_file_path)

  # Plot metrics for each agent
  plt.figure(figsize=(15, 10))

  # Subplot 1: Plot survival times for each agent
  plt.subplot(2, 2, 1)
  for agent, times in survival_times.items():
      plt.plot(range(len(times)), times, label=f"Survival Time - {agent}")

  plt.title('Survival Time Over Time')
  plt.xlabel('Game')
  plt.ylabel('Survival Time (steps)')
  plt.legend()

  # Subplot 2: Plot win rate for each agent
  plt.subplot(2, 2, 2)
  for agent, stats in agent_stats.items():
      total_rounds = stats['total_rounds']
      win_rate = (stats['wins'] / total_rounds) * 100 if total_rounds > 0 else 0
      plt.bar(agent, win_rate)

  plt.title('Win Rate for Each Agent')
  plt.xlabel('Agent')
  plt.ylabel('Win Rate (%)')

  # Subplot 3: Plot survival rate for each agent
  plt.subplot(2, 2, 3)
  for agent, stats in agent_stats.items():
      total_rounds = stats['total_rounds']
      survival_rate = (stats['survived_rounds'] / total_rounds) * 100 if total_rounds > 0 else 0
      plt.bar(agent, survival_rate)

  plt.title('Survival Rate for Each Agent')
  plt.xlabel('Agent')
  plt.ylabel('Survival Rate (%)')

  # Subplot 4: Plot average coins collected for each agent
  plt.subplot(2, 2, 4)
  for agent, stats in agent_stats.items():
      total_rounds = stats['total_rounds']
      avg_coins = stats['total_coins'] / total_rounds if total_rounds > 0 else 0
      plt.bar(agent, avg_coins)

  plt.title('Average Coins Collected for Each Agent')
  plt.xlabel('Agent')
  plt.ylabel('Average Coins')

  # Display the plots
  plt.tight_layout()
  plt.show()