import re
import os
from collections import defaultdict

log_file = os.path.join("logs", "game.log")

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

    # Variables to track the current round
    current_steps = 0

    # Read the log file line by line
    with open(file_path, 'r') as log_file:
        for line in log_file:
            # Track steps to calculate survival time
            step_match = step_pattern.search(line)
            if step_match:
                current_steps = int(step_match.group(1))

            # Check if the round is wrapping up
            if end_of_round_pattern.search(line):
                # Determine the winners of the round
                alive_agents = [agent for agent, stats in agent_stats.items() if stats['alive']]
                if len(alive_agents) == 1:
                    # If only one agent is alive, they win
                    agent_stats[alive_agents[0]]['wins'] += 1
                else:
                    # If multiple agents are alive, find the one(s) with the most coins
                    max_coins = max(agent_stats[agent]['round_coins'] for agent in alive_agents)
                    round_winners = [agent for agent in alive_agents if agent_stats[agent]['round_coins'] == max_coins]
                    for winner in round_winners:
                        agent_stats[winner]['wins'] += 1

                # Update stats for all agents and reset for the next round
                for agent, stats in agent_stats.items():
                    if stats['alive']:
                        stats['survived_rounds'] += 1
                    stats['total_rounds'] += 1
                    stats['round_coins'] = 0  # Reset only the coins for the current round
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

    return agent_stats

if __name__ == "__main__":
    agent_stats = analyze_log(log_file_path)

    for agent, stats in agent_stats.items():
        total_rounds = stats['total_rounds']
        survival_rate = (stats['survived_rounds'] / total_rounds) * 100 if total_rounds > 0 else 0
        win_rate = (stats['wins'] / total_rounds) * 100 if total_rounds > 0 else 0
        avg_coins = stats['total_coins'] / total_rounds if total_rounds > 0 else 0

        print(f"Agent: {agent}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Survival Rate: {survival_rate:.2f}%")
        print(f"  Average Coins Collected: {avg_coins:.2f}")
        print(f"  Total Deaths: {stats['deaths']}")
        print(f"  Total Rounds Played: {total_rounds}")
        print(f"  Total Wins: {stats['wins']}")
        print("------------------------------")
"""
Agent: my_tabQ_agent_v1
  Win Rate: 16.00%
  Survival Rate: 34.00%
  Average Coins Collected: 12.48
  Total Deaths: 33
  Total Rounds Played: 50
  Total Wins: 8
------------------------------
Agent: rule_based_agent_1
  Win Rate: 22.00%
  Survival Rate: 50.00%
  Average Coins Collected: 13.36
  Total Deaths: 25
  Total Rounds Played: 50
  Total Wins: 11
------------------------------
Agent: rule_based_agent_0
  Win Rate: 28.00%
  Survival Rate: 52.00%
  Average Coins Collected: 12.36
  Total Deaths: 24
  Total Rounds Played: 50
  Total Wins: 14
------------------------------
Agent: rule_based_agent_2
  Win Rate: 38.00%
  Survival Rate: 60.00%
  Average Coins Collected: 11.80
  Total Deaths: 20
  Total Rounds Played: 50
  Total Wins: 19
------------------------------

After 100 rounds of training, 50 new test games:
Agent: rule_based_agent_0
  Win Rate: 50.00%
  Survival Rate: 64.00%
  Average Coins Collected: 13.06
  Total Deaths: 18
  Total Rounds Played: 50
  Total Wins: 25
------------------------------
Agent: rule_based_agent_2
  Win Rate: 26.00%
  Survival Rate: 54.00%
  Average Coins Collected: 12.40
  Total Deaths: 23
  Total Rounds Played: 50
  Total Wins: 13
------------------------------
Agent: rule_based_agent_1
  Win Rate: 22.00%
  Survival Rate: 44.00%
  Average Coins Collected: 12.62
  Total Deaths: 28
  Total Rounds Played: 50
  Total Wins: 11
------------------------------
Agent: my_tabQ_agent_v1
  Win Rate: 8.00%
  Survival Rate: 26.00%
  Average Coins Collected: 11.92
  Total Deaths: 37
  Total Rounds Played: 50
  Total Wins: 4
------------------------------

After 700 more training rounds:
Agent: my_tabQ_agent_v1
  Win Rate: 18.00%
  Survival Rate: 32.00%
  Average Coins Collected: 12.90
  Total Deaths: 34
  Total Rounds Played: 50
  Total Wins: 9
------------------------------
Agent: rule_based_agent_1
  Win Rate: 30.00%
  Survival Rate: 56.00%
  Average Coins Collected: 12.02
  Total Deaths: 22
  Total Rounds Played: 50
  Total Wins: 15
------------------------------
Agent: rule_based_agent_0
  Win Rate: 28.00%
  Survival Rate: 60.00%
  Average Coins Collected: 13.30
  Total Deaths: 20
  Total Rounds Played: 50
  Total Wins: 14
------------------------------
Agent: rule_based_agent_2
  Win Rate: 30.00%
  Survival Rate: 62.00%
  Average Coins Collected: 11.78
  Total Deaths: 19
  Total Rounds Played: 50
  Total Wins: 15
------------------------------

"""
