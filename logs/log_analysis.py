import re
from collections import Counter

log_file_path = "game.log"

# Regular expressions to detect specific events
end_of_round_pattern = re.compile(r"WRAPPING UP ROUND")
coin_collected_pattern = re.compile(r"picked up coin")
bomb_dropped_pattern = re.compile(r"drops bomb")
kill_pattern = re.compile(r"killed by bomb placed by Agent <my_tabQ_agent_v1>")
win_pattern = re.compile(r"Agent <my_tabQ_agent_v1> wins the round")
agent_survived_pattern = re.compile(r"Agent <my_tabQ_agent_v1> survived the round")
step_pattern = re.compile(r"STARTING STEP (\d+)")
agent_died_pattern = re.compile(r"Agent <my_tabQ_agent_v1> killed by")

def analyze_log(file_path):
    # Initialize counters for different events
    event_counter = Counter()
    total_rounds = 0
    total_wins = 0
    total_survived_rounds = 0
    total_survival_steps = 0

    # Variables to track the current round
    current_steps = 0
    agent_survived = True

    # Read the log file line by line
    with open(file_path, 'r') as log_file:
        for line in log_file:
            # Track steps to calculate survival time
            step_match = step_pattern.search(line)
            if step_match:
                current_steps = int(step_match.group(1))

            # Check if the agent was killed
            if agent_died_pattern.search(line):
                agent_survived = False
                event_counter['KILLED_SELF'] += 1

            # Check for coin collection
            if coin_collected_pattern.search(line):
                event_counter['COINS_COLLECTED'] += 1

            # Check for bomb drop
            if bomb_dropped_pattern.search(line):
                event_counter['BOMBS_DROPPED'] += 1

            # Check for kills by the agent
            if kill_pattern.search(line):
                event_counter['KILLS'] += 1

            # Check for wins by the agent
            if win_pattern.search(line):
                total_wins += 1

            # Check if the agent survived the round
            if agent_survived_pattern.search(line):
                total_survived_rounds += 1

            # Check if the round is wrapping up
            if end_of_round_pattern.search(line):
                total_rounds += 1
                # If the agent survived, add the steps survived in this round
                if agent_survived:
                    total_survival_steps += current_steps
                # Reset for next round
                agent_survived = True
                current_steps = 0

    return total_rounds, total_wins, total_survived_rounds, total_survival_steps, event_counter

if __name__ == "__main__":
    total_rounds, total_wins, total_survived_rounds, total_survival_steps, event_counts = analyze_log(log_file_path)

    win_rate = (total_wins / total_rounds) * 100 if total_rounds > 0 else 0
    survival_rate = (total_survived_rounds / total_rounds) * 100 if total_rounds > 0 else 0
    average_survival_steps = total_survival_steps / total_rounds if total_rounds > 0 else 0

    print(f"Total Rounds: {total_rounds}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Survival Rate: {survival_rate:.2f}%")
    print(f"Average Survival Steps: {average_survival_steps:.2f}")
    print(f"Total Coins Collected: {event_counts['COINS_COLLECTED']}")
    print(f"Total Bombs Dropped: {event_counts['BOMBS_DROPPED']}")
    print(f"Total Kills: {event_counts['KILLS']}")

# need to get better ways to test this data, so far the eye test of how the bot is doing is still better

# first time running it on 100 rounds before training the results are:
#Total Rounds: 100
#Win Rate: 0.00%
#Survival Rate: 0.00%
#Average Survival Steps: 350.98
#Total Coins Collected: 5000
#Total Bombs Dropped: 3395
#Total Kills: 0

# After training 400 rounds, then tested again on 100 rounds
#otal Rounds: 100
#Win Rate: 0.00%
#Survival Rate: 0.00%
#Average Survival Steps: 345.35
#Total Coins Collected: 5000
#Total Bombs Dropped: 3385
#Total Kills: 0