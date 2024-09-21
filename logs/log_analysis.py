import re
from collections import Counter

# Path to the game log file
log_file_path = "game.log"

# Regular expressions to detect end-of-round and game events
end_of_round_pattern = re.compile(r"WRAPPING UP ROUND")
coin_collected_pattern = re.compile(r"picked up coin")
blown_up_pattern = re.compile(r"blown up by own bomb")

def analyze_log(file_path):
    # Initialize counters for different events
    event_counter = Counter()
    round_events = []

    # Read the log file line by line
    with open(file_path, 'r') as log_file:
        for line in log_file:
            # Check for coin collection
            if coin_collected_pattern.search(line):
                round_events.append('COINs_COLLECTED')
            
            # Check for self-detonation
            if blown_up_pattern.search(line):
                round_events.append('KILLED_SELF')
            
            # Check if the round is wrapping up
            if end_of_round_pattern.search(line):
                if round_events:
                    event_counter.update(round_events)
                    round_events = []  # Reset for the next round

    return event_counter

if __name__ == "__main__":
    # Analyze the log file and get the count of each event
    event_counts = analyze_log(log_file_path)

    # Display the results
    print("End-of-round event counts:")
    for event, count in event_counts.items():
        print(f"{event}: {count}")
