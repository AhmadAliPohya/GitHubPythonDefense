import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

# Epsilon-greedy with recency penalty using decaying epsilon
def epsilon_greedy_with_decaying_epsilon(assignments, current_ratio, target_ratio, history, recency_threshold, epsilon):
    # Filter assignments with counters > 0
    valid_assignments = [a for a in assignments if a['counter'] > 0]
    if not valid_assignments:
        return current_ratio, None, None, assignments

    # Exploration vs exploitation decision
    if np.random.random() < epsilon:
        # Exploration: choose a random assignment
        selected_assignment = np.random.choice(valid_assignments)
    else:
        # Exploitation: choose the assignment with the smallest deviation, considering penalties
        for a in valid_assignments:
            a['penalty'] = 1.5 if a['od_pair'] in history else 0  # Recency penalty
            a['adjusted_deviation'] = abs(a['flight_duration'] - target_ratio) + a['penalty']

        selected_assignment = min(valid_assignments, key=lambda x: x['adjusted_deviation'])

    # Update counters and history
    selected_assignment['counter'] -= 1
    new_fc = current_ratio[1] + 1
    new_ratio = (current_ratio[0] * current_ratio[1] + selected_assignment['flight_duration']) / new_fc

    if len(history) >= recency_threshold:
        history.popleft()
    history.append(selected_assignment['od_pair'])

    return (new_ratio, new_fc), selected_assignment['flight_duration'], selected_assignment['od_pair'], assignments, history

# Main program
if __name__ == "__main__":

    # Generate structured assignments
    n_pairs = 10
    assignments_orig = [
        {
            'od_pair': f"OD{i}_OD{i + 1}",
            'flight_duration': np.random.uniform(2, 7),
            'counter': np.random.randint(5, 15)
        }
        for i in range(n_pairs)
    ]

    for _ in range(10):

        assignments = deepcopy(assignments_orig)

        # Initial conditions
        current_ratio = (np.random.uniform(2,6), 1)  # Current FH and FC totals
        target_ratio = np.random.uniform(2,6)  # Desired FH/FC ratio
        recency_threshold = 10  # Track last 5 OD-pairs
        epsilon = 0.98  # Start with 80% exploration
        epsilon_decay = 1  # Reduce epsilon by 1% each iteration

        ratios_over_time = [current_ratio[0]]
        fhs_over_time = [current_ratio[0]]
        fcs_over_time = [current_ratio[1]]

        history = deque(maxlen=recency_threshold)

        iteration = 0
        while sum(a['counter'] for a in assignments) > 0:
            current_ratio, flight, od_pair, assignments, history = epsilon_greedy_with_decaying_epsilon(
                assignments, current_ratio, target_ratio, history, recency_threshold, epsilon
            )

            if flight is None:
                break  # No more valid flights

            # Log results
            ratios_over_time.append(current_ratio[0])
            fhs_over_time.append(flight)
            fcs_over_time.append(current_ratio[1])

            # Apply epsilon decay
            epsilon *= epsilon_decay
            iteration += 1

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(fcs_over_time, ratios_over_time, label='FH/FC Ratio', color='blue')
        plt.scatter(fcs_over_time, fhs_over_time, label='Flight Duration', color='green', alpha=0.5)
        plt.axhline(y=target_ratio, color='red', linestyle='--', label='Target FH/FC Ratio')
        plt.xlabel("Flight Cycles (FC)")
        plt.ylabel("FH/FC Ratio")
        plt.legend()
        plt.title("Epsilon-Greedy Assignment with Decaying Epsilon and Recency Penalty")
        plt.show()
