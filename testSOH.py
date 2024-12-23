import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 1) Simulation parameters
# ------------------------
num_sims = 1000  # number of battery simulations
num_cycles = 100000  # max number of cycles per battery

# Target mean and std
target_mean = 10000
target_std = 2000

# Time durations for each cycle (same distribution as your original code).
t_durs = np.random.normal(4.75, 1, num_cycles)

# Compute loc and scale to match the target overall mean and std deviation
alpha = 1.0 / 900000.0
loc = alpha * t_durs

# Adjust scale to achieve the desired standard deviation
scale_adjustment_factor = target_std / (np.sqrt(num_sims) * np.std(loc))
scale = scale_adjustment_factor * loc

# -------------------------
# 2) Vectorized SoH updates
# -------------------------
# increments has shape = (num_sims, num_cycles)
raw_increments = np.random.normal(loc, scale, size=(num_sims, num_cycles))

# Clip negative increments to zero so SoH never "heals":
increments = np.clip(raw_increments, a_min=0, a_max=None)

# csum[i, :] is the cumulative sum of increments for the i-th battery
csum = np.cumsum(increments, axis=1)

# SoH[i, :] = 1 - csum[i, :]
soh = 1.0 - csum

# -------------------------
# 3) Identify failure times
# -------------------------
failure_mask = (soh < 0.0)
failure_indices = np.argmax(failure_mask, axis=1)  # 1D array of length num_sims

# If a row never fails, argmax returns 0, but that might be a "false positive."
# We detect rows that never fail:
never_failed = np.all(soh >= 0.0, axis=1)
# For those that never fail, we can mark the fail index as num_cycles
failure_indices[never_failed] = num_cycles

# -----------------------------
# 4) Plot 10 random SoH vs fcs
# -----------------------------
plt.figure(figsize=(8, 6))

# Pick 10 random simulations
np.random.seed(123)  # for reproducibility, optional
random_indices = np.random.choice(num_sims, size=10, replace=False)

for sim_idx in random_indices:
    # The SoH for this simulation
    sim_soh = soh[sim_idx]

    # The cycle at which it fails (0-based index)
    fail_idx = failure_indices[sim_idx]

    # If never failed, we'll plot all cycles
    if fail_idx == num_cycles:
        end_idx = num_cycles
    else:
        end_idx = fail_idx

    # We'll plot from cycle 1..end_idx (in 1-based cycles)
    fcs = np.arange(1, end_idx + 1)  # 1..end_idx
    plt.plot(fcs, sim_soh[:end_idx], alpha=0.8, label=f'Sim {sim_idx}')

plt.xlabel("Cycle count (fcs)")
plt.ylabel("State of Health (SoH)")
plt.title("10 Random Battery SoH Trajectories (No Healing)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # move legend outside
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Histogram of failure times
# -----------------------------
plt.figure()
# Convert 0-based indices to "cycle number" by adding +1 if you want 1-based
failure_cycles = failure_indices + 1
plt.hist(failure_cycles, bins=50, alpha=0.7, edgecolor='k')
plt.xlabel("Cycles to Failure")
plt.ylabel("Count")
plt.title("Distribution of Failure Times (No Healing)")
plt.show()

# -------------------------
# 6) Print summary statistics
# -------------------------
mean_failure_time = np.mean(failure_cycles)
std_failure_time = np.std(failure_cycles)
print("Mean failure time:", mean_failure_time)
print("Std  failure time:", std_failure_time)

# Verify results
assert abs(mean_failure_time - target_mean) < 1e-2, "Mean is not matching target"
assert abs(std_failure_time - target_std) < 1e-2, "Std deviation is not matching target"
