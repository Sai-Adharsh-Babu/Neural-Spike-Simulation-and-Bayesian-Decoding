import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import PoissonNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import neo
    import quantities as pq
    from elephant.statistics import time_histogram
    ELEPHANT_AVAILABLE = True
except Exception:
    ELEPHANT_AVAILABLE = False

from brian2 import NeuronGroup, PoissonGroup, SpikeMonitor, StateMonitor, ms, second, prefs, set_device, defaultclock, run, start_scope

rng = np.random.RandomState(42)
n_neurons = 40
n_trials = 200
trial_duration = 0.5
dt = 0.1e-3
stim_conditions = [0, 1]
frac_train = 0.7
baseline_rate = 5.0
delta_rate = 15.0
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

def simulate_population_poisson(n_neurons, rate_vec, duration, rng):
    spikes = []
    for r in rate_vec:
        n_expected = int(np.ceil(r * duration * 1.5))
        if r <= 0:
            spikes.append(np.array([]))
            continue
        isi = rng.exponential(1.0 / r, size=n_expected)
        t = np.cumsum(isi)
        t = t[t < duration]
        spikes.append(t)
    return spikes

all_spike_trains = []
labels = np.zeros(n_trials, dtype=int)
selective_mask = np.zeros(n_neurons, dtype=int)
selective_mask[: n_neurons // 2] = 1

for trial in range(n_trials):
    stim = rng.choice(stim_conditions)
    labels[trial] = stim
    rates = np.ones(n_neurons) * baseline_rate
    rates[selective_mask == 1] += delta_rate * (1.0 if stim == 1 else 0.2)
    rates[selective_mask == 0] += delta_rate * (1.0 if stim == 0 else 0.2)
    spikes = simulate_population_poisson(n_neurons, rates, trial_duration, rng)
    all_spike_trains.append(spikes)

def extract_features(spike_trains, trial_duration, n_neurons):
    n_trials = len(spike_trains)
    features = np.zeros((n_trials, 3 * n_neurons), dtype=float)
    for i, trial_spikes in enumerate(spike_trains):
        for j in range(n_neurons):
            times = trial_spikes[j]
            total = np.sum((times >= 0) & (times <= trial_duration))
            early = np.sum((times >= 0) & (times < trial_duration / 2))
            late = np.sum((times >= trial_duration / 2) & (times <= trial_duration))
            features[i, j] = total
            features[i, n_neurons + j] = early
            features[i, 2 * n_neurons + j] = late
    return features

X = extract_features(all_spike_trains, trial_duration, n_neurons)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=frac_train, random_state=42, stratify=y)

clf_poisson = PoissonNB()
clf_poisson.fit(X_train, y_train)
y_pred = clf_poisson.predict(X_test)
acc_poisson = accuracy_score(y_test, y_pred)

clf_gauss = GaussianNB()
clf_gauss.fit(X_train, y_train)
y_pred_g = clf_gauss.predict(X_test)
acc_gauss = accuracy_score(y_test, y_pred_g)

cv_score = np.mean(cross_val_score(clf_poisson, X, y, cv=5))
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

bins = np.linspace(0, trial_duration, 51)

def plot_raster_for_trial(trial_idx, spike_trains, n_neurons, trial_duration, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    spikes = spike_trains[trial_idx]
    for neuron_idx in range(n_neurons):
        times = spikes[neuron_idx]
        ax.vlines(times, neuron_idx + 0.5, neuron_idx + 1.5)
    ax.set_ylim(0.5, n_neurons + 0.5)
    ax.set_xlim(0, trial_duration)
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Raster: Trial {trial_idx} (label={labels[trial_idx]})")
    return ax

plt.figure(figsize=(8, 6))
plot_raster_for_trial(0, all_spike_trains, n_neurons, trial_duration)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "raster_trial0.png"), dpi=200)
plt.close()

def compute_psth(spike_trains, labels, condition, bins, neuron_idx):
    trials_idx = np.where(labels == condition)[0]
    counts = np.zeros((len(trials_idx), len(bins) - 1))
    for t_i, tr in enumerate(trials_idx):
        times = spike_trains[tr][neuron_idx]
        counts[t_i, :], _ = np.histogram(times, bins=bins)
    return counts.mean(axis=0) / (bins[1] - bins[0])

plt.figure(figsize=(8, 4))
neuron_to_plot = 0
p1 = compute_psth(all_spike_trains, labels, 1, bins, neuron_to_plot)
p0 = compute_psth(all_spike_trains, labels, 0, bins, neuron_to_plot)
plt.plot(0.5 * (bins[:-1] + bins[1:]), p0, label="stim 0")
plt.plot(0.5 * (bins[:-1] + bins[1:]), p1, label="stim 1")
plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")
plt.legend()
plt.title(f"PSTH neuron {neuron_to_plot}")
plt.savefig(os.path.join(out_dir, "psth_example.png"), dpi=200)
plt.close()

np.savez(os.path.join(out_dir, "sim_spikes_dataset.npz"), X=X, y=y, labels=labels)

if ELEPHANT_AVAILABLE:
    st = neo.SpikeTrain(all_spike_trains[0][0] * pq.s, t_start=0 * pq.s, t_stop=trial_duration * pq.s)
    th = time_histogram([st], bins=bins * pq.s)
