import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# set default plotting style
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.linewidth'] = 1.5

# load data
df = pd.read_csv('metrics.csv')

step_data = np.stack([df['step'], df['train/loss_step']], axis=1)
epoch_data = np.stack([df['epoch'], df['train/loss_epoch']], axis=1)
epoch_data = epoch_data[~np.isnan(epoch_data).any(axis=1)]
epoch_val_data = np.stack([df['epoch'], df['val/loss']], axis=1)
epoch_val_data = epoch_val_data[~np.isnan(epoch_val_data).any(axis=1)]
epoch_val_best_data = np.stack([df['epoch'], df['val/loss_best']], axis=1)
epoch_val_best_data = epoch_val_best_data[~np.isnan(epoch_val_best_data).any(axis=1)]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

nan_idx = 2990

ax.plot(step_data[:, 0], step_data[:, 1], label='train loss (step)', color='blue', alpha=0.4)
ax.plot(epoch_data[:, 0] * nan_idx, epoch_data[:, 1], label='train loss (epoch)', color='black')
ax.plot(epoch_val_data[:, 0] * nan_idx, epoch_val_data[:, 1], label='val loss', color='red', alpha=0.7)
# ax.plot(epoch_val_best_data[:, 0] * nan_idx, epoch_val_best_data[:, 1], label='val loss best', color='green', alpha=0.7)

xticks = np.arange(0, 500000, 100000)
ax.set_xticks(xticks)
ax.set_xticklabels([f'{int(x/1000)}' for x in xticks])

ax.set_ylabel('Loss')
ax.set_xlabel('Data Iteration (k)')

plt.legend()
plt.savefig('loss.pdf', dpi=300)
plt.show()
