# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
temp_df = pd.read_csv('./data/E096/E096_1.csv').set_axis(['Time', 'Pressure', 'Distance'], axis = 1)
# temp_df = temp_df.set_index('Time')

x = temp_df[temp_df.Time >= 800].Time / 1000
y1 = np.array(temp_df[temp_df.Time >= 800].Pressure / 100)
y2 = np.array(temp_df[temp_df.Time >= 800].Distance / 10)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'r-', label = 'Pressure')
ax2.plot(x, y2, 'b-', label = 'Distance')

ln1, label1 = ax1.get_legend_handles_labels()
ln2, label2 = ax2.get_legend_handles_labels()

lines = ln1 + ln2
labels = label1 + label2

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pressure (N/m2)')
ax2.set_ylabel('Distance (cm)')
ax1.legend(lines, labels, loc = 'upper left')
plt.show()

# %%
temp_df = pd.read_csv('./data/E075/E075_1.csv').set_axis(['Time', 'Pressure', 'Distance'], axis = 1)
# temp_df = temp_df.set_index('Time')

x = temp_df[temp_df.Time >= 800].Time / 1000
y1 = np.array(temp_df[temp_df.Time >= 800].Pressure / 100)
y2 = np.array(temp_df[temp_df.Time >= 800].Distance / 10)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'r-', label = 'Pressure')
ax2.plot(x, y2, 'b-', label = 'Distance')

ln1, label1 = ax1.get_legend_handles_labels()
ln2, label2 = ax2.get_legend_handles_labels()

lines = ln1 + ln2
labels = label1 + label2

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pressure (N/m2)')
ax2.set_ylabel('Distance (cm)')
ax1.legend(lines, labels, bbox_to_anchor = (.25, 1.2))
plt.show()

# %%
temp_df = pd.read_csv('./data/E086/E086_1.csv').set_axis(['Time', 'Pressure', 'Distance'], axis = 1)
# temp_df = temp_df.set_index('Time')

x = temp_df[temp_df.Time >= 800].Time / 1000
y1 = np.array(temp_df[temp_df.Time >= 800].Pressure / 100)
y2 = np.array(temp_df[temp_df.Time >= 800].Distance / 10)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'r-', label = 'Pressure')
ax2.plot(x, y2, 'b-', label = 'Distance')

ln1, label1 = ax1.get_legend_handles_labels()
ln2, label2 = ax2.get_legend_handles_labels()

lines = ln1 + ln2
labels = label1 + label2

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pressure (N/m2)')
ax2.set_ylabel('Distance (cm)')
# ax1.legend(lines, labels, loc = 'upper right')
ax1.legend(lines, labels, bbox_to_anchor = (1.3, 1.1))
plt.show()