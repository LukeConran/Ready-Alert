import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv('data/sweep_results.csv')

# Strip % signs if present
for col in ['detection_rate', 'false_alarm_rate']:
    if df[col].dtype == object:
        df[col] = df[col].str.rstrip('%').astype(float)

thresholds = sorted(df['threshold'].unique())
alert_pcts = sorted(df['alert_pct'].unique())

def pivot(col):
    return df.pivot(index='alert_pct', columns='threshold', values=col).sort_index(ascending=False)

det  = pivot('detection_rate')
fa   = pivot('false_alarm_rate')
score = det - fa  # maximize this

# Pareto frontier
pareto_rows = []
for _, row in df.iterrows():
    dominated = ((df['detection_rate'] >= row['detection_rate']) &
                 (df['false_alarm_rate'] <= row['false_alarm_rate']) &
                 ((df['detection_rate'] > row['detection_rate']) |
                  (df['false_alarm_rate'] < row['false_alarm_rate'])))
    if not dominated.any():
        pareto_rows.append(row)
pareto = pd.DataFrame(pareto_rows).sort_values('false_alarm_rate')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('EAR Threshold Sweep — Drowsiness Detection', fontsize=15, fontweight='bold')

def heatmap(ax, data, title, cmap, fmt='.0f'):
    im = ax.imshow(data.values, aspect='auto', cmap=cmap,
                   vmin=data.values.min(), vmax=data.values.max())
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in data.columns], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in data.index], fontsize=7)
    ax.set_xlabel('EAR Threshold', fontsize=10)
    ax.set_ylabel('Alert %', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data.values[i, j], fmt),
                    ha='center', va='center', fontsize=6,
                    color='white' if data.values[i, j] < (data.values.max() * 0.6) else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

heatmap(axes[0, 0], det,   'Detection Rate (%)',   'Greens')
heatmap(axes[0, 1], fa,    'False Alarm Rate (%)', 'Reds')
heatmap(axes[1, 0], score, 'Score (Detection − False Alarm)', 'RdYlGn')

# Pareto frontier
ax = axes[1, 1]
ax.scatter(df['false_alarm_rate'], df['detection_rate'],
           alpha=0.25, s=15, color='steelblue', label='All combos')
ax.plot(pareto['false_alarm_rate'], pareto['detection_rate'],
        'ro-', markersize=6, linewidth=1.5, label='Pareto frontier')

# Annotate recommended point
rec = df[(df['threshold'] == 0.81) & (df['alert_pct'] == 0.32)].iloc[0]
ax.annotate('0.81 / 0.32\n(recommended)',
            xy=(rec['false_alarm_rate'], rec['detection_rate']),
            xytext=(rec['false_alarm_rate'] + 5, rec['detection_rate'] - 6),
            fontsize=8, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'))

ax.set_xlabel('False Alarm Rate (%)', fontsize=10)
ax.set_ylabel('Detection Rate (%)', fontsize=10)
ax.set_title('Pareto Frontier', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 85)
ax.set_ylim(60, 100)

plt.tight_layout()
plt.savefig('data/sweep_charts.png', dpi=150, bbox_inches='tight')
print("Saved data/sweep_charts.png")
plt.show()
