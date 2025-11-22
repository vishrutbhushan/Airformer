import matplotlib.pyplot as plt
import numpy as np

data = {
    'time_windows': {
        'windows': ['1-24h', '25-48h', '49-72h'],
        'old_mae': [15.07, 18.26, 19.59],
        'new_mae': [15.15, 18.02, 19.26],
        'old_rmse': [31.10, 35.93, 37.72],
        'new_rmse': [31.26, 35.46, 37.08],
    },
    'severe_events': {
        'old_mae': 38.0546,
        'new_mae': 38.0165,
        'old_rmse': 62.1589,
        'new_rmse': 62.0968,
    },
    'training_time': {
        'old': 47 * 255.6,
        'new': 40 * 517.0,
    },
    'parameters': {
        'old': 228_335,
        'new': 556_809,
    }
}

def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

def add_percentage_labels(ax, old_bars, new_bars, data_old, data_new):
    for i, (old_bar, new_bar) in enumerate(zip(old_bars, new_bars)):
        old_val = data_old[i]
        new_val = data_new[i]
        pct_change = ((old_val - new_val) / old_val) * 100
        
        new_height = new_bar.get_height()
        max_val = max(data_old + data_new)
        y_pos = new_height + (max_val * 0.08)
        
        color = 'green' if pct_change > 0 else 'red'
        symbol = '▼' if pct_change > 0 else '▲'
        ax.text(new_bar.get_x() + new_bar.get_width()/2., y_pos,
                f'{symbol}{abs(pct_change):.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=0.8))

x = np.arange(len(data['time_windows']['windows']))
width = 0.35
severe_x = len(data['time_windows']['windows']) + 0.5

fig1, ax1 = plt.subplots(figsize=(14, 8))
bars1 = ax1.bar(x - width/2, data['time_windows']['old_mae'], width, label='Old', color='#404040', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, data['time_windows']['new_mae'], width, label='New', color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1.5)
add_value_labels(ax1, bars1)
add_value_labels(ax1, bars2)
add_percentage_labels(ax1, bars1, bars2, data['time_windows']['old_mae'], data['time_windows']['new_mae'])

# Add severe events on the right
bars3 = ax1.bar(severe_x - width/2, data['severe_events']['old_mae'], width, color='#404040', alpha=0.7, edgecolor='black', linewidth=1.5)
bars4 = ax1.bar(severe_x + width/2, data['severe_events']['new_mae'], width, color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1.5)
add_value_labels(ax1, bars3)
add_value_labels(ax1, bars4)
add_percentage_labels(ax1, bars3, bars4, [data['severe_events']['old_mae']], [data['severe_events']['new_mae']])

ax1.set_ylabel('MAE (μg/m³)', fontweight='bold', fontsize=12)
ax1.set_title('Time Window Analysis - MAE (with Severe Events)', fontweight='bold', fontsize=14)
ax1.set_xticks([0, 1, 2, severe_x])
ax1.set_xticklabels(['1-24h', '25-48h', '49-72h', 'Severe\n(PM2.5>75)'], fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
fig1.tight_layout()
fig1.savefig('01_time_window_mae.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_time_window_mae.png")
plt.close(fig1)

# fig2, ax2 = plt.subplots(figsize=(14, 7))
# bars1 = ax2.bar(x - width/2, data['time_windows']['old_rmse'], width, label='Old', color='#404040', alpha=0.7, edgecolor='black', linewidth=1.5)
# bars2 = ax2.bar(x + width/2, data['time_windows']['new_rmse'], width, label='New', color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1.5)
# add_value_labels(ax2, bars1)
# add_value_labels(ax2, bars2)
# add_percentage_labels(ax2, bars1, bars2, data['time_windows']['old_rmse'], data['time_windows']['new_rmse'])
# 
# # Add severe events on the right
# bars3 = ax2.bar(severe_x - width/2, data['severe_events']['old_rmse'], width, color='#404040', alpha=0.7, edgecolor='black', linewidth=1.5)
# bars4 = ax2.bar(severe_x + width/2, data['severe_events']['new_rmse'], width, color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1.5)
# add_value_labels(ax2, bars3)
# add_value_labels(ax2, bars4)
# add_percentage_labels(ax2, bars3, bars4, [data['severe_events']['old_rmse']], [data['severe_events']['new_rmse']])
# 
# ax2.set_ylabel('RMSE (μg/m³)', fontweight='bold', fontsize=12)
# ax2.set_title('Time Window Analysis - RMSE (with Severe Events)', fontweight='bold', fontsize=14)
# ax2.set_xticks([0, 1, 2, severe_x])
# ax2.set_xticklabels(['1-24h', '25-48h', '49-72h', 'Severe\n(PM2.5>75)'], fontsize=11)
# ax2.legend(fontsize=11)
# ax2.grid(axis='y', alpha=0.3, linestyle='--')
# fig2.tight_layout()
# fig2.savefig('02_time_window_rmse.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 02_time_window_rmse.png")
# plt.close(fig2)

# fig3, ax3 = plt.subplots(figsize=(10, 6))
# metrics = ['MAE', 'RMSE']
# old_vals = [data['severe_events']['old_mae'], data['severe_events']['old_rmse']]
# new_vals = [data['severe_events']['new_mae'], data['severe_events']['new_rmse']]
# x_pos = np.arange(len(metrics))
# bars1 = ax3.bar(x_pos - width/2, old_vals, width, label='Old', color='#404040', alpha=0.7, edgecolor='black', linewidth=1.5)
# bars2 = ax3.bar(x_pos + width/2, new_vals, width, label='New', color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1.5)
# add_value_labels(ax3, bars1)
# add_value_labels(ax3, bars2)
# add_percentage_labels(ax3, bars1, bars2, old_vals, new_vals)
# ax3.set_ylabel('Error (μg/m³)', fontweight='bold', fontsize=12)
# ax3.set_title('Severe Pollution Events (PM2.5 > 75) - Standalone', fontweight='bold', fontsize=14)
# ax3.set_xticks(x_pos)
# ax3.set_xticklabels(metrics, fontsize=11)
# ax3.legend(fontsize=11)
# ax3.grid(axis='y', alpha=0.3, linestyle='--')
# fig3.tight_layout()
# fig3.savefig('03_severe_events_standalone.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 03_severe_events_standalone.png")
# plt.close(fig3)

# fig4, ax4 = plt.subplots(figsize=(10, 6))
# times = [data['training_time']['old'], data['training_time']['new']]
# models = ['Old (47 epochs)', 'New (40 epochs)']
# bars = ax4.bar(models, times, color=['#404040', '#CCCCCC'], alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)
# for bar, time in zip(bars, times):
#     h = bar.get_height()
#     hr = int(time // 3600)
#     mn = int((time % 3600) // 60)
#     ax4.text(bar.get_x() + bar.get_width()/2., h, f'{h:.0f}s\n({hr}h {mn}m)',
#             ha='center', va='bottom', fontweight='bold', fontsize=10)
# 
# pct_time_change = ((times[1] - times[0]) / times[0]) * 100
# y_pos = max(times) * 1.02
# ax4.text(0.5, y_pos, f'Time increased: +{pct_time_change:.1f}% ▲', ha='center', va='bottom',
#         transform=ax4.transData, fontweight='bold', fontsize=11, color='red', 
#         bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.85, edgecolor='red', linewidth=1))
# 
# ax4.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=12)
# ax4.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
# ax4.grid(axis='y', alpha=0.3, linestyle='--')
# fig4.tight_layout()
# fig4.savefig('04_training_time.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 04_training_time.png")
# plt.close(fig4)

# Combined Training Time and Parameters Graph
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: Training Time
times = [data['training_time']['old'], data['training_time']['new']]
models = ['Old (47 epochs)', 'New (40 epochs)']
bars = ax6a.bar(models, times, color=['#404040', '#CCCCCC'], alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)
for bar, time in zip(bars, times):
    h = bar.get_height()
    hr = int(time // 3600)
    mn = int((time % 3600) // 60)
    ax6a.text(bar.get_x() + bar.get_width()/2., h, f'{h:.0f}s\n({hr}h {mn}m)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

pct_time_change = ((times[1] - times[0]) / times[0]) * 100
y_pos = max(times) * 1.02
ax6a.text(0.5, y_pos, f'Time increased: +{pct_time_change:.1f}% ▲', ha='center', va='bottom',
        transform=ax6a.transData, fontweight='bold', fontsize=11, color='red', 
        bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.85, edgecolor='red', linewidth=1))

ax6a.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=12)
ax6a.set_title('Training Time Comparison', fontweight='bold', fontsize=13)
ax6a.grid(axis='y', alpha=0.3, linestyle='--')

# Right subplot: Parameters
params = [data['parameters']['old'], data['parameters']['new']]
models_param = ['Old', 'New']
bars = ax6b.bar(models_param, params, color=['#404040', '#CCCCCC'], alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)
for bar, param in zip(bars, params):
    h = bar.get_height()
    mb = param / 1e6 * 4
    ax6b.text(bar.get_x() + bar.get_width()/2., h, f'{param/1e3:.0f}K\n({mb:.2f}MB)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

pct_param_increase = ((params[1] - params[0]) / params[0]) * 100
y_pos = max(params) * 1.02
ax6b.text(0.5, y_pos, f'Model growth: +{pct_param_increase:.1f}% ▲', ha='center', va='bottom',
        transform=ax6b.transData, fontweight='bold', fontsize=11, color='red',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax6b.set_ylabel('Parameters', fontweight='bold', fontsize=12)
ax6b.set_title('Model Parameter Size', fontweight='bold', fontsize=13)
ax6b.grid(axis='y', alpha=0.3, linestyle='--')

fig6.suptitle('Training Cost Analysis', fontweight='bold', fontsize=15, y=1.02)
fig6.tight_layout()
fig6.savefig('04_combined_training_cost.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_combined_training_cost.png")
plt.close(fig6)

# fig5, ax5 = plt.subplots(figsize=(10, 6))
# params = [data['parameters']['old'], data['parameters']['new']]
# models_param = ['Old', 'New']
# bars = ax5.bar(models_param, params, color=['#404040', '#CCCCCC'], alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)
# for bar, param in zip(bars, params):
#     h = bar.get_height()
#     mb = param / 1e6 * 4
#     ax5.text(bar.get_x() + bar.get_width()/2., h, f'{param/1e3:.0f}K\n({mb:.2f}MB)',
#             ha='center', va='bottom', fontweight='bold', fontsize=10)
# 
# pct_param_increase = ((params[1] - params[0]) / params[0]) * 100
# y_pos = max(params) * 1.02
# ax5.text(0.5, y_pos, f'Model growth: +{pct_param_increase:.1f}% ▲', ha='center', va='bottom',
#         transform=ax5.transData, fontweight='bold', fontsize=11, color='red',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# 
# ax5.set_ylabel('Parameters', fontweight='bold', fontsize=12)
# ax5.set_title('Model Parameter Size', fontweight='bold', fontsize=14)
# ax5.grid(axis='y', alpha=0.3, linestyle='--')
# fig5.tight_layout()
# fig5.savefig('05_parameters.png', dpi=300, bbox_inches='tight')
# print("✓ Saved: 05_parameters.png")
# plt.close(fig5)

print("\n✓ All graphs generated successfully!")
print("  - 01_time_window_mae.png")
print("  - 04_combined_training_cost.png")
