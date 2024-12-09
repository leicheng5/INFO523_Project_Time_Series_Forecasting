#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the data
folder_name_TimesNet = 'long_term_forecast_test_TimesNet_custom_ftMS_sl16_ll8_pl5_dm32_nh1_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_test_0'
folder_name_iTransformer = 'long_term_forecast_test_iTransformer_custom_ftMS_sl16_ll8_pl5_dm32_nh1_el1_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_test_0'

folder_path = './results/' + folder_name_iTransformer + '/'
metrics = np.load(folder_path + 'metrics.npy')
preds = np.load(folder_path + 'pred.npy')
trues = np.load(folder_path + 'true.npy')

# Metrics
mae, mse, rmse, mape, mspe = metrics
print(f"Metrics:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nMAPE: {mape}\nMSPE: {mspe}")

# Visualization

# 1. Plot predicted vs true values for a single sample
sample_index = 0  # Example: Visualize the first sample
plt.figure(figsize=(10, 6))
plt.plot(trues[sample_index], label='True Prices', linewidth=2, marker='o')
plt.plot(preds[sample_index], label='Predicted Prices', linewidth=2, linestyle='--', marker='x')
plt.title(f"True vs Predicted Prices (Sample {sample_index})")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig(folder_path + f'true_vs_pred_sample_{sample_index}.png')
plt.close()

# 2. Scatter plot of predicted vs true prices
plt.figure(figsize=(8, 8))
plt.scatter(trues.flatten(), preds.flatten(), alpha=0.5, edgecolor='k')
plt.title("Scatter Plot: True vs Predicted Prices")
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.grid()
plt.savefig(folder_path + 'scatter_true_vs_pred.png')
plt.close()

# 3. Error over time (difference between predictions and true values)
errors = preds - trues
plt.figure(figsize=(12, 6))
plt.plot(errors[sample_index], label='Prediction Error', color='red', linewidth=2)
plt.title(f"Prediction Error Over Time (Sample {sample_index})")
plt.xlabel("Time Steps")
plt.ylabel("Error")
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.grid()
plt.savefig(folder_path + f'error_over_time_sample_{sample_index}.png')
plt.close()

# 4. Distribution of prediction errors
plt.figure(figsize=(8, 6))
plt.hist(errors.flatten(), bins=30, color='purple', alpha=0.7, edgecolor='k')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid()
plt.savefig(folder_path + 'error_distribution.png')
plt.close()

# 5. Aggregate performance: mean predicted vs true prices over all samples
mean_true = trues.mean(axis=0)
mean_pred = preds.mean(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(mean_true, label='Mean True Prices', linewidth=2, marker='o')
plt.plot(mean_pred, label='Mean Predicted Prices', linewidth=2, linestyle='--', marker='x')
plt.title("Mean True vs Predicted Prices Across All Samples")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig(folder_path + 'mean_true_vs_pred.png')
plt.close()

print(f"Plots saved to: {folder_path}")

