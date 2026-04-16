# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 07:42:31 2026

@author: aakas
"""


# India Subdivision Rainfall Analysis (1901–2015)
# Dataset: sub-division_rainfall_act_dep_1901-2015.csv



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load & Prepare Data 
df = pd.read_csv(r'D:\python_s4\sub-division_rainfall_act_dep_1901-2015.csv')

# Filter only actual yearly records (not summary rows)
actual = df[df['Parameter'] == 'Actual'].copy()
actual = actual[actual['YEAR'].str.match(r'^\d{4}$', na=False)].copy()
actual['YEAR'] = actual['YEAR'].astype(int)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total records (Actual): {len(actual)}")
print(f"Year range: {actual['YEAR'].min()} – {actual['YEAR'].max()}")
print(f"Subdivisions: {actual['SUBDIVISION'].nunique()}")
print("\nBasic Statistics:")
print(actual[['ANNUAL', 'JJAS', 'JF', 'MAM', 'OND']].describe().round(2))

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']


# ANALYSIS 1: National Annual Rainfall Trend (1901–2015)
# Objective: Detect long-term change in India's average rainfall

print("\n" + "=" * 60)
print("ANALYSIS 1: National Annual Rainfall Trend")
print("=" * 60)

national = actual.groupby('YEAR')['ANNUAL'].mean().reset_index()
z = np.polyfit(national['YEAR'], national['ANNUAL'], 1)
p = np.poly1d(z)
trend_direction = "declining" if z[0] < 0 else "increasing"
print(f"Trend slope: {z[0]:.3f} mm/year ({trend_direction})")
print(f"Overall mean annual rainfall: {national['ANNUAL'].mean():.1f} mm")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(national['YEAR'], national['ANNUAL'], color=colors[0], linewidth=1.2, alpha=0.7, label='Annual Rainfall')
ax.fill_between(national['YEAR'], national['ANNUAL'], alpha=0.15, color=colors[0])
ax.plot(national['YEAR'], p(national['YEAR']), '--', color=colors[2], linewidth=2,
        label=f'Trend (slope = {z[0]:.2f} mm/yr)')
ax.axhline(national['ANNUAL'].mean(), color='gray', linestyle=':', linewidth=1.5,
           label=f'Mean = {national["ANNUAL"].mean():.1f} mm')
ax.set_title('Analysis 1: National Average Annual Rainfall Trend (1901–2015)', fontsize=13, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Average Annual Rainfall (mm)')
ax.legend()
plt.tight_layout()
plt.savefig('analysis1_trend.png', dpi=150)
plt.show()


# ANALYSIS 2: Seasonal Rainfall Distribution (Box Plot)
# Objective: Compare rainfall across four meteorological seasons

print("\n" + "=" * 60)
print("ANALYSIS 2: Seasonal Rainfall Distribution")
print("=" * 60)

seasons = actual[['JF', 'MAM', 'JJAS', 'OND']].dropna()
season_means = seasons.mean().round(1)
print("Mean rainfall per season (mm):")
for s, v in season_means.items():
    print(f"  {s}: {v}")

season_labels = {
    'JF':   'Jan–Feb\n(Winter)',
    'MAM':  'Mar–May\n(Pre-Monsoon)',
    'JJAS': 'Jun–Sep\n(Monsoon)',
    'OND':  'Oct–Dec\n(Post-Monsoon)'
}
melted = seasons.melt(var_name='Season', value_name='Rainfall')
melted['Season'] = melted['Season'].map(season_labels)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=melted, x='Season', y='Rainfall', palette='Set2', ax=ax, width=0.5)
ax.set_title('Analysis 2: Seasonal Rainfall Distribution Across All Subdivisions', fontsize=13, fontweight='bold')
ax.set_xlabel('Season')
ax.set_ylabel('Rainfall (mm)')
plt.tight_layout()
plt.savefig('analysis2_seasonal.png', dpi=150)
plt.show()


# ANALYSIS 3: Top 10 Wettest vs Driest Subdivisions
# Objective: Identify regions with extreme rainfall patterns

print("\n" + "=" * 60)
print("ANALYSIS 3: Wettest vs Driest Subdivisions")
print("=" * 60)

sub_mean = actual.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False)
print("\nTop 5 Wettest:")
print(sub_mean.head(5).round(1).to_string())
print("\nTop 5 Driest:")
print(sub_mean.tail(5).round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sub_mean.head(10).plot(kind='barh', ax=axes[0], color=colors[0], edgecolor='white')
axes[0].set_title('Top 10 Wettest Subdivisions', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Mean Annual Rainfall (mm)')
axes[0].invert_yaxis()

sub_mean.tail(10).plot(kind='barh', ax=axes[1], color=colors[2], edgecolor='white')
axes[1].set_title('Top 10 Driest Subdivisions', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Mean Annual Rainfall (mm)')
axes[1].invert_yaxis()

fig.suptitle('Analysis 3: Wettest vs Driest Subdivisions (1901–2015 Mean)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis3_wet_dry.png', dpi=150)
plt.show()


# ANALYSIS 4: Monthly Rainfall Correlation Heatmap
# Objective: Understand which months drive annual totals

print("\n" + "=" * 60)
print("ANALYSIS 4: Monthly Rainfall Correlation Heatmap")
print("=" * 60)

monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL']
corr = actual[monthly_cols].dropna().corr()

top_corr = corr['ANNUAL'].drop('ANNUAL').sort_values(ascending=False)
print("Correlation of each month with ANNUAL rainfall:")
print(top_corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, annot_kws={'size': 8}, vmin=-1, vmax=1)
ax.set_title('Analysis 4: Monthly Rainfall Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis4_heatmap.png', dpi=150)
plt.show()




# ANALYSIS 5 (ML): Linear Regression — Predict Annual from JJAS
# Objective: Use supervised ML to predict annual rainfall from monsoon season


print("\n" + "=" * 60)
print("ANALYSIS 6 (ML): Linear Regression — JJAS → Annual Rainfall")
print("=" * 60)

ml_data = actual[['JJAS', 'ANNUAL']].dropna()

scaler = MinMaxScaler()
ml_scaled = scaler.fit_transform(ml_data)
X = ml_scaled[:, 0].reshape(-1, 1)   
y = ml_scaled[:, 1]                   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"Model Coefficient : {model.coef_[0]:.4f}")
print(f"Model Intercept   : {model.intercept_:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score          : {r2:.4f}  ({r2*100:.1f}% variance explained)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter + regression line
axes[0].scatter(X_test, y_test, alpha=0.4, color=colors[0], label='Actual')
x_line = np.sort(X_test, axis=0)
axes[0].plot(x_line, model.predict(x_line), color=colors[2], linewidth=2.5,
             label=f'Regression Line\nR² = {r2:.3f}')
axes[0].set_title('JJAS Monsoon → Annual Rainfall Prediction', fontsize=12, fontweight='bold')
axes[0].set_xlabel('JJAS Rainfall (Normalized)')
axes[0].set_ylabel('Annual Rainfall (Normalized)')
axes[0].legend()

# Residual plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, color=colors[3])
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')

fig.suptitle(f'Analysis 6 (ML) | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}',
             fontsize=11, color='gray')
plt.tight_layout()
plt.savefig('analysis6_ml_regression.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("ALL ANALYSES COMPLETE")
print("Saved: analysis1_trend.png  through  analysis6_ml_regression.png")
print("=" * 60)