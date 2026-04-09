import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# data loading and preprocessing
df = pd.read_csv('techSalaries2017.csv')
print(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# target variable
target = 'totalyearlycompensation'

# quantitative predictors
quant_cols = [
    'yearsofexperience', 'yearsatcompany',
    'Masters_Degree', 'Bachelors_Degree', 'Doctorate_Degree', 'Highschool', 'Some_College',
    'Race_Asian', 'Race_White', 'Race_Two_Or_More', 'Race_Black', 'Race_Hispanic',
    'Age', 'Height', 'Zodiac', 'SAT', 'GPA'
]

# replace na strings with nan
df.replace('NA', np.nan, inplace=True)

# convert columns to numeric
for col in quant_cols + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# handle missing data for the education and race dummy variables
# if education or race is missing the dummy variables are not meaningful
df_clean = df.dropna(subset=['Education', 'Race']).copy()
print(f"After requiring Education and Race info: {df_clean.shape[0]} rows")

# drop rows with missing values in predictors or target
df_clean = df_clean.dropna(subset=quant_cols + [target]).copy()
print(f"After dropping remaining missing values in predictors and target: {df_clean.shape[0]} rows")

# drop one dummy from each group to avoid perfect multicollinearity
# some college is the education baseline and race hispanic is the race baseline
feature_cols = [c for c in quant_cols if c not in ['Some_College', 'Race_Hispanic']]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

X = df_clean[feature_cols].values
y = df_clean[target].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# standardize features for ridge and lasso
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# question 1 ols
print("\nQUESTION 1 OLS REGRESSION")

# univariate models using one predictor at a time
print("\nUNIVARIATE OLS MODELS")
univariate_results = {}
for i, col in enumerate(feature_cols):
    lr = LinearRegression()
    lr.fit(X_train[:, i:i+1], y_train)
    y_pred = lr.predict(X_test[:, i:i+1])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    univariate_results[col] = {
        'r2': r2,
        'rmse': rmse,
        'coef': lr.coef_[0],
        'intercept': lr.intercept_
    }
    print(f"  {col:30s}  R²={r2:.4f}  RMSE={rmse:,.0f}")

best_predictor = max(univariate_results, key=lambda k: univariate_results[k]['r2'])
best_r2 = univariate_results[best_predictor]['r2']
best_rmse = univariate_results[best_predictor]['rmse']
print(f"\nBest univariate predictor: {best_predictor} R²={best_r2:.4f} RMSE={best_rmse:,.0f}")

# full multiple linear regression
ols_full = LinearRegression()
ols_full.fit(X_train, y_train)
y_pred_ols = ols_full.predict(X_test)
ols_r2 = r2_score(y_test, y_pred_ols)
ols_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ols))
print(f"\nFull OLS: R²={ols_r2:.4f} RMSE={ols_rmse:,.0f}")
print("OLS Coefficients")
for col, coef in zip(feature_cols, ols_full.coef_):
    print(f"  {col:30s}: {coef:>12.2f}")
print(f"  {'Intercept':30s}: {ols_full.intercept_:>12.2f}")

# question 2 ridge regression
print("\nQUESTION 2 RIDGE REGRESSION")

# use cross validation to find the best lambda for the full ridge model
alphas_ridge = np.logspace(-2, 6, 100)
ridge_cv = RidgeCV(alphas=alphas_ridge, scoring='r2', cv=5)
ridge_cv.fit(X_train_scaled, y_train)
best_alpha_ridge = ridge_cv.alpha_
print(f"Optimal Ridge lambda: {best_alpha_ridge:.4f}")

# fit the final full ridge model
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"Ridge: R²={ridge_r2:.4f} RMSE={ridge_rmse:,.0f}")
print("Ridge Coefficients standardized")
for col, coef in zip(feature_cols, ridge_model.coef_):
    print(f"  {col:30s}: {coef:>12.2f}")

# univariate ridge models with lambda tuned separately for each predictor
print("\nUNIVARIATE RIDGE MODELS")
univariate_ridge_results = {}
for i, col in enumerate(feature_cols):
    X_train_one = X_train[:, i:i+1]
    X_test_one = X_test[:, i:i+1]

    scaler_one = StandardScaler()
    X_train_one_scaled = scaler_one.fit_transform(X_train_one)
    X_test_one_scaled = scaler_one.transform(X_test_one)

    ridge_uni_cv = RidgeCV(alphas=alphas_ridge, scoring='r2', cv=5)
    ridge_uni_cv.fit(X_train_one_scaled, y_train)
    best_alpha_uni = ridge_uni_cv.alpha_

    ridge_uni = Ridge(alpha=best_alpha_uni)
    ridge_uni.fit(X_train_one_scaled, y_train)
    y_pred_uni = ridge_uni.predict(X_test_one_scaled)
    r2 = r2_score(y_test, y_pred_uni)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_uni))

    univariate_ridge_results[col] = {
        'r2': r2,
        'rmse': rmse,
        'alpha': best_alpha_uni
    }
    print(f"  {col:30s}  R²={r2:.4f}  RMSE={rmse:,.0f}  lambda={best_alpha_uni:.4f}")

best_ridge_uni = max(univariate_ridge_results, key=lambda k: univariate_ridge_results[k]['r2'])
print(
    f"Best univariate Ridge predictor: {best_ridge_uni} "
    f"R²={univariate_ridge_results[best_ridge_uni]['r2']:.4f} "
    f"lambda={univariate_ridge_results[best_ridge_uni]['alpha']:.4f}"
)

# question 3 lasso regression
print("\nQUESTION 3 LASSO REGRESSION")

# use cross validation to find the best lambda for the full lasso model
alphas_lasso = np.logspace(-1, 5, 100)
lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, random_state=42, max_iter=20000)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha_lasso = lasso_cv.alpha_
print(f"Optimal Lasso lambda: {best_alpha_lasso:.4f}")

# fit the final full lasso model
lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=20000)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"Lasso: R²={lasso_r2:.4f} RMSE={lasso_rmse:,.0f}")

zero_coefs = np.sum(lasso_model.coef_ == 0)
print(f"Coefficients shrunk to zero: {zero_coefs} out of {len(feature_cols)}")
print("Lasso Coefficients standardized")
for col, coef in zip(feature_cols, lasso_model.coef_):
    marker = " *ZERO*" if coef == 0 else ""
    print(f"  {col:30s}: {coef:>12.2f}{marker}")

# univariate lasso models with lambda tuned separately for each predictor
print("\nUNIVARIATE LASSO MODELS")
univariate_lasso_results = {}
for i, col in enumerate(feature_cols):
    X_train_one = X_train[:, i:i+1]
    X_test_one = X_test[:, i:i+1]

    scaler_one = StandardScaler()
    X_train_one_scaled = scaler_one.fit_transform(X_train_one)
    X_test_one_scaled = scaler_one.transform(X_test_one)

    lasso_uni_cv = LassoCV(alphas=alphas_lasso, cv=5, random_state=42, max_iter=20000)
    lasso_uni_cv.fit(X_train_one_scaled, y_train)
    best_alpha_uni = lasso_uni_cv.alpha_

    lasso_uni = Lasso(alpha=best_alpha_uni, max_iter=20000)
    lasso_uni.fit(X_train_one_scaled, y_train)
    y_pred_uni = lasso_uni.predict(X_test_one_scaled)
    r2 = r2_score(y_test, y_pred_uni)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_uni))

    univariate_lasso_results[col] = {
        'r2': r2,
        'rmse': rmse,
        'alpha': best_alpha_uni
    }
    print(f"  {col:30s}  R²={r2:.4f}  RMSE={rmse:,.0f}  lambda={best_alpha_uni:.4f}")

best_lasso_uni = max(univariate_lasso_results, key=lambda k: univariate_lasso_results[k]['r2'])
print(
    f"Best univariate Lasso predictor: {best_lasso_uni} "
    f"R²={univariate_lasso_results[best_lasso_uni]['r2']:.4f} "
    f"lambda={univariate_lasso_results[best_lasso_uni]['alpha']:.4f}"
)

# question 4 distribution analysis
print("\nQUESTION 4 DISTRIBUTION ANALYSIS")

# use the full available data for each variable instead of the filtered modeling subset
for var_name in ['totalyearlycompensation', 'Height', 'Age']:
    data = df[var_name].dropna().values
    stat_sw, p_sw = stats.shapiro(data[:5000])
    stat_ks, p_ks = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=0)))
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"\n{var_name}")
    print(f"  N={len(data)} Mean={np.mean(data):.2f} Std={np.std(data, ddof=0):.2f}")
    print(f"  Skewness={skewness:.4f} Kurtosis={kurtosis:.4f}")
    print(f"  Shapiro Wilk: stat={stat_sw:.4f} p={p_sw:.2e}")
    print(f"  K S test: stat={stat_ks:.4f} p={p_ks:.2e}")

# generating figures
print("\nGENERATING FIGURES")

# figure 1 univariate r2 comparison
fig, ax = plt.subplots(figsize=(10, 5))
cols_sorted = sorted(univariate_results.keys(), key=lambda k: univariate_results[k]['r2'], reverse=True)
r2_vals = [univariate_results[c]['r2'] for c in cols_sorted]
ax.barh(range(len(cols_sorted)), r2_vals, color='steelblue')
ax.set_yticks(range(len(cols_sorted)))
ax.set_yticklabels(cols_sorted, fontsize=8)
ax.set_xlabel('R² Score')
ax.set_title('Univariate OLS R² by Predictor')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('fig1_univariate_r2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1_univariate_r2.png')

# figure 2 model comparison
fig, ax = plt.subplots(figsize=(7, 4))
models = ['Best Univariate\nOLS', 'Full OLS', 'Ridge', 'Lasso']
r2_scores = [best_r2, ols_r2, ridge_r2, lasso_r2]

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width / 2, r2_scores, width, label='R²', color='steelblue')
ax.set_ylabel('R² Score')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_title('Model Comparison R² Scores')
ax.legend()
for bar, val in zip(bars1, r2_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=9
    )
plt.tight_layout()
plt.savefig('fig2_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_model_comparison.png')

# figure 3 coefficient comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(feature_cols))
width = 0.25

ols_scaled = LinearRegression()
ols_scaled.fit(X_train_scaled, y_train)

ax.barh(x - width, ols_scaled.coef_, width, label='OLS', color='steelblue', alpha=0.8)
ax.barh(x, ridge_model.coef_, width, label='Ridge', color='coral', alpha=0.8)
ax.barh(x + width, lasso_model.coef_, width, label='Lasso', color='seagreen', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(feature_cols, fontsize=8)
ax.set_xlabel('Standardized Coefficient')
ax.set_title('Coefficient Comparison OLS Ridge Lasso')
ax.legend()
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('fig3_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_coefficients.png')

# figure 4 distribution histograms with normal curves
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
dist_vars = [
    ('totalyearlycompensation', 'Total Yearly Compensation $'),
    ('Height', 'Height inches'),
    ('Age', 'Age years')
]

for ax, (var, label) in zip(axes, dist_vars):
    data = df[var].dropna().values
    ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    mu, sigma = np.mean(data), np.std(data, ddof=0)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {var}')
    ax.legend()

plt.tight_layout()
plt.savefig('fig4_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_distributions.png')

# figure 5 actual vs predicted for full ols
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_ols, alpha=0.3, s=5, color='steelblue')
lims = [0, max(y_test.max(), y_pred_ols.max())]
ax.plot(lims, lims, 'r--', linewidth=1)
ax.set_xlabel('Actual Compensation $')
ax.set_ylabel('Predicted Compensation $')
ax.set_title(f'OLS Actual vs Predicted R²={ols_r2:.4f}')
plt.tight_layout()
plt.savefig('fig5_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5_actual_vs_predicted.png')

# figure 6 extra credit scatter of compensation and experience
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    df_clean['yearsofexperience'],
    df_clean['totalyearlycompensation'],
    alpha=0.1,
    s=3,
    color='steelblue'
)
z = np.polyfit(df_clean['yearsofexperience'].values, df_clean['totalyearlycompensation'].values, 1)
p = np.poly1d(z)
x_line = np.linspace(0, df_clean['yearsofexperience'].max(), 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=2)
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Total Yearly Compensation $')
ax.set_title('Compensation vs Experience')
plt.tight_layout()
plt.savefig('fig6_extra_credit.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig6_extra_credit.png')

# summary
print("\nSUMMARY")
print(f"{'Model':<25s} {'R²':>8s} {'RMSE':>12s}")
print('-' * 45)
print(f"{'Best Univariate OLS':<25s} {best_r2:>8.4f} {best_rmse:>12,.0f}")
print(f"{'Full OLS':<25s} {ols_r2:>8.4f} {ols_rmse:>12,.0f}")
print(f"{'Ridge lambda='+f'{best_alpha_ridge:.2f}':<25s} {ridge_r2:>8.4f} {ridge_rmse:>12,.0f}")
print(f"{'Lasso lambda='+f'{best_alpha_lasso:.2f}':<25s} {lasso_r2:>8.4f} {lasso_rmse:>12,.0f}")
print(f"\nLasso zeroed {zero_coefs} of {len(feature_cols)} coefficients")
