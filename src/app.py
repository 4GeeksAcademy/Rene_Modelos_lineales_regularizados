from utils import db_connect
engine = db_connect()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error


url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv"
df = pd.read_csv(url)

df_model = df.select_dtypes(include=['float64', 'int64']).copy()

df_model.drop(columns=['fips', 'CNTY_FIPS'], errors='ignore', inplace=True)

df_model.dropna(inplace=True)

target = 'anycondition_prevalence'
if target not in df_model.columns:
    df_model[target] = df[target]

features = [
    'Obesity_prevalence',
    'diabetes_prevalence',
    'COPD_prevalence',
    'Heart disease_prevalence',
    'CKD_prevalence',
    "Percent of adults with a bachelor's degree or higher 2014-18",
    'MEDHHINC_2018'
]

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#***
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print(f"Regresión Lineal:\nR² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")

alphas = list(range(0, 21))
r2_scores_lasso = []

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    r2_scores_lasso.append(r2_score(y_test, y_pred))

plt.plot(alphas, r2_scores_lasso, marker='o')
plt.xlabel("Alpha")
plt.ylabel("R²")
plt.title("Evolución de R² en Lasso")
plt.grid(True)
plt.show()

#***
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
print(f"\nMejor alpha (LassoCV): {lasso_cv.alpha_}")
print(f"R² en test set: {lasso_cv.score(X_test_scaled, y_test)}")

lasso_coef = pd.Series(lasso_cv.coef_, index=features)
print("\nCoeficientes aprendidos por LassoCV:")
print(lasso_coef[lasso_coef != 0])

#***
ridge_model = Ridge(alpha=2)
ridge_model.fit(X_train_scaled, y_train)
ridge_coef = pd.Series(ridge_model.coef_, index=features)

print("\nCoeficientes del modelo Ridge (alpha=2):")
print(ridge_coef[ridge_coef != 0])