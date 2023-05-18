import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to calculate confidence intervals
def err_ranges(x, y, func, params, conf=0.95):
    popt, pcov = curve_fit(func, x, y, p0=params)
    alpha = 1 - conf
    n = len(y)
    p = len(params)
    tval = np.abs(np.random.standard_t(df=n - p, size=int(1e6)))
    yhat = func(x, *popt)
    residuals = y - yhat
    mse = np.sum(residuals ** 2) / (n - p)
    rse = np.sqrt(mse)
    upper = yhat + rse * tval[int((1 - alpha / 2) * 1e6)]
    lower = yhat - rse * tval[int((1 - alpha / 2) * 1e6)]
    return lower, upper

# Read the data from a CSV file
df = pd.read_csv('gdp_percapita.csv', skiprows=4)
# Replace NaN values with 0
df.fillna(0, inplace=True)
# Select relevant columns and drop missing values
X = df.select_dtypes(include='number')
X = X.drop('Unnamed: 67', axis=1)

# Standardize the data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_norm)

# Add the cluster membership as a new column to the dataframe
df['Cluster'] = kmeans.labels_


# Line plot of two columns of the GDP per capita dataset
plt.plot(X['1990'], label='1990')
plt.plot(X['2010'], label='2010')
plt.xlabel('Country Index')
plt.ylabel('GDP per capita (current US$)')
plt.title('Change in GDP per Capita from 1990 to 2010')
# Show the legend
plt.legend()
plt.show()

# Plot the data for each cluster separately
for i in range(4):
    plt.scatter(X_norm[kmeans.labels_ == i, 0], X_norm[kmeans.labels_ == i, 1], label='Cluster ' + str(i+1))

# Plot the cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, linewidth=2, c='black', label='Centroids')

# Add axis labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters of GDP per Capita over Time')

# Show the legend
plt.legend()
plt.show()

# Fit an exponential growth model to the data
def exp_growth(x, a, b, c):
    return a * np.exp(b * (x - 1960)) + c

# Perform curve fitting
popt, pcov = curve_fit(exp_growth, X.columns.astype(int), X.mean())

# Calculate confidence intervals for the fit
perr = np.sqrt(np.diag(pcov))
err_lower, err_upper = err_ranges(X.columns.astype(int), X.mean(), exp_growth, popt)

# Plot the data and the fit with confidence intervals
plt.plot(X.columns.astype(int), X.mean(), 'bo', label='Data')
plt.plot(X.columns.astype(int), exp_growth(X.columns.astype(int), *popt), 'r-', label='Fit')
plt.fill_between(X.columns.astype(int), exp_growth(X.columns.astype(int), *(popt - perr)), exp_growth(X.columns.astype(int), *(popt + perr)), color='gray', alpha=0.2, label='Confidence Interval')
plt.xlabel('Year')
plt.ylabel('GDP per capita (current US$)')
plt.title('Exponential Growth of Global GDP per Capita over Time')

# Show the legend
plt.legend()
plt.show()

# Compare the trends in different clusters or countries
cluster_data = df[df['Cluster'] == 0]
grouped_data = cluster_data.iloc[:, 1:-1].groupby(df['Country Name']).mean()
top_countries = grouped_data.mean(axis=1).sort_values(ascending=False).head()
grouped_data.loc[top_countries.index].transpose().plot()
plt.xlabel('Year')
plt.ylabel('GDP per capita (current US$)')
plt.title('Exponential Growth of GDP per Capita over Time by Cluster')
plt.show()
