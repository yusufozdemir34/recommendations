import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
import panda as pd

import matplotlib.pyplot as plt

from service.ModelService import load_model_as_np




utility, test, user, item, user_user_pearson = load_model_as_np()

flights_raw = pd.read_csv('flights.csv')
flights_raw["month"] = pd.Categorical(flights_raw["month"], flights_raw.month.unique())
flights_raw.head()
fig = plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(943, 943))
heatplot = ax.imshow(user_user_pearson, cmap='BuPu')
ax.set_xticklabels(user_user_pearson.columns)
ax.set_yticklabels(user_user_pearson.index)

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_title("Heatmap of Flight Density from 1949 to 1961")
ax.set_xlabel('Year')
ax.set_ylabel('Month')
