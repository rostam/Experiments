# import pandas as pd
#
# listings = pd.read_csv("listings.csv", delimiter=",")
# listings.info()


from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

# generate 2d classification dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=3, center_box=[3, 7])
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0: 'green', 1: 'red', 2: 'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key])
ax.set_ylim(bottom=0, top=10)
ax.set_xlim(left=0, right=10)
pyplot.xlabel("First interview")
pyplot.ylabel("Second interview")

pyplot.show()

