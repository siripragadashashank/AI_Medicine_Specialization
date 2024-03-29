import os
import lifelines
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:

	def __init__(self, threshold):
		#super(DataLoader, self).__init__()
		self.threshold = threshold

	def cindex(self, y_true, scores):
		return lifelines.utils.concordance_index(y_true, scores)

	def nhanesi(self, display=False):
		__location__ = os.path.realpath(
					os.path.join(os.getcwd(), 
					os.path.dirname(__file__))
			)
		X = pd.read_csv(os.path.join(__location__, 'NHANESI_subset_X.csv'))
		y = pd.read_csv(os.path.join(__location__, 'NHANESI_subset_y.csv'))["y"]
		if display:
			X_display = X.copy()
			X_display['Sex'] = ['Male' if v==1 else 'Female' for v in X['Sex']]
			return X_display, np.array(y)
		return X, np.array(y)

	def prob_drop(self, age):
		return 1 - (np.exp(0.25 * age - 5) / (1 + np.exp(0.25 * age - 5)))

	def load_data(self):
		X, y = self.nhanesi()
		df = X.drop([X.columns[0]], axis=1)
		df.loc[:, 'time'] = y
		df.loc[:, 'death'] = np.ones(len(X))
		df.loc[df.time < 0, 'death'] = 0
		df.loc[:, 'time'] = np.abs(df.time)
		df = df.dropna(axis='rows')
		mask = (df.time > self.threshold) | (df.death == 1)
		df = df[mask]
		X = df.drop(['time', 'death'], axis='columns')
		y = df.time < self.threshold


		X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
		feature_y = 'Systolic BP'
		frac = 0.7

		drop_rows = X_dev.sample(
								frac=frac, 
								replace=False,
								weights=[self.prob_drop(X_dev.loc[i, 'Age']) for i in X_dev.index], 
								random_state=10
								)
		drop_rows.loc[:, feature_y] = None
		drop_y = y_dev[drop_rows.index]
		X_dev.loc[drop_rows.index, feature_y] = None

		return X_dev, X_test, y_dev, y_test
