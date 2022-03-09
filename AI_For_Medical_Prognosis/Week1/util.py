import numpy as np
import pandas as pd
import sklearn

class DataGenerator:
	"""Data Generator Class"""
	def __init__(self, nrows):
		super(DataGenerator, self).__init__()
		self.nrows = nrows

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
	
	def f(self, x):
		p = 0.4 * (np.log(x[0]) - np.log(60)) + 0.33 * (
								np.log(x[1]) - np.log(100)) + 0.3 * (
								np.log(x[3]) - np.log(100)) - 2.0 * (
								np.log(x[0]) - np.log(60)) * (
								np.log(x[3]) - np.log(100)) + 0.05 * np.random.logistic()
		if p > 0.0:
			return 1.0
		else:
			return 0.0


	
	def generate_data(self):
		df = pd.DataFrame(
			columns = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol']
			)
		
		df.loc[:, 'Age'] = np.exp(np.log(60) + (1 / 7) * np.random.normal(size = self.nrows))

		df.loc[:, ['Systolic_BP', 'Diastolic_BP', 'Cholesterol']] = np.exp(
																	np.random.multivariate_normal(
																		mean=[np.log(100), np.log(90), np.log(100)],
																		cov=(1 / 45) * np.array([
																		[0.5, 0.2, 0.2],
																		[0.2, 0.5, 0.2],
																		[0.2, 0.2, 0.5]]),
																		size = self.nrows)
																	)

		return df

	def load_data(self):
		np.random.seed(0)
		df = self.generate_data()
		for i in range(len(df)):
			df.loc[i, 'y'] = self.f(df.loc[i, :])
			X = df.drop('y', axis=1)
			y = df.y
		return X, y

