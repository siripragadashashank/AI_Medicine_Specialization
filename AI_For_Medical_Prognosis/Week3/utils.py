from lifelines.datasets import load_lymphoma

class DataLoader:

	def load_data(self):
		df = load_lymphoma()
		df[:, 'Event'] = df.Censor
		df.drop(['Censor'], axis=1)
		return df
