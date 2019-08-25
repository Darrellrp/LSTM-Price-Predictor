import numpy as np
import pandas as pd
from collections import deque
from sklearn import preprocessing
import random
from sklearn.preprocessing import MinMaxScaler


# Functions
def temporalize(data, target, lookback):
	output_x = {}
	output_y = {}

	for i in range(len(data) - lookback - 1):
		# t = np.empty(shape=(_lookback, data.shape[1]))
		# t = None
		t = data.iloc[i:i+lookback, :]
		# for j in range(1, lookback+1):
			# Gather past records up to the lookback period
			# t = data.iloc[i + j + 1].to_frame() if t is None else t.append(data.iloc[i + j + 1])
		output_x[i] = t
		output_y[i] = target.iloc[i+lookback].to_frame()

	df = pd.DataFrame.from_dict(output_x, orient='index')
	tg = pd.DataFrame.from_dict(output_y, orient='index')
	plzz = df.iloc[0].to_frame()
	return df, tg


def preprocess_df(market_item, target_name, lookback):
	print('Pre-processing {}...'.format(target_name))
	print('')

	market_item = market_item.astype('float32')

	market_item['target'] = market_item['{}_close'.format(target_name)]
	# timestamps = market_item.indices
	market_item_targets = market_item['target'].values
	market_item_data = market_item.drop(['target'], axis=1)
	market_item_indices = market_item.index.values
	market_item_columns = market_item.columns.values

	scaler = MinMaxScaler(feature_range=(0, 1))
	market_item_data = scaler.fit_transform(market_item_data.values)

	norm_data = np.column_stack((market_item_data, market_item_targets))

	# Normalize & scale financial data
	# for col in market_item:
	# 	if col is not 'target':
	# 		market_item[col] = market_item[col].pct_change()
	# 		market_item.dropna(inplace=True)
	# 		print('Column: {}'.format(col))
	# 		market_item[col] = preprocessing.scale(market_item[col].values)
	#
	# 	market_item.dropna(inplace=True)

	sequential_data = []
	prev_days = deque(maxlen=lookback)
	print(norm_data[:5])

	for i in norm_data:
		prev_days.append([n for n in i[:-1]])
		if len(prev_days) == lookback:
			sequential_data.append([np.array(prev_days), i[-1]])

	random.shuffle(sequential_data)

	X = []
	y = []

	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), np.array(y)


