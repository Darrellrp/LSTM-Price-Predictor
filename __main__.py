import sys
import random
import quandl
import csv
import os.path
from time import time
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense, LSTM, TimeDistributed, Dropout, BatchNormalization, CuDNNLSTM
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from functions import preprocess_df

# Data Source
AV_API_KEY = 'OHL1UQBAF94QK9FH'
DIR_RAW_DATA = 'data/alpha-vantage/raw'
DIR_CLEAN_DATA = 'data/alpha-vantage/clean'
FILE_PATH_RAW_DATA = DIR_RAW_DATA + '/{}_daily_full.csv'
FILE_PATH_CLEAN_DATA = DIR_CLEAN_DATA + '/cleaned_{}_daily_full_{}.csv'
FETCH_COLLECTION = False
D_TYPE = {
	# 'timestamp': 'str',
	'open': 'float32',
	'high': 'float32',
	'low': 'float32',
	'close': 'float32',
	'volume': 'float32'
}

# Pre processing
TEST_SIZE_PCT = 0.05
SAMPLE_SIZE = None
LOOKBACK = 30

# LSTM
LSTM_HIDDEN_SIZE = 30
LSTM_CODE_SIZE = 100

# Training
EPOCHS = 50
BATCH_SIZE = 64
VERBOSE = 1

# Optimization Neural Network
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = None
DECAY = 1e-6
AMSGRAD = False

# Compilation Neural Network
LOSS = 'mean_squared_error'
# LOSS = 'binary_crossentropy'
METRICS = ['mse', 'mae', 'mape']
# OPTIMIZER = 'adam'
OPTIMIZER = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON, decay=DECAY,
							amsgrad=AMSGRAD)

# Tensorflow
FILE_PATH_TF_LOGS = 'logs/{}'

# Post Training
EVALUATE = False
PREDICT = True
NUMBER_SHOWN_OF_PREDICTIONS = 4
SHOW = False


# Read market items from text file
market_items = open('market-items.txt').readline().split(', ')

# Remove newline character from market items
market_items = [m.replace('\n', '') for m in market_items]

# Get target market item
target_market_item = [i.replace('*', '') for i in market_items if '*' in i]

if len(target_market_item) > 1:
	sys.exit("Error: Multiple target market items")

target_market_item = target_market_item[0]
index = market_items.index('*{}'.format(target_market_item))
market_items[index] = target_market_item


data = None
targets = None

# Forex Data source TODO: Integrate Forex
# fx = ForeignExchange(key=AV_API_KEY, output_format='pandas')
# mmi = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')

# if clean data or targets doesn't exist, clean and store again
if (not os.path.exists('{}/{}'.format(DIR_CLEAN_DATA, 'data_collection.csv'))
		or not os.path.exists('{}/{}'.format(DIR_CLEAN_DATA, 'targets_collection.csv'))) or FETCH_COLLECTION:

	# ***************************************     Retrieve financial data     ***************************************
	for market_item_name in market_items:
		file_path_raw = FILE_PATH_RAW_DATA.format(market_item_name)

		if not os.path.exists(file_path_raw):
			print('Fetching Daily Time Series -csv (Alpha Vantage) - {}...'.format(market_item_name))
			ts = TimeSeries(key=AV_API_KEY, output_format='pandas', indexing_type='date')

			# Get json object with the intraday data and another with  the call's metadata
			market_item, meta_data = ts.get_daily(market_item_name, outputsize='full')

			print(market_item_name)
			print('Record size: {}'.format(market_item.shape[0]))
			print('')

			rename_columns = {
				'1. open': 'open',
				'2. high': 'high',
				'3. low': 'low',
				'4. close': 'close',
				'5. volume': 'volume'
			}

			# Rename columns
			market_item.rename(columns=rename_columns, inplace=True)
			market_item.index.name = 'timestamp'

			market_item_desc = market_item
			market_item = market_item.sort_index()

			# Store Market item data (descending order)
			market_item_desc.to_csv(file_path_raw, header=True)

		else:
			print('Loading raw Daily Time Series -csv (Alpha Vantage) - {}...'.format(market_item_name))
			market_item = pd.read_csv(file_path_raw, index_col='timestamp', parse_dates=['timestamp'], dtype=D_TYPE)
			market_item = market_item.sort_index()

		if SHOW:
			# To bottom of file
			market_item['close'].plot()
			plt.title(market_item_name)
			plt.show()

	# ***************************************     Pre-processing     ***************************************
		file_path_raw = FILE_PATH_RAW_DATA.format(market_item_name)
		file_path_clean_data = FILE_PATH_CLEAN_DATA.format(market_item_name, 'data')
		file_path_clean_targets = FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets')

		# Rename columns
		rename_columns = {
			'open': '{}_open'.format(market_item_name),
			'high': '{}_high'.format(market_item_name),
			'low': '{}_low'.format(market_item_name),
			'close': '{}_close'.format(market_item_name),
			'volume': '{}_volume'.format(market_item_name)
		}
		market_item.rename(columns=rename_columns, inplace=True)

		if data is None:
			data = market_item

		else:
			# TODO: join left nan values to first record
			merge = data.merge(market_item, how='inner', left_index=True, right_index=True)
			data = data if merge.empty else merge

		# if market_item_name is target_market_item:
		# 	targets = market_item_targets

	# data.to_csv(f"{DIR_CLEAN_DATA}/cleaned_{target_market_item}_input_collection.csv", header=True)
#
# else:
# 	print('Loading Collection -csv (Alpha Vantage)...')
# 	data = pd.read_csv('{}/{}_input_collection.csv}'.format(DIR_CLEAN_DATA, target_market_item), header=None)
# 	targets = pd.read_csv('{}/{}_daily_full_targets.csv'.format(DIR_CLEAN_DATA, target_market_item), header=None)

print('')

# Specify number of records | None is all records
if isinstance(SAMPLE_SIZE, int):
	data = data[:SAMPLE_SIZE]

# print('Retrieved Daily Time Series -csv (Alpha Vantage) - {}'.format(market_items))
print('')

# Train Test Splits
print('Generating Train Test split...')
msk = np.random.rand(len(data)) >= TEST_SIZE_PCT

train = data[msk]
test = data[~msk]

x_train, y_train = preprocess_df(train, target_name=target_market_item, lookback=LOOKBACK)
x_test, y_test = preprocess_df(test, target_name=target_market_item, lookback=LOOKBACK)

print('Number of rows train data: {}'.format(len(x_train)))
print('Number of rows test data: {} '.format(len(x_test)))
print('')

input_dim = x_train.shape[1:]
print(x_train.shape)

# ***************************************     Neural Network Architecture     ***************************************
model = Sequential()

# LSTM
model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim, return_sequences=True))
model.add((Dropout(0.2)))
model.add(BatchNormalization())

model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim, return_sequences=True))
model.add((Dropout(0.2)))
model.add(BatchNormalization())

model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(LSTM_HIDDEN_SIZE, input_shape=input_dim))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='relu'))

# adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON, decay=DECAY, amsgrad=AMSGRAD)
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

tensorboard = TensorBoard(log_dir=FILE_PATH_TF_LOGS.format(time()))

# filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='mse', verbose=1, save_best_only=True,
#  mode='max'))

# params callbacks=[tensorboard, checkpoint], validation_data=(x_test, y_test)
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, validation_data=(x_test, y_test)
		, callbacks=[tensorboard])

# ***************************************     Post Training     ***************************************
if EVALUATE:
	score, accuracy = model.evaluate(y_test, y_test)
	print('Test score:', score)
	print('Test accuracy:', accuracy)

if PREDICT:
	actual_values = y_test
	predictions = model.predict(x_test)

	# Show a number of predictions
	for i in range(NUMBER_SHOWN_OF_PREDICTIONS):
		# Generate random index
		rand_i = random.randrange(0, x_test.shape[0])

		actual = actual_values[rand_i]
		prediction = predictions[rand_i]

		# print(actual_values)

		# Print actual label
		print('Actual closing price: {}'.format(actual))
		print('Predicted closing price: {}'.format(prediction))

	plt.plot(actual_values, color='green')
	plt.plot(predictions, color='red')
	plt.show()


# model.summary()
