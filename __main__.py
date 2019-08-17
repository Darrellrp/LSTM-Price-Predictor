import random
import quandl
import csv
import os.path
from time import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense, Input, LSTM, TimeDistributed
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange

# Data Source
AV_API_KEY = 'OHL1UQBAF94QK9FH'
DIR_RAW_DATA = 'data/alpha-vantage/raw'
DIR_RAW_CLEAN = 'data/alpha-vantage/clean'
FILE_PATH_RAW_DATA = DIR_RAW_DATA + '/{}_daily_full.csv'
FILE_PATH_CLEAN_DATA = DIR_RAW_CLEAN + '/{}_daily_full_{}.csv'
FETCH_COLLECTION = False

# Pre processing
TRAIN_TEST_SPLIT = 0.8

# Autocoder TODO: Adjust size to input & output dimensions
HIDDEN_SIZE = 10
CODE_SIZE = 6

# Training
EPOCHS = 20
BATCH_SIZE = 32
VERBOSE = 1

# Optimization Neural Network
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = None
DECAY = 0.0
AMSGRAD = False

# Compilation Neural Network
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['accuracy']


# Tensorflow
FILE_PATH_TF_LOGS = 'logs/{}'

# Post Training
EVALUATE = False
PREDICT = True
NUMBER_SHOWN_OF_PREDICTIONS = 4


# Read market items from text file
market_items = open('market-items.txt').readline().split(', ')

# Remove newline character from market items
market_items = [m.replace('\n', '') for m in market_items]

data = None
targets = None

# Forex Data source TODO: Integrate Forex
# fx = ForeignExchange(key=AV_API_KEY, output_format='pandas')
# mmi = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')

# if clean data or targets doesn't exist, clean and store again
if (not os.path.exists('{}/{}'.format(DIR_RAW_CLEAN, 'data_collection.csv'))
		or not os.path.exists('{}/{}'.format(DIR_RAW_CLEAN, 'targets_collection.csv'))) or FETCH_COLLECTION:

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

			rename_columns = {'1. open': 'open', '2. high': 'high', '3. low': ' low',
							  '4. close': 'close', '5. volume': 'volume'}

			# Rename columns
			market_item.rename(columns=rename_columns, inplace=True)
			market_item.index.name = 'timestamp'

			# Store Market item data
			market_item.to_csv(file_path_raw, header=True)

		else:
			print('Loading Daily Time Series -csv (Alpha Vantage) - {}...'.format(market_item_name))
			market_item = pd.read_csv(file_path_raw, parse_dates=['timestamp'])
			market_item.drop('timestamp', axis=1, inplace=True)

		market_item['close'].plot()
		plt.title(market_item_name)
		# plt.show()

	# ***************************************     Pre-processing     ***************************************
		file_path_raw = FILE_PATH_RAW_DATA.format(market_item_name)
		file_path_clean_data = FILE_PATH_CLEAN_DATA.format(market_item_name, 'data')
		file_path_clean_targets = FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets')

		# if clean data or targets doesn't exist, clean and store again
		if not os.path.exists(file_path_clean_data) or not os.path.exists(file_path_clean_targets):
			print('Pre-processing {}...'.format(market_item_name))
			print('')

			# Remove last row, Drop timestamp column
			market_item_data = market_item.drop(market_item.tail(1).index)

			# Select close columns, Drop First row (Next days close price)
			market_item_targets = market_item['close'].drop(market_item.head(1).index)

			# Convert to float32
			market_item_data = market_item_data.astype('float32')
			market_item_targets = market_item_targets.astype('float32')

			scaler = MinMaxScaler(feature_range=(0, 1))
			scaler = scaler.fit(market_item_data.values)

			# Normalize financial data
			market_item_data = scaler.transform(market_item_data.values)

			# Store cleaned financial data
			np.savetxt(FILE_PATH_CLEAN_DATA.format(market_item_name, 'data'), market_item_data, delimiter=',')
			np.savetxt(FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets'), market_item_targets, delimiter=',')

		else:
			# Read cleaned financial data
			market_item_data = pd.read_csv(FILE_PATH_CLEAN_DATA.format(market_item_name, 'data'), header=None).values
			market_item_targets = pd.read_csv(FILE_PATH_CLEAN_DATA.format(market_item_name, 'targets'), header=None).values

		if data is None:
			data = market_item_data
			targets = market_item_targets
		else:
			# TODO: Concatenate columns grouped by timestamp
			data = np.concatenate((data, market_item_data))
			targets = np.concatenate((targets, market_item_targets))

	np.savetxt('{}/{}'.format(DIR_RAW_CLEAN, 'data_collection.csv'), data, delimiter=',')
	np.savetxt('{}/{}'.format(DIR_RAW_CLEAN, 'targets_collection.csv'), targets, delimiter=',')

else:
	print('Loading Collection -csv (Alpha Vantage)...')
	data = pd.read_csv('{}/{}'.format(DIR_RAW_CLEAN, 'data_collection.csv'), header=None).values
	targets = pd.read_csv('{}/{}'.format(DIR_RAW_CLEAN, 'targets_collection.csv'), header=None).values

print('')

# Train Test Splits
print('Generating Train Test split...')
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

print('Number of rows train data: {}'.format(len(x_train)))
print('Number of rows test data: {} '.format(len(x_test)))
print('')

input_size = x_train.shape[1]

# ***************************************     Neural Network Architecture     ***************************************
# Input layer (size=748)
input_layer = Input(shape=(input_size,))

# Autoencoder 1.
hidden_1 = Dense(HIDDEN_SIZE, activation='relu')(input_layer)
code_1 = Dense(CODE_SIZE, activation='relu')(hidden_1)

# Autoencoder 2.
hidden_2 = Dense(HIDDEN_SIZE, activation='relu')(code_1)
code_2 = Dense(CODE_SIZE, activation='relu')(hidden_2)

# Autoencoder 3.
hidden_3 = Dense(HIDDEN_SIZE, activation='relu')(code_2)
code_3 = Dense(CODE_SIZE, activation='relu')(hidden_3)

# Autoencoder 2.
hidden_4 = Dense(HIDDEN_SIZE, activation='relu')(code_3)
code_4 = Dense(CODE_SIZE, activation='relu')(hidden_4)
hidden_5 = Dense(HIDDEN_SIZE, activation='relu')(code_4)


# Output layer (Reconstructed image, siz=748)
output_autoencoder = TimeDistributed((Dense(input_size, activation='sigmoid')(hidden_5)))

# TODO: Append LSTM
layer1 = LSTM(HIDDEN_SIZE, return_sequences=True)(output_autoencoder)
layer2 = LSTM(HIDDEN_SIZE, return_sequences=True)(layer1)
layer3 = LSTM(HIDDEN_SIZE, return_sequences=True)(layer2)
layer4 = LSTM(HIDDEN_SIZE, return_sequences=True)(layer3)
layer5 = LSTM(HIDDEN_SIZE, return_sequences=True)(layer4)

output_lstm = Dense(1)(layer5)

stacked_autoencoder = Model(input_layer, output_lstm)

adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON, decay=DECAY, amsgrad=AMSGRAD)
# optimizers: 'adadelta' | loss: 'mape', 'binary_crossentropy'
stacked_autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

tensorboard = TensorBoard(log_dir=FILE_PATH_TF_LOGS.format(time()))

stacked_autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
						verbose=VERBOSE, callbacks=[tensorboard])

# ***************************************     Post Training     ***************************************
if EVALUATE:
	score, accuracy = stacked_autoencoder.evaluate(x_test, x_test)
	print('Test score:', score)
	print('Test accuracy:', accuracy)

if PREDICT:
	reconstructed = stacked_autoencoder.predict(x_test)

	# Show a number of predictions
	for i in range(NUMBER_SHOWN_OF_PREDICTIONS):
		# Generate random index
		rand_i = random.randrange(0, x_test.shape[0])

		print('')

		# Print actual label
		print('Actual data: {}'.format(x_test[rand_i]))
		print('Reconstructed data: {}'.format(reconstructed[rand_i]))

