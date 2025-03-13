import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras.losses import MeanSquaredError 
import seaborn as sns

# Load the data
msft = pd.read_csv('/Users/HP/Downloads/MSFT.csv')
googl = pd.read_csv('/Users/HP/Downloads/GOOGL.csv')
aapl = pd.read_csv('/Users/HP/Downloads/AAPL.csv')
nvda = pd.read_csv('/Users/HP/Downloads/NVDA.csv')
meta = pd.read_csv('/Users/HP/Downloads/META.csv')
ibm = pd.read_csv('/Users/HP/Downloads/IBM.csv')
adbe = pd.read_csv('/Users/HP/Downloads/ADBE.csv')
jnj = pd.read_csv('/Users/HP/Downloads/JNJ.csv')
vz = pd.read_csv('/Users/HP/Downloads/VZ.csv')
unh = pd.read_csv('/Users/HP/Downloads/UNH.csv')
t = pd.read_csv('/Users/HP/Downloads/T.csv')
ups = pd.read_csv('/Users/HP/Downloads/UPS.csv')


def prepare_data(df, look_back=60):
    # Use closing price
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back), 0])
        y.append(data_scaled[i + look_back, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Prepare datasets for all companies
look_back = 60
X_msft, y_msft, scaler_msft = prepare_data(msft, look_back)
X_googl, y_googl, scaler_googl = prepare_data(googl, look_back)
X_aapl, y_aapl, scaler_aapl = prepare_data(aapl, look_back)
X_nvda, y_nvda, scaler_nvda = prepare_data(nvda, look_back)
X_meta, y_meta, scaler_meta = prepare_data(meta, look_back)
X_ibm, y_ibm, scaler_ibm = prepare_data(ibm, look_back)
X_adbe, y_adbe, scaler_adbe = prepare_data(adbe, look_back)
X_jnj, y_jnj, scaler_jnj = prepare_data(jnj, look_back)
X_vz, y_vz, scaler_vz = prepare_data(vz, look_back)
X_unh, y_unh, scaler_unh = prepare_data(unh, look_back)
X_t, y_t, scaler_t = prepare_data(t, look_back)
X_ups, y_ups, scaler_ups = prepare_data(ups, look_back)

# Split data into train and test sets (80-20 split)
train_size = int(len(X_msft) * 0.8)
X_train_msft, X_test_msft = X_msft[:train_size], X_msft[train_size:]
y_train_msft, y_test_msft = y_msft[:train_size], y_msft[train_size:]

# Build LSTM model
def create_model(look_back):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

# Train and save models for all companies
# Microsoft
model_msft = create_model(look_back)
history_msft = model_msft.fit(X_msft, y_msft, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_msft.save('models/MSFT_model.h5', save_format='h5')

# Google
model_googl = create_model(look_back)
history_googl = model_googl.fit(X_googl, y_googl, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_googl.save('models/GOOGL_model.h5', save_format='h5')

# Apple
model_aapl = create_model(look_back)
history_aapl = model_aapl.fit(X_aapl, y_aapl, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_aapl.save('models/AAPL_model.h5', save_format='h5')

# NVIDIA
model_nvda = create_model(look_back)
history_nvda = model_nvda.fit(X_nvda, y_nvda, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_nvda.save('models/NVDA_model.h5', save_format='h5')

# META
model_meta = create_model(look_back)
history_meta = model_meta.fit(X_meta, y_meta, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_meta.save('models/META_model.h5', save_format='h5')

# IBM
model_ibm = create_model(look_back)
history_ibm = model_ibm.fit(X_ibm, y_ibm, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_ibm.save('models/IBM_model.h5', save_format='h5')

# Adobe
model_adbe = create_model(look_back)
history_adbe = model_adbe.fit(X_adbe, y_adbe, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_adbe.save('models/ADBE_model.h5', save_format='h5')

#Johnson&Johnson
model_jnj= create_model(look_back)
history_jnj = model_jnj.fit(X_jnj, y_jnj, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_jnj.save('models/JNJ_model.h5', save_format='h5')

# Verizon
model_vz = create_model(look_back)
history_vz = model_vz.fit(X_vz, y_vz, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_vz.save('models/VZ_model.h5', save_format='h5')

# United Health Group
model_unh = create_model(look_back)
history_unh = model_unh.fit(X_unh, y_unh, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_unh.save('models/UNH_model.h5', save_format='h5')

# AT&T
model_t = create_model(look_back)
history_t = model_t.fit(X_t, y_t, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_t.save('models/T_model.h5', save_format='h5')

# UPS
model_ups = create_model(look_back)
history_ups = model_ups.fit(X_ups, y_ups, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model_ups.save('models/UPS_model.h5', save_format='h5')

# Make predictions
predictions_msft = model_msft.predict(X_test_msft)

# Inverse transform predictions and actual values
predictions_msft = scaler_msft.inverse_transform(predictions_msft)
y_test_msft_actual = scaler_msft.inverse_transform([y_test_msft])

# Plot results
plt.figure(figsize=(15,6))
plt.plot(y_test_msft_actual.T, label='Actual')
plt.plot(predictions_msft, label='Predicted')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions_msft - y_test_msft_actual.T)**2))
print(f'RMSE: {rmse}')

# Data Visualization and Analysis
def plot_stock_analysis(df, company_name):
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # Plot 1: Close Price with Moving Averages
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.8)
    ax1.plot(df.index, df['MA20'], label='20-day MA', alpha=0.7)
    ax1.plot(df.index, df['MA50'], label='50-day MA', alpha=0.7)
    ax1.plot(df.index, df['MA200'], label='200-day MA', alpha=0.7)
    ax1.set_title(f'{company_name} Stock Price and Moving Averages')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Close Price History
    ax2.plot(df.index, df['Close'], label='Close Price')
    ax2.fill_between(df.index, df['Close'], alpha=0.3)
    ax2.set_title(f'{company_name} Close Price History')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.grid(True)
    
    # Plot 3: Daily Returns Distribution
    ax3.hist(df['Returns'].dropna(), bins=100, density=True, alpha=0.7)
    ax3.set_title(f'{company_name} Daily Returns Distribution')
    ax3.set_xlabel('Daily Returns')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    print(f"\n{company_name} Statistical Summary:")
    print("Average Daily Return:", df['Returns'].mean() * 100, "%")
    print("Daily Return Std Dev:", df['Returns'].std() * 100, "%")
    print("Maximum Daily Return:", df['Returns'].max() * 100, "%")
    print("Minimum Daily Return:", df['Returns'].min() * 100, "%")

# Analyze each company's data
# Microsoft Analysis
msft['Date'] = pd.to_datetime(msft['Date'])
msft.set_index('Date', inplace=True)
plot_stock_analysis(msft, 'Microsoft')

# Google Analysis
googl['Date'] = pd.to_datetime(googl['Date'])
googl.set_index('Date', inplace=True)
plot_stock_analysis(googl, 'Google')

# Apple Analysis
aapl['Date'] = pd.to_datetime(aapl['Date'])
aapl.set_index('Date', inplace=True)
plot_stock_analysis(aapl, 'Apple')

# Compare returns of all companies
plt.figure(figsize=(15, 6))
plt.plot(msft.index, msft['Close']/msft['Close'].iloc[0], label='Microsoft')
plt.plot(googl.index, googl['Close']/googl['Close'].iloc[0], label='Google')
plt.plot(aapl.index, aapl['Close']/aapl['Close'].iloc[0], label='Apple')
plt.title('Normalized Price Comparison')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()

# NVIDIA Analysis
nvda['Date'] = pd.to_datetime(nvda['Date'])
nvda.set_index('Date', inplace=True)
plot_stock_analysis(nvda, 'NVIDIA')

# META Analysis
meta['Date'] = pd.to_datetime(meta['Date'])
meta.set_index('Date', inplace=True)
plot_stock_analysis(meta, 'META Platforms')

# IBM Analysis
ibm['Date'] = pd.to_datetime(ibm['Date'])
ibm.set_index('Date', inplace=True)
plot_stock_analysis(ibm, 'IBM')

# Adobe Analysis
adbe['Date'] = pd.to_datetime(adbe['Date'])
adbe.set_index('Date', inplace=True)
plot_stock_analysis(adbe, 'Adobe')

# Johnson&Johnson Analysis
jnj['Date'] = pd.to_datetime(jnj['Date'])
jnj.set_index('Date', inplace=True)
plot_stock_analysis(jnj, 'Johnson&Johnson')

# Verizon Analysis
vz['Date'] = pd.to_datetime(vz['Date'])
vz.set_index('Date', inplace=True)
plot_stock_analysis(vz, 'Verizon')

# United Health Group Analysis
unh['Date'] = pd.to_datetime(unh['Date'])
unh.set_index('Date', inplace=True)
plot_stock_analysis(unh, 'United Health Group')

# AT&T Analysis
t['Date'] = pd.to_datetime(t['Date'])
t.set_index('Date', inplace=True)
plot_stock_analysis(t, 'AT&T')

# UPS Analysis
ups['Date'] = pd.to_datetime(ups['Date'])
ups.set_index('Date', inplace=True)
plot_stock_analysis(ups, 'UPS')

# Add a comparison plot for all companies
plt.figure(figsize=(15, 8))
plt.plot(msft.index, msft['Close']/msft['Close'].iloc[0], label='Microsoft')
plt.plot(googl.index, googl['Close']/googl['Close'].iloc[0], label='Google')
plt.plot(aapl.index, aapl['Close']/aapl['Close'].iloc[0], label='Apple')
plt.plot(nvda.index, nvda['Close']/nvda['Close'].iloc[0], label='NVIDIA')
plt.plot(meta.index, meta['Close']/meta['Close'].iloc[0], label='META')
plt.plot(ibm.index, ibm['Close']/ibm['Close'].iloc[0], label='IBM')
plt.plot(adbe.index, adbe['Close']/adbe['Close'].iloc[0], label='Adobe')
plt.plot(jnj.index, jnj['Close']/jnj['Close'].iloc[0], label='Johnson&Johnson')
plt.plot(vz.index, vz['Close']/vz['Close'].iloc[0], label='Verizon')
plt.plot(unh.index, unh['Close']/unh['Close'].iloc[0], label='United Health Group')
plt.plot(t.index, t['Close']/t['Close'].iloc[0], label='AT&T')
plt.plot(ups.index, ups['Close']/ups['Close'].iloc[0], label='UPS')
plt.title('Normalized Price Comparison - All Companies')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()

# Add a correlation heatmap
plt.figure(figsize=(12, 8))
# Create a dataframe with all stock closes
all_stocks = pd.DataFrame({
    'Microsoft': msft['Close'],
    'Google': googl['Close'],
    'Apple': aapl['Close'],
    'NVIDIA': nvda['Close'],
    'META': meta['Close'],
    'IBM': ibm['Close'],
    'Adobe': adbe['Close'],
    'Johnson&Johnson': jnj['Close'],
    'Verizon': vz['Close'],
    'United Health Group': unh['Close'],
    'AT&T': t['Close'],
    'UPS': ups['Close']
})
# Calculate correlation matrix
correlation_matrix = all_stocks.corr()
# Create heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Stock Price Correlation Matrix')
plt.show()

# Add volume comparison
plt.figure(figsize=(15, 8))
fig, axs = plt.subplots(7, 1, figsize=(15, 20))
fig.suptitle('Trading Volume Comparison')

# Microsoft Volume
axs[0].fill_between(msft.index, msft['Volume'], alpha=0.3)
axs[0].set_title('Microsoft Trading Volume')
axs[0].grid(True)

# Google Volume
axs[1].fill_between(googl.index, googl['Volume'], alpha=0.3)
axs[1].set_title('Google Trading Volume')
axs[1].grid(True)

# Apple Volume
axs[2].fill_between(aapl.index, aapl['Volume'], alpha=0.3)
axs[2].set_title('Apple Trading Volume')
axs[2].grid(True)

# NVIDIA Volume
axs[3].fill_between(nvda.index, nvda['Volume'], alpha=0.3)
axs[3].set_title('NVIDIA Trading Volume')
axs[3].grid(True)

# META Volume
axs[4].fill_between(meta.index, meta['Volume'], alpha=0.3)
axs[4].set_title('META Trading Volume')
axs[4].grid(True)

# IBM Volume
axs[5].fill_between(ibm.index, ibm['Volume'], alpha=0.3)
axs[5].set_title('IBM Trading Volume')
axs[5].grid(True)

# Adobe Volume
axs[6].fill_between(adbe.index, adbe['Volume'], alpha=0.3)
axs[6].set_title('Adobe Trading Volume')
axs[6].grid(True)

# Johnson&Johnson Volume
axs[7].fill_between(jnj.index, jnj['Volume'], alpha=0.3)
axs[7].fill_between('Johnson&Johnson Trading Volume')
axs[7].grid(True)

# Verizon Volume
axs[8].fill_between(vz.index, vz['Volume'], alpha=0.3)
axs[8].fill_between('Verizon Trading Volume')
axs[8].grid(True)

# United Health Group Volume
axs[9].fill_between(unh.index, unh['Volume'], alpha=0.3)
axs[9].fill_between('United Health Group Trading Volume')
axs[9].grid(True)

# AT&T Volume
axs[10].fill_between(t.index, t['Volume'], alpha=0.3)
axs[10].fill_between('AT&T Trading Volume')
axs[10].grid(True)

# UPS Volume    
axs[11].fill_between(ups.index, ups['Volume'], alpha=0.3)
axs[11].fill_between('UPS Trading Volume')
axs[11].grid(True)

plt.tight_layout()
plt.show()

# Print summary statistics for all companies
companies = {
    'Microsoft': msft,
    'Google': googl,
    'Apple': aapl,
    'NVIDIA': nvda,
    'META': meta,
    'IBM': ibm,
    'Adobe': adbe,
    'Johnson&Johnson': jnj,
    'Verizon': vz,
    'United Health Group': unh,
    'AT&T': t,
    'UPS': ups
}

print("\nComparative Analysis of All Companies:")
print("=====================================")
for company_name, df in companies.items():
    print(f"\n{company_name} Summary Statistics:")
    print("----------------------------")
    print(f"Average Daily Return: {df['Returns'].mean() * 100:.2f}%")
    print(f"Daily Return Std Dev: {df['Returns'].std() * 100:.2f}%")
    print(f"Maximum Daily Return: {df['Returns'].max() * 100:.2f}%")
    print(f"Minimum Daily Return: {df['Returns'].min() * 100:.2f}%")
    print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"52-Week High: ${df['Close'].rolling(window=252).max().iloc[-1]:.2f}")
    print(f"52-Week Low: ${df['Close'].rolling(window=252).min().iloc[-1]:.2f}")

