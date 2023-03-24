import os
import discord
import pandas as pd
import numpy as np
import requests
import logging
import asyncio
import pytz
import pandas_ta as ta
from discord.ext import commands, tasks
import requests

#test

# Set up logging
logging.basicConfig(filename='bot.log', level=logging.ERROR)

# Set up the bot with intents
intents = discord.Intents.default()
intents.members = True  # Enable member-related intents
intents.guild_messages = True  # Enable guild message intents
bot = commands.Bot(command_prefix='!', intents=intents)


# Set up Coingecko API
COINGECKO_API_URL = 'https://api.coingecko.com/api/v3'

# Define the function to get price data from Coingecko API
def get_price_data(symbol, vs_currency, days):
    logging.info('Getting price data from Coingecko API...')
    url = f'{COINGECKO_API_URL}/coins/{symbol}/market_chart?vs_currency={vs_currency}&days={days}'
    response = requests.get(url)
    data = response.json()
    print(f"Response status code: {response.status_code}")  # Add this line
    print(f"Response data: {data}")  # Add this line
    if response.status_code != 200:
        logging.error(f'Error retrieving data for {symbol}: {response.text}')
        return pd.DataFrame(columns=['timestamp', 'close'])
    price_data = data['prices']
    open_data = data['market_caps']
    try:
        price_df = pd.DataFrame(price_data, columns=['timestamp', 'close'])
        open_df = pd.DataFrame(open_data, columns=['timestamp', 'open'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
        open_df['timestamp'] = pd.to_datetime(open_df['timestamp'], unit='ms')
        price_df.set_index('timestamp', inplace=True)
        open_df.set_index('timestamp', inplace=True)
        price_df['open'] = open_df['open']

        # Convert the timezone
        local_tz = pytz.timezone('America/New_York')
        price_df.index = price_df.index.tz_localize('UTC').tz_convert(local_tz)

    except ValueError:
        logging.error(f'Error processing data for {symbol}: {price_data}')
        price_df = pd.DataFrame(columns=['timestamp', 'close'])
    return price_df

def get_periodic_table(data, sma_window, ema_window):
    sma = calculate_sma(data, sma_window)
    ema = calculate_ema(data, ema_window)
    data['sma'] = sma
    data['ema'] = ema
    data['sma_direction'] = data['sma'].diff().apply(lambda x: 'Up' if x > 0 else 'Down')
    data['ema_direction'] = data['ema'].diff().apply(lambda x: 'Up' if x > 0 else 'Down')
    
    data = data[['close', 'sma_direction', 'ema_direction']].tail(1)
    data.index = pd.DatetimeIndex(data.index)
    data.index = data.index.strftime('%m/%d %I:%M%p')
    return data.to_string()

def calculate_sma(data, window):
    sma = data['close'].rolling(window=window).mean()
    return sma

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, window):
    ema = data['close'].ewm(span=window, adjust=False).mean()
    return ema

def calculate_alligator(data, jaw_period=13, teeth_period=8, lips_period=5):
    jaw = ta.sma(data['close'], length=jaw_period)
    teeth = ta.sma(data['close'], length=teeth_period)
    lips = ta.sma(data['close'], length=lips_period)
    return jaw, teeth, lips

def calculate_accelerator_oscillator(data, short_period=5, long_period=34):
    estimated_median_price = (data['open'] + data['close']) / 2
    short_sma = ta.sma(estimated_median_price, length=short_period)
    long_sma = ta.sma(estimated_median_price, length=long_period)
    return short_sma - long_sma

def estimate_high_low(data):
    data['high'] = data[['open', 'close']].max(axis=1)
    data['low'] = data[['open', 'close']].min(axis=1)
    return data

def calculate_adx(data, period=14):
    data = estimate_high_low(data)
    adx = ta.adx(data['high'], data['low'], data['close'], length=period)
    return adx

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def macd_sma_trading_strategy(data):
    # Calculating indicators
    sma_40 = calculate_sma(data, 40)
    ema_40 = calculate_ema(data, 40)
    macd, signal_line = calculate_macd(data)
    jaw, teeth, lips = calculate_alligator(data)
    accelerator_oscillator = calculate_accelerator_oscillator(data)
    adx = calculate_adx(data)
    upper_band, lower_band = calculate_bollinger_bands(data)

    # Getting the latest values
    latest_close_price = data['close'].iloc[-1]
    latest_sma_40 = sma_40.iloc[-1]
    latest_ema_40 = ema_40.iloc[-1]
    latest_macd = macd.iloc[-1]
    latest_signal_line = signal_line.iloc[-1]
    previous_macd = macd.iloc[-2]
    previous_signal_line = signal_line.iloc[-2]
    latest_jaw = jaw.iloc[-1]
    latest_teeth = teeth.iloc[-1]
    latest_lips = lips.iloc[-1]
    latest_accelerator_oscillator = accelerator_oscillator.iloc[-1]
    latest_adx = adx.iloc[-1]
    latest_upper_band = upper_band.iloc[-1]
    latest_lower_band = lower_band.iloc[-1]
    
    # Trading strategy logic
    if (latest_macd > latest_signal_line.item() and
        latest_close_price > latest_sma_40.item() and
        latest_close_price > latest_ema_40.item() and
        latest_lips > latest_teeth.item() > latest_jaw.item() and
        latest_accelerator_oscillator > 0 and
        latest_adx.iloc[-1] > 25):
    
        return 'BULLISH: BUY'

    elif (latest_macd < latest_signal_line.item() and
        latest_close_price < latest_sma_40.item() and
        latest_close_price < latest_ema_40.item() and
        latest_lips < latest_teeth.item() < latest_jaw.item() and
        latest_accelerator_oscillator < 0 and
        latest_adx.iloc[-1] > 25):
    
        return 'BEARISH: SELL'    
    
    elif latest_macd > latest_signal_line.item() and previous_macd <= previous_signal_line.item():
        return 'BULLISH: MACD Crossover'

    elif latest_macd < latest_signal_line.item() and previous_macd >= previous_signal_line.item():
        return 'BEARISH: MACD Crossover'

    elif latest_close_price > latest_upper_band.item():
        return '🚨🚨🚨 BREAKOUT: BULLISH 🚨🚨🚨'

    elif latest_close_price < latest_lower_band.item():
        return '🚨🚨🚨 BREAKOUT: BEARISH 🚨🚨🚨'

    else:
        return 'HOLD' 
       
def alligator_trading_strategy(data):
    # Calculating indicators
    jaw, teeth, lips = calculate_alligator(data)
    accelerator_oscillator = calculate_accelerator_oscillator(data)
    adx = calculate_adx(data)
    upper_band, lower_band = calculate_bollinger_bands(data)

    # Getting the latest values
    latest_close_price = data['close'].iloc[-1]
    latest_jaw = jaw.iloc[-1]
    latest_teeth = teeth.iloc[-1]
    latest_lips = lips.iloc[-1]
    latest_accelerator_oscillator = accelerator_oscillator.iloc[-1]
    latest_adx = adx.iloc[-1]
    latest_upper_band = upper_band.iloc[-1]
    latest_lower_band = lower_band.iloc[-1]
    
    # Trading strategy logic
    if (latest_lips > latest_teeth.item() > latest_jaw.item() and
        latest_accelerator_oscillator > 0 and
        latest_adx.iloc[-1] > 25):
    
        return 'BULLISH: BUY'

    elif (latest_lips < latest_teeth.item() < latest_jaw.item() and
        latest_accelerator_oscillator < 0 and
        latest_adx.iloc[-1] > 25):
    
        return 'BEARISH: SELL'    
    
    elif latest_close_price > latest_upper_band.item():
        return '🚨🚨🚨 BREAKOUT: BULLISH 🚨🚨🚨'

    elif latest_close_price < latest_lower_band.item():
        return '🚨🚨🚨 BREAKOUT: BEARISH 🚨🚨🚨'

    else:
        return 'HOLD' 

@tasks.loop(minutes=1)
# Define a background task for sending price data
async def send_price_data():
    # Set the symbol, vs_currency, and days for the price data
    symbol = 'maker'
    vs_currency = 'usd'
    days = '1'

    # Get the latest price data
    price_data = get_price_data(symbol, vs_currency, days)
    if price_data.empty:
        print('Error: Unable to retrieve price data.')
        return

    # Generate the periodic table
    sma_window = 20
    ema_window = 20
    periodic_table = get_periodic_table(price_data, sma_window, ema_window)

    # Set the channel ID where the message will be sent
    channel_id = 1088830587101523988

    # Get the channel object by its ID
    channel = bot.get_channel(channel_id)

    # Send the message to the channel
    await channel.send(f'```Price data for {symbol.upper()}:\n{price_data.tail()}\n```')
    await channel.send(f'```\n{periodic_table}\n```')

# Define a background task for sending trading signals
previous_signals = {}

@tasks.loop(seconds=15)
async def send_trading_signal():
    # Set the symbol, vs_currency, and days for the price data
    symbol = 'maker'
    vs_currency = 'usd'
    days = '1'
    
    # Get the latest price data
    price_data = get_price_data(symbol, vs_currency, days)
    if price_data.empty:
        print('Error: Unable to retrieve price data.')
        return
    
    # Apply the trading strategy
    macd_sma_signal = macd_sma_trading_strategy(price_data)
    alligator_signal = alligator_trading_strategy(price_data)

    # Generate the periodic table
    sma_window = 20
    ema_window = 20
    periodic_table = get_periodic_table(price_data, sma_window, ema_window)
    
    # Check if the signal has changed and send the message
    signals = {
        'macd_sma': macd_sma_signal,
        'alligator': alligator_signal,
    }
    for signal_type, signal in signals.items():
        if previous_signals.get(signal_type) != signal:
            previous_signals[signal_type] = signal
            # Set the channel ID where the message will be sent
            channel_id = 1079557519757811805

            # Get the channel object by its ID
            channel = bot.get_channel(channel_id)

            # Send the message to the channel
            await channel.send(f'{signal_type.upper()} trading signal for {symbol.upper()}: {signal}')
            await channel.send(f'```\n{periodic_table}\n```')

# Start the background task when the bot is ready
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')
    send_trading_signal.start()
    send_price_data.start()
# Run the bot
TOKEN = 'MTA3OTU0NTU1MDY3MzgwNTQ2NA.GwDwXd.FI8YvbcqzpV51LD-sD7c_DsSUTDy21IywQrFRA'
bot.run(TOKEN)