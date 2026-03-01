import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def compute_features(group):
    # Bollinger Bands
    # length=20, std=2
    bbands = group.ta.bbands(length=20, std=2)
    if bbands is None:
        return group
    # print(bbands.columns) # Debug
    # pandas_ta bbands returns: BBL_20_2.0 (lower), BBM_20_2.0 (mid), BBU_20_2.0 (upper), BBB_20_2.0 (bandwidth), BBP_20_2.0 (%B)
    group['bb_percent'] = bbands.iloc[:, 4] # BBP is usually the 5th column
    group['bb_width'] = bbands.iloc[:, 3]   # BBB is usually the 4th column

    # RSI
    # length=14
    group['rsi'] = group.ta.rsi(length=14)
    group['rsi_change'] = group['rsi'].diff()

    # MACD
    # fast=12, slow=26, signal=9
    macd = group.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        # returns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        group['macd'] = macd.iloc[:, 0] / group['close']
        group['macd_signal'] = macd.iloc[:, 2] / group['close']
        group['macd_hist'] = macd.iloc[:, 1] / group['close']

    return group
