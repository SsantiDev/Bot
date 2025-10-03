#Feature Engineering para Bot de Trading de Oro
#Este script crea los indicadores técnicos necesarios para el análisis y la toma de decisiones en el trading de oro.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')


class GoldFeatureEngineering:
    """Clase para crear features técnicas para el trading de oro."""
    
    def __init__(self, df):
        """Inicializa con un DataFrame que debe contener columnas: Open, High, Low, Close, Volume.
        y un indice de tipo datatime.
        """
        self.df = df.copy()
        self.features_created = []
        
    def add_moving_averages(self):
        """Agrega medias móviles simples y exponenciales."""
        print("Agregando medias móviles...")
        
        #Medias moviles simples
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            sma = SMAIndicator(close=self.df['Close'], window=period)
            self.df[f'SMA_{period}'] = sma.sma_indicator()
            self.features_created.append(f'SMA_{period}')

        #Medias moviles exponenciales 
        for period in [9, 12, 21, 26, 50]:
            ema = EMAIndicator(close=self.df['Close'], window=period)
            self.df[f'EMA_{period}'] = ema.ema_indicator()
            self.features_created.append(f'EMA_{period}')
            
        #Cruce de medias moviles (Señales importantes)
        self.df['SMA_Cross_20_50'] = np.where(self.df['SMA_20'] > self.df['SMA_50'], 1, 0)
        self.df['EMA_Cross_12_26'] = np.where(self.df['EMA_12'] > self.df['EMA_26'], 1, 0)
        
        print(f" {len(periods)*2 + 2} indicadores de medias móviles agregados.")
        
    def add_momentum_indicators(self):
        """Agrega indicadores de momentum."""
        print("Agregando indicadores de momentum...")
        
        #RSI (Indice de Fuerza Relativa)
        for period in [14, 21]:
            rsi = RSIIndicator(close=self.df['Close'], window=period)
            self.df[f'RSI_{period}'] = rsi.rsi()
            self.features_created.append(f'RSI_{period}')
            
        #Stochastic Oscillator
        stoch = StochasticOscillator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=14, smooth_window=3)
        self.df['Stoch_K'] = stoch.stoch()
        self.df['Stoch_D'] = stoch.stoch_signal()
        self.features_created.extend(['Stoch_K', 'Stoch_D'])
        
        #MACD (Moving Average Convergence Divergence)
        macd = MACD(close=self.df['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        self.df['MACD_Diff'] = macd.macd_diff()
        self.features_created.extend(['MACD', 'MACD_Signal', 'MACD_Diff'])
        
        #Rate of Change (ROC)
        for period in [10, 20]:
            self.df[f'ROC_{period}'] = (
                (self.df['Close'] - self.df['Close'].shift(period)) / 
                self.df['Close'].shift(period) * 100
            )
            self.features_created.append(f'ROC_{period}')
            
        print(f"9 indicadores de momentum agregados.")
        
    def add_volatility_indicators(self):
        """Agrega indicadores de volatilidad."""
        print("Agregando indicadores de volatilidad...")
        
        #ATR (Average True Range) - Importante para Stop Loss
        for period in [14, 20]:
            atr = AverageTrueRange(
                high=self.df['High'], 
                low=self.df['Low'],
                close=self.df['Close'],
                window=period
            )
            self.df[f'ATR_{period}'] = atr.average_true_range()
            self.features_created.append(f'ATR_{period}')
            
            
        #Bollinger Bands
        bb = BollingerBands(close=self.df['Close'], window=20, window_dev=2)
        self.df['BB_High'] = bb.bollinger_hband()
        self.df['BB_Mid'] = bb.bollinger_mavg()
        self.df['BB_Low'] = bb.bollinger_lband()
        self.df['BB_Width'] = bb.bollinger_wband()
        self.df['BB_Pct'] = bb.bollinger_pband()
        self.features_created.extend(['BB_High', 'BB_Mid', 'BB_Low', 'BB_Width', 'BB_Pct'])
        
        #Volatilidad historica (Rolling Std Dev)
        for period in [10, 20, 30]:
            returns = self.df['Close'].pct_change()
            self.df[f'Volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)
            self.features_created.append(f'Volatility_{period}')
            
        print(f" 10 indicadores de volatilidad agregados.")
        
        
    def add_volume_indicators(self):
        """Agrega indicadores basados en volumen."""
        if 'Volume' not in self.df.columns:
            print("No hay datos de volumen, saltando indicadores de volumen.")
            return
        
        print("Agregando indicadores de volumen...")
        
        #On Balance Volume (OBV)
        obv = OnBalanceVolumeIndicator(close=self.df['Close'], volume=self.df['Volume'])
        self.df['OBV'] = obv.on_balance_volume()
        self.features_created.append('OBV')
        
        
        #Volume Moving Averages
        for period in [10, 20]:
            self.df[f'Volume_SMA_{period}'] = self.df['Volume'].rolling(window=period).mean()
            self.features_created.append(f'Volume_SMA_{period}')
            
        #Volume Ratio (Volumen actual / Volumen promedio)
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_SMA_20']
        self.features_created.append('Volume_Ratio')
        
        print(f" 4 indicadores de volumen agregados.")
        
     