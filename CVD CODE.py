def load_models(self, model_dir="models"):
        """
        Load pre-trained models from disk
        """
        # Create models directory if it doesn't exist
        Path(model_dir).mkdir(exist_ok=True)
        
        # Look for model files
        model_files = list(Path(model_dir).glob("*.joblib")) + list(Path(model_dir).glob("lstm_*"))
        
        if not model_files:
            print("No pre-trained models found.")
            return
        
        # Load each model
        for model_path in model_files:
            model_name = model_path.stem
            
            try:
                if "lstm" in model_name:
                    # Load TensorFlow model
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                else:
                    # Load scikit-learn model
                    self.models[model_name] = joblib.load(model_path)
                
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
        
        print(f"Loaded {len(self.models)} models.")

    def save_models(self, model_dir="models"):
        """
        Save trained models to disk
        """
        # Create models directory if it doesn't exist
        Path(model_dir).mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            try:
                if "lstm" in model_name:
                    # Save TensorFlow model
                    model.save(Path(model_dir) / model_name)
                else:
                    # Save scikit-learn model
                    joblib.dump(model, Path(model_dir) / f"{model_name}.joblib")
                
                print(f"Saved model: {model_name}")
            except Exception as e:
                print(f"Error saving model {model_name}: {e}")

    def optimize_parameters(self, symbol, param_grid, start_time=None, end_time=None, initial_capital=100000):
        """
        Optimize strategy parameters using grid search
        """
        print(f"Optimizing parameters for {symbol}...")
        
        # Fetch data once for all parameter combinations
        print("Fetching market data...")
        market_data = self.fetch_aggregate_market_data(symbol, start_time, end_time)
        
        # Run base pipeline steps that don't depend on parameters
        print("Calculating CVD...")
        cvd_data = self.calculate_cvd(market_data)
        
        # Prepare results storage
        results = []
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for i, combination in enumerate(combinations):
            # Update configuration with current parameters
            params = dict(zip(param_names, combination))
            for param_name, param_value in params.items():
                self.config[param_name] = param_value
            
            # Process data with current parameters
            print(f"Testing combination {i+1}/{len(combinations)}: {params}")
            try:
                # Execute pipeline with current parameters
                divergence_data = self.detect_divergences(cvd_data)
                context_data = self.add_market_context(divergence_data)
                vp_data = self.add_volume_profile_features(context_data)
                
                # Generate signals based on current parameters
                signals = self.generate_signals(vp_data, use_ml_models=False)  # Skip ML for parameter optimization
                
                # Calculate performance
                performance, metrics = self.calculate_performance_metrics(signals, initial_capital)
                
                # Store results with metrics and parameters
                result = {
                    'params': params,
                    'metrics': metrics
                }
                results.append(result)
                
                print(f"  Win Rate: {metrics['win_rate']:.2%}, Profit Factor: {metrics['profit_factor']:.2f}, "
                      f"Annual Return: {metrics['annual_return']:.2%}, Max Drawdown: {metrics['max_drawdown']:.2%}")
            
            except Exception as e:
                print(f"Error with parameter combination {i+1}: {e}")
        
        # Find best parameter set based on different metrics
        if results:
            # Sort by different metrics
            best_return = max(results, key=lambda x: x['metrics']['annual_return'])
            best_sharpe = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
            best_profit_factor = max(results, key=lambda x: x['metrics']['profit_factor'])
            
            print("\nOptimization Results:")
            print(f"Best Annual Return: {best_return['metrics']['annual_return']:.2%}")
            print(f"Parameters: {best_return['params']}")
            
            print(f"\nBest Sharpe Ratio: {best_sharpe['metrics']['sharpe_ratio']:.2f}")
            print(f"Parameters: {best_sharpe['params']}")
            
            print(f"\nBest Profit Factor: {best_profit_factor['metrics']['profit_factor']:.2f}")
            print(f"Parameters: {best_profit_factor['params']}")
            
            # Save optimization results
            opt_results_df = pd.DataFrame([
                {**{'params': str(r['params'])}, 
                **{f"metrics_{k}": v for k, v in r['metrics'].items()}}     def calculate_performance_metrics(self, signals_df, initial_capital=100000):
        """
        Calculate performance metrics for the trading signals
        """
        performance = signals_df.copy()
        
        # Initialize columns
        performance['position'] = 0.0  # Current position size
        performance['entry_price'] = np.nan  # Entry price for current position
        performance['exit_price'] = np.nan  # Exit price for closed positions
        performance['pnl'] = 0.0  # Profit and loss for each trade
        performance['capital'] = initial_capital  # Running capital
        
        # Set up tracking variables
        in_position = False
        position_type = None  # 'long' or 'short'
        entry_price = 0.0
        entry_idx = None
        position_size = 0.0
        
        # Calculate trade performance
        for i, idx in enumerate(performance.index):
            row = performance.loc[idx]
            
            # Check for exit condition if in position
            if in_position:
                # Exit long position on bearish signal
                if position_type == 'long' and row['final_bearish_signal']:
                    exit_price = row['close']
                    pnl = position_size * (exit_price - entry_price) / entry_price
                    
                    # Update performance dataframe
                    performance.loc[idx, 'exit_price'] = exit_price
                    performance.loc[idx, 'pnl'] = pnl
                    performance.loc[idx, 'capital'] = performance.loc[entry_idx, 'capital'] * (1 + pnl)
                    
                    # Reset position
                    in_position = False
                    position_type = None
                    
                # Exit short position on bullish signal
                elif position_type == 'short' and row['final_bullish_signal']:
                    exit_price = row['close']
                    pnl = position_size * (entry_price - exit_price) / entry_price
                    
                    # Update performance dataframe
                    performance.loc[idx, 'exit_price'] = exit_price
                    performance.loc[idx, 'pnl'] = pnl
                    performance.loc[idx, 'capital'] = performance.loc[entry_idx, 'capital'] * (1 + pnl)
                    
                    # Reset position
                    in_position = False
                    position_type = None
            
            # Check for entry condition if not in position
            if not in_position:
                # Enter long position
                if row['final_bullish_signal']:
                    in_position = True
                    position_type = 'long'
                    entry_price = row['close']
                    entry_idx = idx
                    position_size = abs(row['position_size'])
                    
                    # Update performance dataframe
                    performance.loc[idx, 'position'] = position_size
                    performance.loc[idx, 'entry_price'] = entry_price
                
                # Enter short position
                elif row['final_bearish_signal']:
                    in_position = True
                    position_type = 'short'
                    entry_price = row['close']
                    entry_idx = idx
                    position_size = abs(row['position_size'])
                    
                    # Update performance dataframe
                    performance.loc[idx, 'position'] = -position_size  # Negative for short
                    performance.loc[idx, 'entry_price'] = entry_price
            
            # Fill capital forward
            if i > 0 and performance.loc[idx, 'capital'] == initial_capital:
                performance.loc[idx, 'capital'] = performance.iloc[i-1]['capital']
        
        # Calculate cumulative metrics
        performance['returns'] = performance['pnl'].fillna(0)
        performance['cumulative_returns'] = (1 + performance['returns']).cumprod() - 1
        
        # Calculate summary statistics
        total_trades = (performance['pnl'] != 0).sum()
        winning_trades = (performance['pnl'] > 0).sum()
        losing_trades = (performance['pnl'] < 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = performance.loc[performance['pnl'] > 0, 'pnl'].mean() if winning_trades > 0 else 0
        avg_loss = performance.loc[performance['pnl'] < 0, 'pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades) / abs(avg_loss * losing_trades) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Calculate drawdown
        performance['peak'] = performance['capital'].cummax()
        performance['drawdown'] = (performance['capital'] - performance['peak']) / performance['peak']
        max_drawdown = performance['drawdown'].min()
        
        # Calculate annualized return and Sharpe ratio
        if len(performance) > 1:
            days = (performance.index[-1] - performance.index[0]).days
            if days > 0:
                annual_return = (performance['capital'].iloc[-1] / initial_capital) ** (365 / days) - 1
                
                # Calculate Sharpe ratio
                daily_returns = performance['returns'].fillna(0)
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            else:
                annual_return = 0
                sharpe_ratio = 0
        else:
            annual_return = 0
            sharpe_ratio = 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_capital': performance['capital'].iloc[-1],
            'total_return': (performance['capital'].iloc[-1] / initial_capital) - 1,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio
        }
        
        return performance, metrics

    def generate_performance_report(self, performance_df, metrics, save_path=None):
        """
        Generate a comprehensive performance report with visualizations
        """
        # Create figure for performance visualization
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot equity curve
        axs[0].plot(performance_df.index, performance_df['capital'], color='blue', linewidth=2)
        axs[0].set_title('Equity Curve', fontsize=14)
        axs[0].set_ylabel('Capital ($)', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        
        # Highlight drawdown periods
        underwater = performance_df['drawdown'] < 0
        for i in range(len(performance_df)-1):
            if underwater.iloc[i] and underwater.iloc[i+1]:
                axs[0].axvspan(performance_df.index[i], performance_df.index[i+1], 
                             alpha=0.2, color='red')
        
        # Plot drawdown
        axs[1].fill_between(performance_df.index, performance_df['drawdown']*100, 0, 
                         color='red', alpha=0.3)
        axs[1].set_title('Drawdown (%)', fontsize=14)
        axs[1].set_ylabel('Drawdown %', fontsize=12)
        axs[1].grid(True, alpha=0.3)
        
        # Plot trade outcomes
        trade_results = performance_df[performance_df['pnl'] != 0]['pnl']
        if len(trade_results) > 0:
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_results]
            axs[2].bar(trade_results.index, trade_results*100, color=colors, alpha=0.7)
            axs[2].set_title('Trade Results (%)', fontsize=14)
            axs[2].set_ylabel('Profit/Loss %', fontsize=12)
            axs[2].grid(True, alpha=0.3)
        
        # Add horizontal line at zero
        axs[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Format the x-axis
        for ax in axs:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Create metrics table
        metrics_table = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        # Format metrics for display
        formatted_metrics = metrics.copy()
        for k in ['win_rate', 'max_drawdown', 'total_return', 'annual_return']:
            if k in formatted_metrics:
                formatted_metrics[k] = f"{formatted_metrics[k]*100:.2f}%"
        
        formatted_metrics['final_capital'] = f"${formatted_metrics['final_capital']:,.2f}"
        if 'profit_factor' in formatted_metrics:
            formatted_metrics['profit_factor'] = f"{formatted_metrics['profit_factor']:.2f}"
        if 'average_win' in formatted_metrics:
            formatted_metrics['average_win'] = f"{formatted_metrics['average_win']*100:.2f}%"
        if 'average_loss' in formatted_metrics:
            formatted_metrics['average_loss'] = f"{formatted_metrics['average_loss']*100:.2f}%"
        if 'sharpe_ratio' in formatted_metrics:
            formatted_metrics['sharpe_ratio'] = f"{formatted_metrics['sharpe_ratio']:.2f}"
        
        # Create the metrics visualization
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Create table
        table_data = []
        for k, v in formatted_metrics.items():
            table_data.append([k.replace('_', ' ').title(), v])
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                       loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Color coding for better visualization
        for i in range(len(table_data)):
            if 'win_rate' in table_data[i][0].lower():
                win_rate_value = float(table_data[i][1].replace('%', ''))
                if win_rate_value > 50:
                    table[(i+1, 1)].set_facecolor('lightgreen')
                else:
                    table[(i+1, 1)].set_facecolor('lightcoral')
            
            if 'drawdown' in table_data[i][0].lower():
                drawdown_value = float(table_data[i][1].replace('%', ''))
                if abs(drawdown_value) < 10:
                    table[(i+1, 1)].set_facecolor('lightgreen')
                elif abs(drawdown_value) < 20:
                    table[(i+1, 1)].set_facecolor('lightyellow')
                else:
                    table[(i+1, 1)].set_facecolor('lightcoral')
            
            if 'return' in table_data[i][0].lower():
                return_value = float(table_data[i][1].replace('%', ''))
                if return_value > 0:
                    table[(i+1, 1)].set_facecolor('lightgreen')
                else:
                    table[(i+1, 1)].set_facecolor('lightcoral')
        
        plt.title('Trading Performance Metrics', fontsize=16)
        plt.tight_layout()
        
        # Save or display
        if save_path:
            fig.savefig(f"{save_path}_equity_curve.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{save_path}_metrics.png", dpi=300, bbox_inches='tight')
            
            # Also save raw data
            performance_df.to_csv(f"{save_path}_performance_data.csv")
            metrics_table.to_csv(f"{save_path}_metrics.csv", index=False)
            
            print(f"Performance report saved to {save_path}")
        else:
            plt.show()
        
        return fig, fig2

    def run_backtest(self, symbol, start_time=None, end_time=None, initial_capital=100000):
        """
        Run a complete backtest of the CVD strategy
        """
        print(f"Running backtest for {symbol}...")
        
        # Fetch data
        print("Fetching market data...")
        market_data = self.fetch_aggregate_market_data(symbol, start_time, end_time)
        
        # Process data through pipeline
        print("Calculating CVD...")
        cvd_data = self.calculate_cvd(market_data)
        
        print("Detecting divergences...")
        divergence_data = self.detect_divergences(cvd_data)
        
        print("Adding market context...")
        context_data = self.add_market_context(divergence_data)
        
        print("Adding volume profile analysis...")
        vp_data = self.add_volume_profile_features(context_data)
        
        # Prepare ML features
        print("Preparing machine learning features...")
        ml_features = self.prepare_features_for_ml(vp_data)
        
        # Train ML models if none exist
        if not self.models:
            print("Training machine learning models...")
            for target_days in [1, 5, 10]:
                self.train_ml_models(ml_features, target_days=target_days)
        
        # Generate trading signals
        print("Generating trading signals...")
        signals = self.generate_signals(vp_data, use_ml_models=True)
        
        # Calculate performance
        print("Calculating performance metrics...")
        performance, metrics = self.calculate_performance_metrics(signals, initial_capital)
        
        # Generate performance report
        print("Generating performance report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"visualizations/{symbol.replace('/', '_')}_{timestamp}"
        self.generate_performance_report(performance, metrics, save_path)
        
        # Create trading signals visualization
        self.visualize_signals(signals, save_path=f"{save_path}_signals.png")
        
        print("Backtest complete!")
        return performance, metrics, signalsimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import ccxt
import ta
import requests
import time
import warnings
from datetime import datetime
import joblib
from pathlib import Path
warnings.filterwarnings('ignore')

class UltimateCVDSystem:
    def __init__(self, config=None):
        """
        Initialize the CVD system with configuration options
        """
        self.config = config or {
            "exchanges": ["binance", "coinbase", "kraken"],  # For crypto
            "timeframes": ["1h", "4h", "1d"],               # Multiple timeframes
            "volume_ema_periods": [9, 21, 50],              # EMAs for volume
            "cvd_ema_periods": [9, 21, 50],                 # EMAs for CVD
            "divergence_lookback_periods": [10, 20, 50],     # For divergence detection
            "ml_features_lookback": 100,                    # Historical data for ML
            "model_type": "ensemble",                       # 'lstm', 'ensemble', 'hybrid'
            "risk_reward_ratio": 2.5,                       # For position sizing
            "max_risk_per_trade": 0.02,                     # 2% risk per trade
            "use_accelerated_delta": True,                  # Enhanced delta calculation
            "use_auction_theory": True,                     # Include auction market theory
            "use_volume_profile": True,                     # Volume profile analysis
            "use_market_context": True,                     # Market regime detection
        }
        self.exchange_clients = self._initialize_exchanges()
        self.models = {}
        self.market_data = {}
        self.cvd_data = {}
        self.divergence_signals = {}
        
        # Create directory for model storage
        Path("models").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("visualizations").mkdir(exist_ok=True)

    def _initialize_exchanges(self):
        """Set up exchange connections based on config"""
        exchange_clients = {}
        for exchange_id in self.config["exchanges"]:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange_clients[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                })
            except Exception as e:
                print(f"Error initializing {exchange_id}: {e}")
        return exchange_clients

    def fetch_aggregate_market_data(self, symbol, start_time=None, end_time=None):
        """
        Fetch and aggregate data from multiple exchanges for the given symbol
        """
        all_data = []
        for exchange_id, client in self.exchange_clients.items():
            try:
                for timeframe in self.config["timeframes"]:
                    ohlcv = client.fetch_ohlcv(symbol, timeframe, since=start_time, limit=1000)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['exchange'] = exchange_id
                    df['timeframe'] = timeframe
                    all_data.append(df)
            except Exception as e:
                print(f"Error fetching data from {exchange_id}: {e}")
        
        # Consolidate data
        if not all_data:
            raise ValueError("No data fetched from any exchange")
        
        master_df = pd.concat(all_data)
        
        # Aggregate volume across exchanges
        aggregated = master_df.groupby(['timestamp', 'timeframe']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Convert timestamp to datetime
        aggregated['datetime'] = pd.to_datetime(aggregated['timestamp'], unit='ms')
        
        # Sort by time
        aggregated.sort_values(['timeframe', 'datetime'], inplace=True)
        
        return aggregated

    def calculate_cvd(self, df):
        """
        Calculate Cumulative Volume Delta with enhanced methodologies
        """
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Standard CVD calculation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_change'] = df['typical_price'].diff()
        
        # Create buying and selling volume
        df['buying_volume'] = df['volume'] * ((df['close'] - df['low']) / (df['high'] - df['low']))
        df['selling_volume'] = df['volume'] * ((df['high'] - df['close']) / (df['high'] - df['low']))
        
        # Handle division by zero
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Volume delta
        df['volume_delta'] = df['buying_volume'] - df['selling_volume']
        
        # Standard CVD
        df['cvd'] = df['volume_delta'].cumsum()
        
        # Add accelerated delta if configured
        if self.config["use_accelerated_delta"]:
            # Add acceleration factor based on volatility
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['volatility_factor'] = atr / atr.rolling(30).mean()
            df['accelerated_delta'] = df['volume_delta'] * df['volatility_factor']
            df['accelerated_cvd'] = df['accelerated_delta'].cumsum()
        
        # Add volume EMAs for comparison
        for period in self.config["volume_ema_periods"]:
            df[f'volume_ema_{period}'] = ta.trend.ema_indicator(df['volume'], window=period)
        
        # Add CVD EMAs for divergence detection
        for period in self.config["cvd_ema_periods"]:
            df[f'cvd_ema_{period}'] = ta.trend.ema_indicator(df['cvd'], window=period)
        
        return df

    def detect_divergences(self, df):
        """
        Detect regular and hidden divergences between price and CVD
        """
        # Find price swings (local maxima and minima)
        window = 5  # Window to look for local extrema
        
        df['price_swing_high'] = df['close'].rolling(2*window+1, center=True).apply(
            lambda x: 1 if x.iloc[window] == max(x) else 0, raw=True)
        
        df['price_swing_low'] = df['close'].rolling(2*window+1, center=True).apply(
            lambda x: 1 if x.iloc[window] == min(x) else 0, raw=True)
        
        # Find CVD swings
        df['cvd_swing_high'] = df['cvd'].rolling(2*window+1, center=True).apply(
            lambda x: 1 if x.iloc[window] == max(x) else 0, raw=True)
        
        df['cvd_swing_low'] = df['cvd'].rolling(2*window+1, center=True).apply(
            lambda x: 1 if x.iloc[window] == min(x) else 0, raw=True)
        
        # Find regular divergences (price makes higher high but CVD makes lower high)
        for lookback in self.config["divergence_lookback_periods"]:
            # Regular bearish divergence: price higher high, CVD lower high
            condition1 = df['price_swing_high'] == 1
            condition2 = df['close'] > df['close'].shift(lookback)
            condition3 = df['cvd'] < df['cvd'].shift(lookback)
            df[f'reg_bearish_div_{lookback}'] = condition1 & condition2 & condition3
            
            # Regular bullish divergence: price lower low, CVD higher low
            condition1 = df['price_swing_low'] == 1
            condition2 = df['close'] < df['close'].shift(lookback)
            condition3 = df['cvd'] > df['cvd'].shift(lookback)
            df[f'reg_bullish_div_{lookback}'] = condition1 & condition2 & condition3
            
            # Hidden bearish divergence: price lower high, CVD higher high
            condition1 = df['price_swing_high'] == 1
            condition2 = df['close'] < df['close'].shift(lookback)
            condition3 = df['cvd'] > df['cvd'].shift(lookback)
            df[f'hidden_bearish_div_{lookback}'] = condition1 & condition2 & condition3
            
            # Hidden bullish divergence: price higher low, CVD lower low
            condition1 = df['price_swing_low'] == 1
            condition2 = df['close'] > df['close'].shift(lookback)
            condition3 = df['cvd'] < df['cvd'].shift(lookback)
            df[f'hidden_bullish_div_{lookback}'] = condition1 & condition2 & condition3
        
        # Create composite divergence signals with strength
        df['bullish_divergence_strength'] = 0
        df['bearish_divergence_strength'] = 0
        
        for lookback in self.config["divergence_lookback_periods"]:
            # Weight longer timeframe divergences more heavily
            weight = lookback / sum(self.config["divergence_lookback_periods"])
            df['bullish_divergence_strength'] += (
                df[f'reg_bullish_div_{lookback}'].astype(int) * weight * 1.0 + 
                df[f'hidden_bullish_div_{lookback}'].astype(int) * weight * 0.7  # Hidden divs slightly less weight
            )
            df['bearish_divergence_strength'] += (
                df[f'reg_bearish_div_{lookback}'].astype(int) * weight * 1.0 + 
                df[f'hidden_bearish_div_{lookback}'].astype(int) * weight * 0.7
            )
        
        return df

    def add_market_context(self, df):
        """
        Add market context features for improved signal quality
        """
        if not self.config["use_market_context"]:
            return df
            
        # Add trend detection
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['trend_bullish'] = df['sma_50'] > df['sma_200']
        
        # Volatility context
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['volatility_percentile'] = df['atr_14'].rolling(90).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
        
        # Market regime detection (using volatility and trend)
        conditions = [
            (df['trend_bullish'] & (df['volatility_percentile'] < 0.3)),
            (df['trend_bullish'] & (df['volatility_percentile'] >= 0.3)),
            (~df['trend_bullish'] & (df['volatility_percentile'] < 0.3)),
            (~df['trend_bullish'] & (df['volatility_percentile'] >= 0.3))
        ]
        choices = ['Bull-Low-Vol', 'Bull-High-Vol', 'Bear-Low-Vol', 'Bear-High-Vol']
        df['market_regime'] = np.select(conditions, choices, default='Unknown')
        
        # Money flow
        df['money_flow'] = df['typical_price'] * df['volume']
        df['money_flow_index'] = ta.volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'], window=14)
        
        # Relative volume
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df

    def add_volume_profile_features(self, df, bins=100, window=30):
        """
        Add volume profile analysis features
        """
        if not self.config["use_volume_profile"]:
            return df
            
        # Rolling window to create recent volume profile
        result = df.copy()
        
        # Function to calculate point of control and value areas
        def calculate_poc_and_value_areas(window_df):
            if len(window_df) < 5:  # Need enough data
                return pd.Series({
                    'poc_price': np.nan,
                    'value_area_high': np.nan,
                    'value_area_low': np.nan,
                    'price_to_poc_ratio': np.nan
                })
                
            # Create price bins
            min_price = window_df['low'].min()
            max_price = window_df['high'].max()
            
            if min_price == max_price:  # Handle edge case
                return pd.Series({
                    'poc_price': min_price,
                    'value_area_high': max_price,
                    'value_area_low': min_price,
                    'price_to_poc_ratio': 1.0
                })
                
            price_bins = np.linspace(min_price, max_price, bins)
            
            # Initialize volume profile
            volume_profile = np.zeros(len(price_bins)-1)
            
            # Distribute volume across price range for each candle
            for _, row in window_df.iterrows():
                candle_min_idx = max(0, np.searchsorted(price_bins, row['low']) - 1)
                candle_max_idx = min(len(price_bins)-1, np.searchsorted(price_bins, row['high']))
                
                # Distribute volume proportionally across the price range
                if candle_max_idx > candle_min_idx:
                    volume_per_bin = row['volume'] / (candle_max_idx - candle_min_idx)
                    volume_profile[candle_min_idx:candle_max_idx] += volume_per_bin
            
            # Find POC (Point of Control) - price with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * 0.7
            
            # Sort price levels by volume
            sorted_indices = np.argsort(volume_profile)[::-1]
            
            # Add volume until reaching target
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                value_area_indices.append(idx)
                cumulative_volume += volume_profile[idx]
                if cumulative_volume >= target_volume:
                    break
            
            # Find high and low of value area
            min_va_idx = min(value_area_indices)
            max_va_idx = max(value_area_indices)
            
            value_area_low = (price_bins[min_va_idx] + price_bins[min_va_idx + 1]) / 2
            value_area_high = (price_bins[max_va_idx] + price_bins[max_va_idx + 1]) / 2
            
            # Calculate ratio of current price to POC
            last_price = window_df['close'].iloc[-1]
            price_to_poc_ratio = last_price / poc_price if poc_price > 0 else 1.0
            
            return pd.Series({
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'price_to_poc_ratio': price_to_poc_ratio
            })
        
        # Apply the volume profile analysis on rolling window
        vp_features = df.rolling(window).apply(
            calculate_poc_and_value_areas, raw=False)
        
        # Add to result dataframe
        result['poc_price'] = vp_features['poc_price']
        result['value_area_high'] = vp_features['value_area_high']
        result['value_area_low'] = vp_features['value_area_low']
        result['price_to_poc_ratio'] = vp_features['price_to_poc_ratio']
        
        # Add features indicating if price is above/below POC and in/out of value area
        result['price_above_poc'] = result['close'] > result['poc_price']
        result['in_value_area'] = (result['close'] >= result['value_area_low']) & (result['close'] <= result['value_area_high'])
        
        return result

    def prepare_features_for_ml(self, df):
        """
        Prepare features for machine learning models
        """
        # Create features dataframe
        features = df.copy()
        
        # Add technical indicators
        features['rsi'] = ta.momentum.rsi(features['close'], window=14)
        features['macd'] = ta.trend.macd_diff(features['close'])
        features['bb_width'] = ta.volatility.bollinger_pband(features['close'])
        
        # Price momentum features
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = features['close'].pct_change(period)
        
        # CVD features
        features['cvd_roc'] = features['cvd'].pct_change(5)
        features['cvd_acceleration'] = features['cvd_roc'].diff()
        
        # Divergence features are already added in detect_divergences()
        
        # Logarithmic transformations for some features
        for col in ['volume', 'relative_volume']:
            if col in features.columns:
                features[f'log_{col}'] = np.log1p(features[col])
        
        # Feature normalization/scaling
        for col in ['rsi', 'macd', 'bb_width']:
            if col in features.columns:
                # Min-max normalization
                min_val = features[col].min()
                max_val = features[col].max()
                if min_val != max_val:  # Avoid division by zero
                    features[f'norm_{col}'] = (features[col] - min_val) / (max_val - min_val)
        
        # Create target variables for ML
        # 1 for price increase in next N days, 0 for decrease
        for forecast_days in [1, 3, 5, 10]:
            features[f'target_{forecast_days}d'] = (
                features['close'].shift(-forecast_days) > features['close']).astype(int)
        
        # Create delta strength features
        features['delta_strength'] = features['volume_delta'] / features['volume']
        features['delta_strength_ma5'] = features['delta_strength'].rolling(5).mean()
        
        # Drop NaN values
        features.dropna(inplace=True)
        
        return features

    def train_ml_models(self, features_df, target_days=5):
        """
        Train machine learning models for predicting price movements based on CVD and divergences
        """
        print(f"Training models to predict price movement {target_days} days ahead")
        
        # Define target variable
        target_col = f'target_{target_days}d'
        
        if target_col not in features_df.columns:
            raise ValueError(f"Target column {target_col} not found in features dataframe")
            
        # Drop unnecessary columns for modeling
        drop_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']
        drop_cols += [col for col in features_df.columns if col.startswith('target_') and col != target_col]
        
        # Keep only columns that exist
        drop_cols = [col for col in drop_cols if col in features_df.columns]
        
        X = features_df.drop(drop_cols + [target_col], axis=1)
        y = features_df[target_col]
        
        # Handle categorical variables
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
            
        # Ensure no remaining NaN values
        X.fillna(0, inplace=True)
        
        # Time-based train-test split to prevent lookahead bias
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Train different model types based on configuration
        if self.config["model_type"] in ["ensemble", "hybrid"]:
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            rf_model.fit(X_train, y_train)
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            
            # Evaluate ensemble models
            rf_pred = rf_model.predict(X_test)
            gb_pred = gb_model.predict(X_test)
            
            print("\nRandom Forest Performance:")
            print(classification_report(y_test, rf_pred))
            
            print("\nGradient Boosting Performance:")
            print(classification_report(y_test, gb_pred))
            
            # Store models
            self.models[f'rf_{target_days}d'] = rf_model
            self.models[f'gb_{target_days}d'] = gb_model
            
            # Save models to disk
            joblib.dump(rf_model, f"models/rf_{target_days}d.joblib")
            joblib.dump(gb_model, f"models/gb_{target_days}d.joblib")
            
            # Feature importance
            rf_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features (Random Forest):")
            print(rf_importance.head(10))
            
            # Save feature importance
            rf_importance.to_csv(f"models/rf_{target_days}d_feature_importance.csv", index=False)
        
        if self.config["model_type"] in ["lstm", "hybrid"]:
            # Prepare data for LSTM
            sequence_length = 10  # Look back 10 time steps
            
            def create_sequences(X, y, seq_length):
                X_seq, y_seq = [], []
                for i in range(len(X) - seq_length):
                    X_seq.append(X.iloc[i:i+seq_length].values)
                    y_seq.append(y.iloc[i+seq_length])
                return np.array(X_seq), np.array(y_seq)
            
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
            
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(units=30, return_sequences=False),
                Dropout(0.2),
                BatchNormalization(),
                Dense(units=1, activation='sigmoid')
            ])
            
            lstm_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping to prevent overfitting
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train LSTM model
            print("\nTraining LSTM model...")
            lstm_history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate LSTM
            lstm_pred = (lstm_model.predict(X_test_seq) > 0.5).astype(int)
            
            print("\nLSTM Performance:")
            print(classification_report(y_test_seq, lstm_pred))
            
            # Store model
            self.models[f'lstm_{target_days}d'] = lstm_model
            
            # Save model
            lstm_model.save(f"models/lstm_{target_days}d")
            
            # Plot training history
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(lstm_history.history['loss'], label='Training Loss')
            plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Training History - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(lstm_history.history['accuracy'], label='Training Accuracy')
            plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('LSTM Training History - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"visualizations/lstm_{target_days}d_training_history.png")
        
        return self.models

    def generate_signals(self, df, use_ml_models=True):
        """
        Generate trading signals based on divergences and ML predictions
        """
        # Start with divergence signals
        signals = df.copy()
        
        # Basic signals based on divergence strength
        signals['bullish_signal'] = signals['bullish_divergence_strength'] > 0.5
        signals['bearish_signal'] = signals['bearish_divergence_strength'] > 0.5
        
        # Enhance with ML predictions if available
        if use_ml_models and self.models:
            # Prepare features
            features = self.prepare_features_for_ml(signals)
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                if model_name.startswith('rf_') or model_name.startswith('gb_'):
                    # For ensemble models
                    try:
                        days = int(model_name.split('_')[1].replace('d', ''))
                        pred_col = f'ml_pred_{model_name}'
                        # Only keep relevant columns
                        X_pred = features.drop([col for col in features.columns if col.startswith('target_')], axis=1)
                        # Remove datetime if it exists
                        if 'datetime' in X_pred.columns:
                            X_pred = X_pred.drop('datetime', axis=1)
                        
                        signals[pred_col] = model.predict_proba(X_pred)[:, 1]
                    except Exception as e:
                        print(f"Error getting predictions from {model_name}: {e}")
                
                elif model_name.startswith('lstm_'):
                    # For LSTM models - need to create sequences
                    try:
                        sequence_length = 10
                        days = int(model_name.split('_')[1].replace('d', ''))
                        
                        # Create sequences
                        X_seq = []
                        # Only keep relevant columns
                        X_pred = features.drop([col for col in features.columns if col.startswith('target_')], axis=1)
                        # Remove datetime if it exists
                        if 'datetime' in X_pred.columns:
                            X_pred = X_pred.drop('datetime', axis=1)
                            
                        for i in range(sequence_length, len(X_pred)):
                            X_seq.append(X_pred.iloc[i-sequence_length:i].values)
                        
                        X_seq = np.array(X_seq)
                        temp_preds = model.predict(X_seq).flatten()
                        
                        # Create a series of NaN values
                        predictions = np.full(len(signals), np.nan)
                        # Fill in the predictions where we have them (after sequence_length)
                        predictions[sequence_length:sequence_length+len(temp_preds)] = temp_preds
                        
                        signals[f'ml_pred_{model_name}'] = predictions
                        
                        # Fill NaN values for first sequence_length rows
                        signals[f'ml_pred_{model_name}'].fillna(0.5, inplace=True)
                    except Exception as e:
                        print(f"Error getting predictions from {model_name}: {e}")
            
            # Combine ML predictions with divergence signals
            ml_pred_cols = [col for col in signals.columns if col.startswith('ml_pred_')]
            if ml_pred_cols:
                signals['ml_bullish_consensus'] = signals[ml_pred_cols].mean(axis=1)
                signals['ml_enhanced_bullish'] = (signals['bullish_signal'] & (signals['ml_bullish_consensus'] > 0.6))
                signals['ml_enhanced_bearish'] = (signals['bearish_signal'] & (signals['ml_bullish_consensus'] < 0.4))
            else:
                # If no ML predictions, fall back to divergence signals
                signals['ml_enhanced_bullish'] = signals['bullish_signal']
                signals['ml_enhanced_bearish'] = signals['bearish_signal']
        
        # Add contextual filters
        if 'market_regime' in signals.columns:
            # Reduce signal strength in counter-trend and high volatility environments
            signals['final_bullish_signal'] = signals['ml_enhanced_bullish'] if use_ml_models else signals['bullish_signal']
            signals['final_bearish_signal'] = signals['ml_enhanced_bearish'] if use_ml_models else signals['bearish_signal']
            
            # Strengthen signals in trending environments
            signals.loc[signals['market_regime'] == 'Bull-Low-Vol', 'final_bullish_signal'] = \
                signals['final_bullish_signal'] | (signals['bullish_divergence_strength'] > 0.3)
            
            signals.loc[signals['market_regime'] == 'Bear-Low-Vol', 'final_bearish_signal'] = \
                signals['final_bearish_signal'] | (signals['bearish_divergence_strength'] > 0.3)
        else:
            signals['final_bullish_signal'] = signals['ml_enhanced_bullish'] if use_ml_models else signals['bullish_signal']
            signals['final_bearish_signal'] = signals['ml_enhanced_bearish'] if use_ml_models else signals['bearish_signal']
        
        # Calculate risk-adjusted position sizes
        signals['position_size'] = 0.0
        
        # For bullish signals
        bull_mask = signals['final_bullish_signal']
        if bull_mask.any():
            # Base position size on divergence strength
            signals.loc[bull_mask, 'position_size'] = signals.loc[bull_mask, 'bullish_divergence_strength'] * self.config['max_risk_per_trade']
            
            # Add ML confidence if available
            if use_ml_models and 'ml_bullish_consensus' in signals.columns:
                signals.loc[bull_mask, 'position_size'] *= signals.loc[bull_mask, 'ml_bullish_consensus']
        
        # For bearish signals (negative position size indicates short)
        bear_mask = signals['final_bearish_signal']
        if bear_mask.any():
            signals.loc[bear_mask, 'position_size'] = -signals.loc[bear_mask, 'bearish_divergence_strength'] * self.config['max_risk_per_trade']
            
            # Add ML confidence if available
            if use_ml_models and 'ml_bullish_consensus' in signals.columns:
                signals.loc[bear_mask, 'position_size'] *= (1 - signals.loc[bear_mask, 'ml_bullish_consensus'])
        
        return signals