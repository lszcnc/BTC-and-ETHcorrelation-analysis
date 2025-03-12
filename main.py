import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# 初始化交易所 API
exchange = ccxt.binance({
    'apiKey': '',  # 如果需要可以添加API密钥
    'secret': '',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
        'recvWindow': 60000
    }
})

    

# 存储数据
data = pd.DataFrame(columns=["timestamp", "BTC", "ETH", "ratio", "mean", "std", "z_score"])

# 配置参数
window = 30  # 滚动窗口
alpha = 2  # Z-score 阈值
time_interval = 10  # 每隔10秒更新一次数据
max_retries = 3  # 最大重试次数

# 交易参数
position = {
    'BTC': 0,  # BTC仓位，正数表示做多，负数表示做空
    'ETH': 0,  # ETH仓位，正数表示做多，负数表示做空
    'entry_ratio': None,  # 开仓时的BTC/ETH比率
    'entry_time': None  # 开仓时间
}

# 交易费率和滑点设置
fee_rate = 0.001  # 交易费率 0.1%
slippage = 0.001  # 滑点 0.1%

def calculate_position_size(btc_price, eth_price):
    """计算开仓数量，确保BTC和ETH的价值相等"""
    # 这里假设每次交易使用1000 USDT
    trade_value = 1000
    btc_amount = trade_value / btc_price
    eth_amount = trade_value / eth_price
    return btc_amount, eth_amount

def check_trading_signals(row):
    """检查交易信号并执行交易"""
    global position
    
    # 如果当前没有持仓
    if position['BTC'] == 0 and position['ETH'] == 0:
        # 检查开仓信号
        if row['z_score'] < -alpha:  # 比率偏低，做多BTC/ETH
            btc_amount, eth_amount = calculate_position_size(row['BTC'], row['ETH'])
            position['BTC'] = btc_amount
            position['ETH'] = -eth_amount
            position['entry_ratio'] = row['ratio']
            position['entry_time'] = row['timestamp']
            print(f"开仓信号 - 做多BTC/做空ETH: BTC数量={btc_amount:.6f}, ETH数量={eth_amount:.6f}, 比率={row['ratio']:.4f}")
            
        elif row['z_score'] > alpha:  # 比率偏高，做空BTC/ETH
            btc_amount, eth_amount = calculate_position_size(row['BTC'], row['ETH'])
            position['BTC'] = -btc_amount
            position['ETH'] = eth_amount
            position['entry_ratio'] = row['ratio']
            position['entry_time'] = row['timestamp']
            print(f"开仓信号 - 做空BTC/做多ETH: BTC数量={btc_amount:.6f}, ETH数量={eth_amount:.6f}, 比率={row['ratio']:.4f}")
    
    # 如果已有持仓，检查平仓信号
    elif position['entry_ratio'] is not None:
        # 计算收益
        if abs(row['z_score']) < 0.5:  # 当比率回归到均值附近时平仓
            btc_pnl = position['BTC'] * (row['BTC'] - position['entry_ratio'] * row['ETH'])
            eth_pnl = position['ETH'] * (row['ETH'] - row['BTC'] / position['entry_ratio'])
            total_pnl = btc_pnl + eth_pnl
            
            # 计算持仓时间
            hold_time = row['timestamp'] - position['entry_time']
            
            print(f"平仓信号 - 持仓时间: {hold_time}, 总收益: {total_pnl:.2f} USDT")
            
            # 清空仓位
            position['BTC'] = 0
            position['ETH'] = 0
            position['entry_ratio'] = None
            position['entry_time'] = None


def fetch_prices():
    """实时获取 BTC 和 ETH 价格"""
    for retry in range(max_retries):
        try:
            btc_ticker = exchange.fetch_ticker('BTC/USDT')
            eth_ticker = exchange.fetch_ticker('ETH/USDT')

            btc_price = btc_ticker['last']
            eth_price = eth_ticker['last']

            print(f"获取价格: BTC={btc_price}, ETH={eth_price}")  # 调试信息
            return btc_price, eth_price

        except Exception as e:
            print(f"获取价格失败 (尝试 {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                print(f"等待 {time_interval} 秒后重试...")
                plt.pause(time_interval)  # 使用 plt.pause 代替 time.sleep
            else:
                print("达到最大重试次数，将在下一个周期重试")
    return None, None


def update_data():
    """更新数据并计算比值、均值、标准差、Z-score"""
    global data
    btc_price, eth_price = fetch_prices()

    if btc_price is None or eth_price is None:
        return data  # 避免空数据进入 DataFrame

    timestamp = datetime.now()
    ratio = btc_price / eth_price

    new_row = pd.DataFrame([{"timestamp": timestamp, "BTC": btc_price, "ETH": eth_price, "ratio": ratio}])

    if data.empty:
        data = new_row  # 直接赋值，避免 concat 空 DataFrame 警告
    else:
        data = pd.concat([data, new_row], ignore_index=True)

    # 计算均值和标准差
    if len(data) > window:
        data["mean"] = data["ratio"].rolling(window).mean()
        data["std"] = data["ratio"].rolling(window).std()
        data["z_score"] = (data["ratio"] - data["mean"]) / data["std"]
        
        # 检查最新的交易信号
        latest_row = data.iloc[-1]
        if not pd.isna(latest_row['z_score']):
            check_trading_signals(latest_row)

    print(data.tail(3))  # 调试信息，查看数据是否正确
    return data


# 画图
fig, ax = plt.subplots(figsize=(12, 6))


def animate(i):
    global data
    try:
        data = update_data()

        if len(data) < window:
            print("数据不足，等待更多数据...")
            return

        ax.clear()
        ax.plot(data["timestamp"], data["ratio"], label="BTC/ETH Ratio", color="blue")
        ax.plot(data["timestamp"], data["mean"], label="Mean", linestyle="dashed", color="red")
        ax.fill_between(data["timestamp"],
                        data["mean"] + alpha * data["std"],
                        data["mean"] - alpha * data["std"],
                        color='gray', alpha=0.3, label="2 Std Dev")

        # 交易信号
        buy_signals = data[data["z_score"] < -alpha]
        sell_signals = data[data["z_score"] > alpha]

        ax.scatter(buy_signals["timestamp"], buy_signals["ratio"], color="green", marker="^", label="Buy Signal", s=100)
        ax.scatter(sell_signals["timestamp"], sell_signals["ratio"], color="red", marker="v", label="Sell Signal", s=100)

        ax.set_title("BTC/ETH Ratio & Mean Reversion Signals")
        ax.set_xlabel("Time")
        ax.set_ylabel("BTC/ETH Ratio")
        ax.legend()
        ax.grid(True)
    except Exception as e:
        print(f"动画更新出错: {str(e)}")
        print("等待下一次更新...")


# 动画更新
ani = animation.FuncAnimation(fig, animate, interval=time_interval * 1000, save_count=100)

# 确保程序不会退出
plt.show(block=False)

# 持续运行，防止 Python 进程退出
try:
    while True:
        plt.pause(0.1)  # 保持 Matplotlib 窗口运行
except KeyboardInterrupt:
    print("程序正在退出...")
except Exception as e:
    print(f"发生错误: {str(e)}")
    print("程序将继续运行...")
    while True:
        try:
            plt.pause(0.1)
        except Exception:
            continue
