#!/usr/bin/env python3
"""
重新计算增强版策略的所有指标
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def recalculate_all_metrics():
    """重新计算所有回测指标"""
    
    print("🔄 重新计算增强版策略所有指标")
    print("=" * 80)
    
    # 1. 读取交易数据
    trades_df = pd.read_csv('logs/EnhancedHybridStrategy_trades.csv')
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    print(f"📊 原始交易记录数: {len(trades_df)}")
    
    # 2. 计算成对交易
    trade_pairs = []
    open_positions = {}
    
    for _, trade in trades_df.iterrows():
        symbol = trade['symbol']
        side = trade['side']
        price = trade['fill_price']
        quantity = trade['fill_quantity']
        commission = trade['commission']
        timestamp = trade['timestamp']
        
        if side == 'BUY':
            # 开仓
            open_positions[symbol] = {
                'entry_price': price,
                'quantity': quantity,
                'entry_time': timestamp,
                'entry_commission': commission
            }
        elif side == 'SELL' and symbol in open_positions:
            # 平仓
            pos = open_positions[symbol]
            
            # 计算盈亏 (考虑2x杠杆)
            leverage = 2.0
            price_change_pct = (price - pos['entry_price']) / pos['entry_price']
            leveraged_return_pct = price_change_pct * leverage
            
            # 计算实际盈亏金额
            position_value = pos['quantity'] * pos['entry_price']
            gross_pnl = position_value * leveraged_return_pct
            total_commission = pos['entry_commission'] + commission
            net_pnl = gross_pnl - total_commission
            
            # 计算持仓时间
            hold_time = timestamp - pos['entry_time']
            hold_hours = hold_time.total_seconds() / 3600
            
            trade_pairs.append({
                'symbol': symbol,
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'entry_price': pos['entry_price'],
                'exit_price': price,
                'quantity': pos['quantity'],
                'position_value': position_value,
                'price_change_pct': price_change_pct * 100,
                'leveraged_return_pct': leveraged_return_pct * 100,
                'gross_pnl': gross_pnl,
                'total_commission': total_commission,
                'net_pnl': net_pnl,
                'hold_hours': hold_hours,
                'is_profitable': net_pnl > 0
            })
            
            del open_positions[symbol]
    
    # 转换为DataFrame
    pairs_df = pd.DataFrame(trade_pairs)
    
    print(f"📈 完整交易对数: {len(pairs_df)}")
    print(f"📉 未平仓交易数: {len(open_positions)}")
    
    # 3. 计算基础指标
    total_trades = len(pairs_df)
    winning_trades = len(pairs_df[pairs_df['is_profitable']])
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 总盈亏
    total_pnl = pairs_df['net_pnl'].sum()
    total_commission = pairs_df['total_commission'].sum()
    
    # 盈亏分析
    winning_pnl = pairs_df[pairs_df['is_profitable']]['net_pnl'].sum()
    losing_pnl = pairs_df[~pairs_df['is_profitable']]['net_pnl'].sum()
    
    avg_win = pairs_df[pairs_df['is_profitable']]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = pairs_df[~pairs_df['is_profitable']]['net_pnl'].mean() if losing_trades > 0 else 0
    
    profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
    
    # 4. 计算收益率和回撤
    initial_capital = 100000  # 初始资金
    
    # 按时间顺序计算累计收益
    pairs_df_sorted = pairs_df.sort_values('exit_time')
    cumulative_pnl = pairs_df_sorted['net_pnl'].cumsum()
    portfolio_values = initial_capital + cumulative_pnl
    
    # 总收益率
    final_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else initial_capital
    total_return_pct = (final_value - initial_capital) / initial_capital * 100
    
    # 最大回撤
    peak_values = portfolio_values.cummax()
    drawdowns = (portfolio_values - peak_values) / peak_values * 100
    max_drawdown = drawdowns.min()
    
    # 5. 计算年化收益率和夏普比率
    start_date = pairs_df['entry_time'].min()
    end_date = pairs_df['exit_time'].max()
    total_days = (end_date - start_date).days
    years = total_days / 365.25
    
    annualized_return = ((final_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # 计算日收益率的标准差
    daily_returns = []
    current_value = initial_capital
    
    for _, trade in pairs_df_sorted.iterrows():
        daily_return = trade['net_pnl'] / current_value
        daily_returns.append(daily_return)
        current_value += trade['net_pnl']
    
    if len(daily_returns) > 1:
        returns_std = np.std(daily_returns) * np.sqrt(252)  # 年化标准差
        sharpe_ratio = (annualized_return / 100) / returns_std if returns_std > 0 else 0
    else:
        sharpe_ratio = 0
    
    # 6. 输出完整报告
    print("\n" + "="*80)
    print("📊 重新计算的完整指标报告")
    print("="*80)
    
    print(f"\n💰 收益指标:")
    print(f"   总收益率: {total_return_pct:.2f}%")
    print(f"   年化收益率: {annualized_return:.2f}%")
    print(f"   总盈亏: ${total_pnl:,.2f}")
    print(f"   初始资金: ${initial_capital:,.2f}")
    print(f"   最终资金: ${final_value:,.2f}")
    
    print(f"\n📈 交易统计:")
    print(f"   总交易数: {total_trades}")
    print(f"   盈利交易: {winning_trades}")
    print(f"   亏损交易: {losing_trades}")
    print(f"   胜率: {win_rate:.2f}%")
    
    print(f"\n💵 盈亏分析:")
    print(f"   平均盈利: ${avg_win:,.2f}")
    print(f"   平均亏损: ${avg_loss:,.2f}")
    print(f"   盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   盈亏比: ∞")
    print(f"   盈利因子: {profit_factor:.2f}")
    print(f"   总盈利: ${winning_pnl:,.2f}")
    print(f"   总亏损: ${losing_pnl:,.2f}")
    
    print(f"\n⚠️  风险指标:")
    print(f"   最大回撤: {max_drawdown:.2f}%")
    print(f"   夏普比率: {sharpe_ratio:.3f}")
    print(f"   总手续费: ${total_commission:,.2f}")
    print(f"   手续费占比: {(total_commission/abs(total_pnl)*100):.2f}%" if total_pnl != 0 else "   手续费占比: N/A")
    
    print(f"\n⏱️  时间分析:")
    print(f"   回测期间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"   总天数: {total_days} 天")
    print(f"   平均持仓时间: {pairs_df['hold_hours'].mean():.1f} 小时")
    
    # 7. 最佳和最差交易
    print(f"\n🏆 最佳交易 (前5名):")
    best_trades = pairs_df.nlargest(5, 'net_pnl')
    for i, (_, trade) in enumerate(best_trades.iterrows(), 1):
        print(f"   {i}. ${trade['net_pnl']:,.2f} ({trade['leveraged_return_pct']:.2f}%) - {trade['exit_time'].strftime('%Y-%m-%d')}")
    
    print(f"\n💸 最差交易 (前5名):")
    worst_trades = pairs_df.nsmallest(5, 'net_pnl')
    for i, (_, trade) in enumerate(worst_trades.iterrows(), 1):
        print(f"   {i}. ${trade['net_pnl']:,.2f} ({trade['leveraged_return_pct']:.2f}%) - {trade['exit_time'].strftime('%Y-%m-%d')}")
    
    # 8. 保存详细交易记录
    pairs_df.to_csv('logs/enhanced_strategy_trade_pairs.csv', index=False)
    print(f"\n💾 详细交易记录已保存至: logs/enhanced_strategy_trade_pairs.csv")
    
    # 9. 月度收益分析
    pairs_df_sorted['month'] = pairs_df_sorted['exit_time'].dt.to_period('M')
    monthly_pnl = pairs_df_sorted.groupby('month')['net_pnl'].sum()
    
    print(f"\n📅 月度收益分析:")
    for month, pnl in monthly_pnl.items():
        print(f"   {month}: ${pnl:,.2f}")
    
    return {
        'total_return_pct': total_return_pct,
        'annualized_return': annualized_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'total_commission': total_commission
    }

if __name__ == "__main__":
    metrics = recalculate_all_metrics() 