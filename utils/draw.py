import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import glob

class Drawer:
    
    @staticmethod
    def draw_combined_chart(json_data, output_path='combined_chart.png'):
        """
        Draws all figures in one graph (subplots) based on the provided JSON data.
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        base_date_str = data.get('date', '2025-06-12')
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
        datasets = data.get('datasets', {})

        # 1. Prepare DataFrames
        df_pv_load = Drawer._prepare_pv_load(datasets.get('pv_load', []), base_date)
        df_price = Drawer._prepare_price(datasets.get('price', []), base_date)
        df_ai = Drawer._prepare_battery(datasets.get('ai_mode', []), base_date)
        df_self = Drawer._prepare_battery(datasets.get('self_mode', []), base_date)
        
        # Align all dataframes to a common 1-min index for calculations
        common_index = pd.date_range(start=datetime.combine(base_date, datetime.min.time()), 
                                     end=datetime.combine(base_date, datetime.max.time()), 
                                     freq='1min')
        
        def reindex_df(df):
            if df is None or df.empty:
                return pd.DataFrame(index=common_index)
            return df.reindex(common_index).interpolate(method='linear').fillna(0)

        df_pv_load_r = reindex_df(df_pv_load)
        df_price_r = reindex_df(df_price).ffill().bfill() 
        
        df_ai_r = reindex_df(df_ai)
        df_self_r = reindex_df(df_self)

        # 2. Setup Plot Layout
        # 5 Plots + 1 Table area (Removed Theoretical Mode)
        fig = plt.figure(figsize=(18, 24))
        gs = fig.add_gridspec(6, 1, height_ratios=[3, 1.5, 3, 3, 3, 2])
        plt.subplots_adjust(hspace=0.35)

        ax_pv = fig.add_subplot(gs[0])
        ax_price = fig.add_subplot(gs[1], sharex=ax_pv)
        ax_ai = fig.add_subplot(gs[2], sharex=ax_pv)
        ax_self = fig.add_subplot(gs[3], sharex=ax_pv)
        ax_grid = fig.add_subplot(gs[4], sharex=ax_pv)
        ax_table = fig.add_subplot(gs[5])
        ax_table.axis('off')

        # 3. Draw Plots
        Drawer._plot_pv_load(ax_pv, df_pv_load)
        Drawer._plot_price(ax_price, df_price)
        Drawer._plot_battery(ax_ai, df_ai, "AI mode", ax_pv)
        Drawer._plot_battery(ax_self, df_self, "self-consumption mode", ax_pv)
        Drawer._plot_grid_comparison(ax_grid, df_pv_load_r, df_ai_r, df_self_r)

        # 4. Draw Table
        Drawer._draw_table(ax_table, df_pv_load_r, df_price_r, df_ai_r, df_self_r)

        # 5. Global Formatting
        fig.suptitle(f"ps_id:{data.get('ps_id','N/A')} ,date:{base_date_str}, AI mode running result...", fontsize=14, y=0.995)
        
        # Format X-axis for the last plot
        ax_grid.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_grid.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Chart saved to {output_path}")

    @staticmethod
    def _prepare_pv_load(data, base_date):
        if not data: return pd.DataFrame()
        rows = []
        for item in data:
            dt = datetime.combine(base_date, datetime.strptime(item['time'], "%H:%M").time())
            rows.append({
                'Time': dt,
                'PV_Forecast': item.get('pv_forecast', 0),
                'PV_Real': item.get('pv_real', 0),
                'Load_Forecast': item.get('load_forecast', 0),
                'Load_Real': item.get('load_real', 0)
            })
        df = pd.DataFrame(rows).set_index('Time')
        return df.resample('1min').interpolate(method='linear')

    @staticmethod
    def _prepare_price(data, base_date):
        if not data: return pd.DataFrame()
        rows = []
        for item in data:
            dt = datetime.combine(base_date, datetime.strptime(item['time'], "%H:%M").time())
            rows.append({
                'Time': dt,
                'Price_Buy': item.get('price_buy', 0),
                'Price_Sell': item.get('price_sell', 0)
            })
        df = pd.DataFrame(rows).set_index('Time')
        # Price is step data, so we use ffill for resampling to keep steps
        return df.resample('1min').ffill()

    @staticmethod
    def _prepare_battery(data, base_date):
        if not data: return pd.DataFrame()
        rows = []
        day_offset = 0
        last_dt = None
        for item in data:
            dt_time = datetime.strptime(item['time'], "%H:%M").time()
            dt = datetime.combine(base_date + timedelta(days=day_offset), dt_time)
            if last_dt and dt.hour < last_dt.hour and last_dt.hour > 12:
                dt += timedelta(days=1)
                day_offset += 1
            
            # Calculate base for stacking
            # Discharge (Positive Total): Base is 'battery_to_load'
            # Charge (Negative Total): Base is 'battery_from_pv' (negative value)
            
            to_load = abs(item.get('battery_to_load', 0))
            to_grid = abs(item.get('battery_to_grid', 0))
            from_pv = abs(item.get('battery_from_pv', 0))
            from_grid = abs(item.get('battery_from_grid', 0))
            
            # Total Power: Discharge (+), Charge (-)
            total = (to_load + to_grid) - (from_pv + from_grid)
            
            rows.append({
                'Time': dt,
                'Total': total,
                'To_Load': to_load,
                'To_Grid': to_grid,
                'From_PV': -from_pv, # Store as negative for plotting
                'From_Grid': -from_grid, # Store as negative
                'SOC': item.get('soc', 0)
            })
            last_dt = dt
            
        df = pd.DataFrame(rows).set_index('Time')
        return df.resample('1min').interpolate(method='linear')

    @staticmethod
    def _plot_pv_load(ax, df):
        if df.empty: return
        
        # Colors
        c_pv_f = '#90ee90'
        c_pv_r = '#2ECC71' # Green fill
        c_load_f = '#E74C3C' # Pink/Red line
        c_load_r = '#F39C12' # Orange fill

        ax.plot(df.index, df['PV_Forecast'], color=c_pv_f, label='PV forecast power')
        ax.fill_between(df.index, 0, df['PV_Real'], color=c_pv_r, alpha=0.4, label='PV real power')
        
        ax.plot(df.index, df['Load_Forecast'], color=c_load_f, label='load forecast power')
        ax.fill_between(df.index, 0, df['Load_Real'], color=c_load_r, alpha=0.4, label='load real power')

        ax.set_ylabel("Power(W)")
        ax.set_ylim(-1000, 15000) # Fixed Range
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=8)

    @staticmethod
    def _plot_price(ax, df):
        if df.empty: return
        ax.step(df.index, df['Price_Buy'], where='post', color='#E67E22', label='buying price')
        ax.step(df.index, df['Price_Sell'], where='post', color='#34495E', label='selling price')
        ax.set_ylabel("electricity price")
        ax.set_ylim(0, 1.0) # Fixed Range
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8)

    @staticmethod
    def _plot_battery(ax, df, mode_name, sharex_ax=None):
        if df.empty: return
        
        ax2 = ax.twinx()
        
        # Colors
        c_line = '#34495E'
        c_from_pv = '#2ECC71' # Green
        c_from_grid = '#F1C40F' # Yellow
        c_to_load = '#E67E22' # Orange
        c_to_grid = '#3498DB' # Blue
        c_soc = '#C0392B' # Red

        # Stacked Area Plotting
        # Discharge (Positive)
        ax.fill_between(df.index, 0, df['To_Load'], color=c_to_load, alpha=0.6, label='battery_to_load')
        ax.fill_between(df.index, df['To_Load'], df['To_Load'] + df['To_Grid'], color=c_to_grid, alpha=0.6, label='battery_to_grid')
        
        # Charge (Negative)
        # From PV is bottom (closest to 0), From Grid is below it
        ax.fill_between(df.index, 0, df['From_PV'], color=c_from_pv, alpha=0.6, label='battery_from_pv')
        ax.fill_between(df.index, df['From_PV'], df['From_PV'] + df['From_Grid'], color=c_from_grid, alpha=0.6, label='battery_from_grid')

        # Line
        ax.plot(df.index, df['Total'], color=c_line, lw=1, label=f'{mode_name} battery power')
        
        # SOC
        ax2.plot(df.index, df['SOC'], color=c_soc, lw=1, label=f'{mode_name} battery SOC')
        ax2.set_ylabel("SOC")
        ax2.set_ylim(0, 1.1) # Fixed Range

        ax.set_ylabel("Power(W)")
        ax.set_ylim(-15000, 15000) # Fixed Range
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Filter duplicates if any
        by_label = dict(zip(labels1 + labels2, lines1 + lines2))
        # Custom order
        order = [f'{mode_name} battery power', 'battery_from_pv', 'battery_from_grid', 'battery_to_load', 'battery_to_grid', f'{mode_name} battery SOC']
        ordered_handles = [by_label[l] for l in order if l in by_label]
        ordered_labels = [l for l in order if l in by_label]
        
        ax.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=False, fontsize=8)

    @staticmethod
    def _plot_grid_comparison(ax, df_pv_load, df_ai, df_self):
        # Calculate Grid Power
        # Grid = Load - PV - Battery
        # Note: Battery Total > 0 is Discharge (helps load), < 0 is Charge (consumes grid/pv)
        # Grid_Import = Load - PV - Battery_Discharge + Battery_Charge_from_Grid?
        # Let's use the net formula: Grid_Net = Load - PV - Battery_Total
        # If Grid_Net > 0: Import. If < 0: Export.
        
        if df_pv_load.empty: return

        grid_ai = df_pv_load['Load_Real'] - df_pv_load['PV_Real'] - (df_ai['Total'] if not df_ai.empty else 0)
        grid_self = df_pv_load['Load_Real'] - df_pv_load['PV_Real'] - (df_self['Total'] if not df_self.empty else 0)

        c_ai = '#E67E22' # Orange
        c_self = '#2ECC71' # Green

        ax.fill_between(df_pv_load.index, 0, grid_ai, color=c_ai, alpha=0.5, label='AI mode - grid power')
        ax.fill_between(df_pv_load.index, 0, grid_self, color=c_self, alpha=0.5, label='self-consumption - grid power')
        
        ax.set_ylabel("Power(W)")
        ax.set_ylim(-15000, 15000) # Fixed Range
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8)

    @staticmethod
    def _draw_table(ax, df_pv_load, df_price, df_ai, df_self):
        # Calculate Stats
        # Interval in hours (1 min = 1/60 hours)
        interval = 1/60.0
        
        def calc_stats(df_bat, df_pl, df_pr):
            if df_bat.empty or df_pl.empty: return ['-']*9
            
            pv_kwh = df_pl['PV_Real'].sum() * interval / 1000
            load_kwh = df_pl['Load_Real'].sum() * interval / 1000
            
            # Charge: sum of negative parts
            charge_kwh = abs(df_bat[df_bat['Total'] < 0]['Total'].sum()) * interval / 1000
            discharge_kwh = df_bat[df_bat['Total'] > 0]['Total'].sum() * interval / 1000
            
            # Grid
            grid_power = df_pl['Load_Real'] - df_pl['PV_Real'] - df_bat['Total']
            fetch_kwh = grid_power[grid_power > 0].sum() * interval / 1000
            feed_kwh = abs(grid_power[grid_power < 0].sum()) * interval / 1000
            
            # Bill
            # Need to align price
            # Assuming df_pr is aligned
            if df_pr.empty:
                bill = 0
            else:
                # Cost = Fetch * BuyPrice - Feed * SellPrice
                # We need element-wise multiplication
                cost_series = (grid_power[grid_power > 0] * df_pr['Price_Buy']) - (abs(grid_power[grid_power < 0]) * df_pr['Price_Sell'])
                bill = cost_series.sum() * interval / 1000
            
            # Revenue (Savings vs Baseline?)
            # Baseline: No Battery. Grid = Load - PV.
            grid_base = df_pl['Load_Real'] - df_pl['PV_Real']
            if not df_pr.empty:
                cost_base = (grid_base[grid_base > 0] * df_pr['Price_Buy']) - (abs(grid_base[grid_base < 0]) * df_pr['Price_Sell'])
                bill_base = cost_base.sum() * interval / 1000
            else:
                bill_base = 0
                
            revenue = bill_base - bill
            rev_rate = (revenue / abs(bill_base) * 100) if bill_base != 0 else 0
            
            return [
                f"{pv_kwh:.2f}", f"{load_kwh:.2f}", f"{charge_kwh:.2f}", f"{discharge_kwh:.2f}",
                f"{feed_kwh:.2f}", f"{fetch_kwh:.2f}", f"{bill:.2f}", f"{revenue:.2f}", f"{rev_rate:.2f}"
            ]

        stats_ai = calc_stats(df_ai, df_pv_load, df_price)
        stats_self = calc_stats(df_self, df_pv_load, df_price)

        col_labels = ['PV(kWh)', 'Load(kWh)', 'Charge(kWh)', 'Discharge(kWh)', 'Feed-in(kWh)', 'Fetch(kWh)', 'Bill', 'Revenue', 'Revenue Rate(%)']
        row_labels = ['AI mode', 'Self-consumption mode']
        cell_text = [stats_ai, stats_self]

        table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

if __name__ == "__main__":
    
    try:
        base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        input_dir = os.path.join(base_dir, "../series")
        output_dir = os.path.join(base_dir, "../series_graph")
        os.makedirs(output_dir, exist_ok=True)

        # Ensure input directory exists; if not, skip processing
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist. Skipping.")
        else:
            pattern = os.path.join(input_dir, "*.json")
            files = sorted(glob.glob(pattern))
            if not files:
                print(f"No json files found in {input_dir}")
            else:
                for fp in files:
                    try:
                        name = os.path.splitext(os.path.basename(fp))[0]
                        out_path = os.path.join(output_dir, f"{name}.png")

                        if os.path.exists(out_path):
                            print(f"Skipping {fp}, output already exists at {out_path}")
                            continue

                        with open(fp, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        print(f"Processing {fp} -> {out_path}")
                        Drawer.draw_combined_chart(data, output_path=out_path)
                    except Exception as e:
                        print(f"Failed processing {fp}: {e}")
    except Exception as e:
        print(f"Error: {e}")