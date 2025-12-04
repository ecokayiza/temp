import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api import ChatClient

import time

# transfer origin data to input data for the AI model
class Transfer:
    def __init__(self):
        self.client = ChatClient()

    def _call_api_with_retry(self, prompt, json_format=True, max_retries=3):
        for i in range(max_retries):
            try:
                return self.client.generate(prompt, json_format=json_format)
            except Exception as e:
                if "429" in str(e) or "RateLimitError" in str(e):
                    wait_time = 60 * (i + 1)
                    print(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API Error: {e}")
                    # For other errors, maybe don't retry or retry short
                    time.sleep(5)
        raise Exception("Max retries exceeded")

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process(self, input_path, output_path):
        print(f"Processing {input_path}...")
        data = self.load_data(input_path)
        base_date = data.get('date', '2025-06-12')
        df = self._prepare_dataframe(data, base_date)
        
        # 1. Global Calculations
        profit_ai = self._calc_profit(df, 'ai')
        profit_self = self._calc_profit(df, 'self')
        
        # 2. Get Segments from API
        print("Asking API for segments...")
        segments_meta = self._get_segments_from_api(df)
        
        # 3. Process Segments
        final_segments = []
        print(f"Processing {len(segments_meta)} segments...")
        
        for seg in segments_meta:
            print(f"  - Segment: {seg['start_time']} - {seg['end_time']} ({seg['period_type']})")
            # Calculate Stats
            stats = self._calc_segment_stats(df, seg['start_time'], seg['end_time'])
            
            # Get Descriptions from API
            descriptions = self._get_descriptions_from_api(seg, stats)
            
            # Merge Stats and Descriptions
            
            # Construct descriptions object with stats
            descriptions_data = {
                "pv_load": {**stats['pv_load'], "description": descriptions['descriptions']['pv_load']['description']},
                "price": {**stats['price'], "description": descriptions['descriptions']['price']['description']}
            }

            ai_mode_data = {
                "pv_action": {**stats['ai_mode']['pv_action'], "description": descriptions['ai_mode']['pv_action']['description']},
                "grid_action": {**stats['ai_mode']['grid_action'], "description": descriptions['ai_mode']['grid_action']['description']},
                "battery_soc": {**stats['ai_mode']['battery_soc'], "description": descriptions['ai_mode']['battery_soc']['description']}
            }
            
            self_mode_data = {
                "pv_action": {**stats['self_mode']['pv_action'], "description": descriptions['self_mode']['pv_action']['description']},
                "grid_action": {**stats['self_mode']['grid_action'], "description": descriptions['self_mode']['grid_action']['description']},
                "battery_soc": {**stats['self_mode']['battery_soc'], "description": descriptions['self_mode']['battery_soc']['description']}
            }

            segment_data = {
                "start_time": seg['start_time'],
                "end_time": seg['end_time'],
                "time_period": f"{seg['start_time']} - {seg['end_time']}",
                "period_type": seg['period_type'],
                "descriptions": descriptions_data, # PV/Load, Price descriptions with stats
                "ai_mode": ai_mode_data,
                "self_mode": self_mode_data
            }
            final_segments.append(segment_data)
            
        # 4. Final Output
        output = {
            "ps_id": data.get('ps_id'),
            "date": data.get('date'),
            "system_specs": {
                "battery_capacity_kWh": 22.4,
                "max_charge_power_W": 5000,
                "max_discharge_power_W": 5000,
                "soc_min_limit": 0.10,
                "soc_max_limit": 1.00
            },
            "profit": {
                "ai_mode": round(profit_ai, 2),
                "self_mode": round(profit_self, 2)
            },
            "segments": final_segments
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"Saved to {output_path}")

    # df, contains:
    """
    索引 (Index):
    Time: 时间戳，从当天的 00:00 到 23:45，每 15 分钟一个点（共 96 行）。
    列 (Columns):

    PV & Load (光伏与负载)

    pv_forecast: 光伏预测功率
    pv_real: 光伏实际功率
    load_forecast: 负载预测功率
    load_real: 负载实际功率
    Price (电价)

    price_buy: 买电价格
    price_sell: 卖电价格
    AI Mode (AI 模式数据) - 前缀 ai_

    ai_soc: 电池荷电状态 (SOC)
    ai_battery_from_pv: 电池从光伏充电功率
    ai_battery_from_grid: 电池从电网充电功率
    ai_battery_to_load: 电池向负载放电功率
    ai_battery_to_grid: 电池向电网放电功率
    ai_grid_power: 电网交互功率（正数为买电，负数为卖电，来自 grid_power 数据集）
    ai_total_power: 计算字段，电池总净功率（放电为正，充电为负）。
    计算公式：(to_load + to_grid) - (from_pv + from_grid)
    """
    def _prepare_dataframe(self, data, base_date):
        datasets = data.get('datasets', {})
        
        def to_df(data_list, prefix=''):
            if not data_list: return pd.DataFrame()
            rows = []
            for item in data_list:
                dt = datetime.combine(datetime.strptime(base_date, "%Y-%m-%d").date(), 
                                      datetime.strptime(item['time'], "%H:%M").time())
                row = {'Time': dt}
                for k, v in item.items():
                    if k != 'time':
                        row[k] = v
                rows.append(row)
            return pd.DataFrame(rows).set_index('Time')

        df_pv = to_df(datasets.get('pv_load', []))
        df_price = to_df(datasets.get('price', []))
        df_ai = to_df(datasets.get('ai_mode', []))
        df_self = to_df(datasets.get('self_mode', []))
        df_grid = to_df(datasets.get('grid_power', []))
        
        df_ai = df_ai.add_prefix('ai_')
        df_self = df_self.add_prefix('self_')
        
        # Fix double prefix if exists (e.g. ai_ai_total_power -> ai_total_power)
        df_ai.columns = [c.replace('ai_ai_', 'ai_') for c in df_ai.columns]
        df_self.columns = [c.replace('self_self_', 'self_') for c in df_self.columns]
        
        common_index = pd.date_range(start=f"{base_date} 00:00", end=f"{base_date} 23:45", freq='15min')
        
        def reindex(df):
            if df.empty: return pd.DataFrame(index=common_index)
            df = df[~df.index.duplicated(keep='first')]
            return df.reindex(common_index).interpolate(method='linear').fillna(0)

        df_pv = reindex(df_pv)
        df_price = reindex(df_price).ffill().bfill()
        df_ai = reindex(df_ai)
        df_self = reindex(df_self)
        df_grid = reindex(df_grid)
        
        df_final = pd.concat([df_pv, df_price, df_ai, df_self, df_grid], axis=1)
        
        # Calculate total battery power (Discharge +, Charge -)
        for mode in ['ai', 'self']:
            prefix = f"{mode}_"
            # Ensure columns exist (fill with 0 if missing)
            cols = ['battery_to_load', 'battery_to_grid', 'battery_from_pv', 'battery_from_grid']
            for c in cols:
                if f'{prefix}{c}' not in df_final.columns:
                    df_final[f'{prefix}{c}'] = 0
            
            # Use pre-calculated total_power if available, otherwise calculate it
            if f'{prefix}total_power' not in df_final.columns:
                df_final[f'{prefix}total_power'] = (
                    df_final[f'{prefix}battery_to_load'] + 
                    df_final[f'{prefix}battery_to_grid']
                ) - (
                    df_final[f'{prefix}battery_from_pv'] + 
                    df_final[f'{prefix}battery_from_grid']
                )

        return df_final

    def _calc_profit(self, df, mode):
        prefix = f"{mode}_"
        bat_total = df[f'{prefix}total_power']
        
        grid_col = f'{prefix}grid_power'
        if grid_col in df.columns:
            grid_net = df[grid_col]
        else:
            grid_net = df['load_real'] - df['pv_real'] - bat_total
        
        interval = 0.25
        fetch = grid_net.clip(lower=0) * interval / 1000
        feed = abs(grid_net.clip(upper=0)) * interval / 1000
        
        cost = (fetch * df['price_buy']).sum()
        revenue = (feed * df['price_sell']).sum()
        
        return revenue - cost

    def _get_segments_from_api(self, df):
        # data given
        csv_data = "Time,Price,AI_SOC,AI_Bat_Power,AI_Grid_Power,Self_SOC,Self_Bat_Power,Self_Grid_Power\n"
        # Iterate over all rows (15-min interval)
        for idx, row in df.iterrows():
            time_str = idx.strftime("%H:%M")
            
            def g(k): return row.get(k, 0)
            
            price = g('price_buy')
            
            ai_soc = g('ai_soc')
            ai_bat = g('ai_total_power')
            ai_grid = g('ai_grid_power')
            
            self_soc = g('self_soc')
            self_bat = g('self_total_power')
            self_grid = g('self_grid_power')

            csv_data += f"{time_str},{price:.2f},{ai_soc:.2f},{ai_bat:.0f},{ai_grid:.0f},{self_soc:.2f},{self_bat:.0f},{self_grid:.0f}\n"
            
        prompt = f"""
        Analyze the following 15-minute interval energy data.
        Divide the day (00:00 - 23:45) into 4 to 8 meaningful time segments based on battery behavior, grid interaction, and electricity price.
        Segments can start and end at any 15-minute mark (e.g., 12:30 - 14:15).
        
        Columns:
        - Price: Electricity buying price.
        - AI_SOC / Self_SOC: State of Charge (0-1).
        - AI_Bat_Power / Self_Bat_Power: Battery power (Positive = Discharging, Negative = Charging).
        - AI_Grid_Power / Self_Grid_Power: Grid power (Positive = Buying, Negative = Selling).

        Examples of period types: "charging_low_price", "discharging_high_price", "peak_shaving", "idle", "solar_charging".
        
        Data:
        {csv_data}
        
        Return a JSON list of objects with these fields:
        - start_time (HH:MM, e.g., "00:00")
        - end_time (HH:MM, e.g., "06:00")
        - period_type (string)
        
        Ensure the segments cover the entire day or the most important active periods.
        """
        
        response = self._call_api_with_retry(prompt, json_format=True)
        try:
            res_json = json.loads(response)
            if isinstance(res_json, list):
                return res_json
            elif 'segments' in res_json:
                return res_json['segments']
            else:
                return [{"start_time": "00:00", "end_time": "23:45", "period_type": "all_day"}]
        except:
            return [{"start_time": "00:00", "end_time": "23:45", "period_type": "all_day"}]

    def _calc_segment_stats(self, df, start_time, end_time):
        mask = (df.index.strftime('%H:%M') >= start_time) & (df.index.strftime('%H:%M') <= end_time)
        sub_df = df[mask]
        
        if sub_df.empty:
            return self._empty_stats()
            
        interval = 0.25
        
        def get_mode_stats(prefix):
            avg_pv_val = sub_df['pv_real'].mean()
            pv_to_bat = abs(sub_df[f'{prefix}battery_from_pv']).mean()
            
            bat_total = sub_df[f'{prefix}total_power']
            
            grid_col = f'{prefix}grid_power'
            if grid_col in sub_df.columns:
                grid_net = sub_df[grid_col]
            else:
                grid_net = sub_df['load_real'] - sub_df['pv_real'] - bat_total
            
            # Handle NaNs for empty slices or no matching conditions
            total_export = abs(grid_net[grid_net < 0]).mean()
            if pd.isna(total_export): total_export = 0
            
            bat_to_grid = abs(sub_df[f'{prefix}battery_to_grid']).mean()
            
            pv_to_grid_val = max(0, total_export - bat_to_grid)
            pv_to_load_val = max(0, avg_pv_val - pv_to_bat - pv_to_grid_val)
            
            buy_val = grid_net[grid_net > 0].mean()
            if pd.isna(buy_val): buy_val = 0
            
            sell_val = abs(grid_net[grid_net < 0]).mean()
            if pd.isna(sell_val): sell_val = 0
            
            start_soc = sub_df[f'{prefix}soc'].iloc[0]
            end_soc = sub_df[f'{prefix}soc'].iloc[-1]
            
            trend = "stable"
            if end_soc > start_soc + 0.05: trend = "increasing"
            elif end_soc < start_soc - 0.05: trend = "decreasing"
            
            return {
                "pv_action": {
                    "to_battery": round(pv_to_bat, 1),
                    "to_load": round(pv_to_load_val, 1),
                    "to_grid": round(pv_to_grid_val, 1)
                },
                "grid_action": {
                    "buy_energy": round(buy_val, 1),
                    "sell_energy": round(sell_val, 1)
                },
                "battery_soc": {
                    "start_soc": round(start_soc, 2),
                    "end_soc": round(end_soc, 2),
                    "trend": trend
                }
            }

        ai_stats = get_mode_stats('ai_')
        self_stats = get_mode_stats('self_')
        
        avg_pv = sub_df['pv_real'].mean()
        avg_load = sub_df['load_real'].mean()
        avg_price = sub_df['price_buy'].mean()
        
        pv_trend = "balanced"
        if avg_pv > avg_load * 1.2: pv_trend = "surplus"
        elif avg_pv < avg_load * 0.8: pv_trend = "deficit"
        
        price_trend = "stable"
        if sub_df['price_buy'].iloc[-1] > sub_df['price_buy'].iloc[0]: price_trend = "rising"
        elif sub_df['price_buy'].iloc[-1] < sub_df['price_buy'].iloc[0]: price_trend = "falling"
        if avg_price > df['price_buy'].mean() * 1.2: price_trend = "high"
        
        return {
            "pv_load": {
                "avg_pv": round(avg_pv, 1),
                "avg_load": round(avg_load, 1),
                "trend": pv_trend
            },
            "price": {
                "avg_value": round(avg_price, 2),
                "trend": price_trend
            },
            "ai_mode": ai_stats,
            "self_mode": self_stats
        }

    def _get_descriptions_from_api(self, seg, stats):
        prompt = f"""
        Generate qualitative descriptions for an energy system report for the time period {seg['start_time']} - {seg['end_time']} ({seg['period_type']}).
        
        Stats:
        PV/Load: Avg PV {stats['pv_load']['avg_pv']}W, Avg Load {stats['pv_load']['avg_load']}W, Trend: {stats['pv_load']['trend']}
        Price: Avg {stats['price']['avg_value']}, Trend: {stats['price']['trend']}
        
        AI Mode:
        - PV to Battery: {stats['ai_mode']['pv_action']['to_battery']}W
        - PV to Load: {stats['ai_mode']['pv_action']['to_load']}W
        - Grid Buy: {stats['ai_mode']['grid_action']['buy_energy']}W, Sell: {stats['ai_mode']['grid_action']['sell_energy']}W
        - SOC: {stats['ai_mode']['battery_soc']['start_soc']} -> {stats['ai_mode']['battery_soc']['end_soc']} ({stats['ai_mode']['battery_soc']['trend']})
        
        Self Mode:
        - PV to Battery: {stats['self_mode']['pv_action']['to_battery']}W
        - PV to Load: {stats['self_mode']['pv_action']['to_load']}W
        - Grid Buy: {stats['self_mode']['grid_action']['buy_energy']}W, Sell: {stats['self_mode']['grid_action']['sell_energy']}W
        - SOC: {stats['self_mode']['battery_soc']['start_soc']} -> {stats['self_mode']['battery_soc']['end_soc']} ({stats['self_mode']['battery_soc']['trend']})
        
        Return a JSON object with the following structure (descriptions in Chinese):
        {{
            "descriptions": {{
                "pv_load": {{ "description": "..." }},
                "price": {{ "description": "..." }}
            }},
            "ai_mode": {{
                "pv_action": {{ "description": "..." }},
                "grid_action": {{ "description": "..." }},
                "battery_soc": {{ "description": "..." }}
            }},
            "self_mode": {{
                "pv_action": {{ "description": "..." }},
                "grid_action": {{ "description": "..." }},
                "battery_soc": {{ "description": "..." }}
            }}
        }}
        Keep descriptions concise and professional.
        """
        
        response = self._call_api_with_retry(prompt, json_format=True)
        try:
            return json.loads(response)
        except:
            return self._empty_descriptions()

    def _empty_stats(self):
        return {
            "pv_load": {"avg_pv": 0, "avg_load": 0, "trend": "none"},
            "price": {"avg_value": 0, "trend": "none"},
            "ai_mode": {
                "pv_action": {"to_battery": 0, "to_load": 0, "to_grid": 0},
                "grid_action": {"buy_energy": 0, "sell_energy": 0},
                "battery_soc": {"start_soc": 0, "end_soc": 0, "trend": "none"}
            },
            "self_mode": {
                "pv_action": {"to_battery": 0, "to_load": 0, "to_grid": 0},
                "grid_action": {"buy_energy": 0, "sell_energy": 0},
                "battery_soc": {"start_soc": 0, "end_soc": 0, "trend": "none"}
            }
        }

    def _empty_descriptions(self):
        return {
            "descriptions": {
                "pv_load": { "description": "" },
                "price": { "description": "" }
            },
            "ai_mode": {
                "pv_action": { "description": "" },
                "grid_action": { "description": "" },
                "battery_soc": { "description": "" }
            },
            "self_mode": {
                "pv_action": { "description": "" },
                "grid_action": { "description": "" },
                "battery_soc": { "description": "" }
            }
        }

if __name__ == "__main__":
    transfer = Transfer()
    input_dir = "series"
    output_dir = "series_segments_with_api"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(input_dir):
        files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        print(f"Found {len(files)} files in {input_dir}")
        
        for filename in files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                print(f"Skipping {filename} (already processed)")
                continue
                
            try:
                transfer.process(input_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    else:
        print(f"Directory {input_dir} not found.")
