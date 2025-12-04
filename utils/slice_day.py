import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================
class Config:
    INPUT_DIR = './series'
    OUTPUT_DIR = './series_segments'
    
    # Smoothing Constraints
    TARGET_MAX_SEGMENTS = 7
    TARGET_MIN_SEGMENTS = 4
    
    # Time Definitions (HH:MM)
    DAY_START = "06:00"
    DAY_END = "17:00"
    
    # Minimum Duration (in 15-min points)
    MIN_DURATION_DAY = 2      # Allow shorter segments during day
    MIN_DURATION_NIGHT = 10    # Force longer segments during night

    # Weights for State Distance (Vector Comparison)
    # Assuming units: Price(EU), Power (kW)
    # Typical Price range: 0.0 - 0.9 
    # Typical Power range: 0 - 8 kW. 
    
    W_PRICE = 4.0       
    W_BATTERY = 5.0    
    W_GRID = 3.0       
    W_PV = 3.0        
    
    # Initial Segmentation Threshold
    # Distance > Threshold -> New Segment
    SEGMENTATION_THRESHOLD = 5.0

# =============================================================================
# Data Loading & Preparation
# =============================================================================
class DataLoader:
    @staticmethod
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def prepare_dataframe(data):
        datasets = data.get('datasets', {})
        
        def to_df(data_list):
            if not data_list: return pd.DataFrame()
            return pd.DataFrame(data_list)

        df_pv = to_df(datasets.get('pv_load', []))
        df_price = to_df(datasets.get('price', []))
        df_ai = to_df(datasets.get('ai_mode', []))
        df_self = to_df(datasets.get('self_mode', []))
        df_grid = to_df(datasets.get('grid_power', []))

        def process_df(df, prefix=''):
            if df.empty or 'time' not in df.columns: return None
            df = df.set_index('time')
            if prefix:
                df = df.add_prefix(prefix)
                # Fix double prefix if exists
                df.columns = [c.replace(f'{prefix}{prefix}', prefix) for c in df.columns]
            return df

        df_pv = process_df(df_pv)
        df_price = process_df(df_price)
        df_ai = process_df(df_ai, 'ai_')
        df_self = process_df(df_self, 'self_')
        df_grid = process_df(df_grid)

        dfs = [d for d in [df_pv, df_price, df_ai, df_self, df_grid] if d is not None]
        
        if not dfs:
            return pd.DataFrame()
            
        df_final = pd.concat(dfs, axis=1)
        df_final = df_final.fillna(0)
        
        # Ensure numeric types
        cols = df_final.columns
        df_final[cols] = df_final[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Calculate total power if missing
        for mode in ['ai', 'self']:
            prefix = f"{mode}_"
            if f'{prefix}total_power' not in df_final.columns:
                 try:
                     df_final[f'{prefix}total_power'] = (
                        df_final.get(f'{prefix}battery_to_load', 0) + 
                        df_final.get(f'{prefix}battery_to_grid', 0)
                    ) - (
                        df_final.get(f'{prefix}battery_from_pv', 0) + 
                        df_final.get(f'{prefix}battery_from_grid', 0)
                    )
                 except:
                     df_final[f'{prefix}total_power'] = 0

        return df_final

# =============================================================================
# Segmentation Logic
# =============================================================================
class Segmenter:
    def __init__(self, df):
        self.df = df
        self.thresholds = self._calculate_price_thresholds(df)

    def _calculate_price_thresholds(self, df):
        prices = df['price_buy']
        positive_prices = prices[prices >= 0]
        
        if positive_prices.empty:
            return {'low': 0, 'flat': 0, 'high': 0}
            
        q1 = positive_prices.quantile(0.25)
        q3 = positive_prices.quantile(0.75)
        p90 = positive_prices.quantile(0.90)
        
        return {
            'low': q1,
            'flat': q3,
            'high': p90
        }

    def _get_state(self, row):
        # 1. Price State
        price = row['price_buy']
        if price < 0:
            price_state = 'negative'
        elif price <= self.thresholds['low']:
            price_state = 'low'
        elif price <= self.thresholds['flat']:
            price_state = 'flat'
        elif price <= self.thresholds['high']:
            price_state = 'high'
        else:
            price_state = 'peak'

        # Helper for Power Levels (Quantization)
        # 0: Idle (<10W)
        # 1: Low (<1kW)
        # 2: Medium (<3kW)
        # 3: High (>=3kW)
        def get_power_level(power):
            p_abs = abs(power)
            sign = 1 if power > 0 else -1
            if p_abs < 10: return 0
            if p_abs < 1000: return 1 * sign
            if p_abs < 3000: return 2 * sign
            return 3 * sign

        # 2. Battery Action (AI & Self)
        bat_ai_level = get_power_level(row.get('ai_total_power', 0))
        bat_self_level = get_power_level(row.get('self_total_power', 0))

        # 3. Grid Interaction (AI & Self)
        grid_ai_level = get_power_level(row.get('ai_grid_power', 0))
        grid_self_level = get_power_level(row.get('self_grid_power', 0))

        # 4. PV State
        pv = row.get('pv_real', 0)
        load = row.get('load_real', 0)
        
        if pv <= 10:
            pv_state = 'no_sun'
        elif pv < load:
            pv_state = 'sun_less_load'
        else:
            pv_state = 'sun_more_load'

        return (price_state, bat_ai_level, bat_self_level, grid_ai_level, grid_self_level, pv_state)

    def _get_vector(self, row):
        # Map categorical to numeric
        # Price
        price = row['price_buy']
        if price < 0: p_val = 0
        elif price <= self.thresholds['low']: p_val = 1
        elif price <= self.thresholds['flat']: p_val = 2
        elif price <= self.thresholds['high']: p_val = 3
        else: p_val = 4
        
        # PV
        pv = row.get('pv_real', 0)
        load = row.get('load_real', 0)
        if pv <= 10: pv_val = 0
        elif pv < load: pv_val = 1
        else: pv_val = 2
        
        # Power (kW)
        v_bat_ai = row.get('ai_total_power', 0) / 1000.0
        v_bat_self = row.get('self_total_power', 0) / 1000.0
        v_grid_ai = row.get('ai_grid_power', 0) / 1000.0
        v_grid_self = row.get('self_grid_power', 0) / 1000.0
        
        return np.array([p_val, v_bat_ai, v_bat_self, v_grid_ai, v_grid_self, pv_val])

    def _calculate_distance(self, v1, v2):
        weights = np.array([
            Config.W_PRICE, 
            Config.W_BATTERY, 
            Config.W_BATTERY, 
            Config.W_GRID, 
            Config.W_GRID, 
            Config.W_PV
        ])
        return np.sqrt(np.sum(weights * (v1 - v2)**2))

    def initial_segmentation(self):
        segments = []
        current_segment = []
        
        segment_start_vector = None
        segment_start_state = None
        
        for time, row in self.df.iterrows():
            current_vector = self._get_vector(row)
            current_state_label = self._get_state(row)
            
            if not current_segment:
                current_segment.append((time, row))
                segment_start_vector = current_vector
                segment_start_state = current_state_label
            else:
                dist = self._calculate_distance(segment_start_vector, current_vector)
                
                if dist > Config.SEGMENTATION_THRESHOLD:
                    # Break segment
                    segments.append({
                        'state': segment_start_state,
                        'data': current_segment
                    })
                    # Start new
                    current_segment = [(time, row)]
                    segment_start_vector = current_vector
                    segment_start_state = current_state_label
                else:
                    current_segment.append((time, row))
            
        # Add last segment
        if current_segment:
            segments.append({
                'state': segment_start_state,
                'data': current_segment
            })
            
        return segments

# =============================================================================
# Smoothing Logic (Time-Aware)
# =============================================================================
class Smoother:
    @staticmethod
    def is_daytime(time_str):
        # time_str format: "HH:MM" or "YYYY-MM-DD HH:MM:SS"
        # We only care about HH:MM
        try:
            if ' ' in time_str:
                t = time_str.split(' ')[1][:5]
            else:
                t = time_str[:5]
            return Config.DAY_START <= t <= Config.DAY_END
        except:
            return False

    @staticmethod
    def get_segment_weight(segment):
        """
        Calculate 'effective length' for merging priority.
        Night segments appear shorter (easier to merge).
        Day segments appear longer (harder to merge).
        """
        data = segment['data']
        start_time = data[0][0]
        length = len(data)
        
        if Smoother.is_daytime(start_time):
            # Day: Multiply length to resist merging
            return length * 2.0
        else:
            # Night: Divide length to encourage merging
            return length * 0.5

    @staticmethod
    def get_state_diff(segment1, segment2):
        # Calculate average vectors for each segment
        def get_avg_vector(segment):
            data = segment['data']
            df = pd.DataFrame([x[1] for x in data])
            
            # Price (Categorical -> Numeric approx)
            # We use the state value from the segment definition for price/pv as they are stable
            # But for power, we calculate the actual average from data
            
            state = segment['state']
            map_price = {'negative':0, 'low':1, 'flat':2, 'high':3, 'peak':4}
            map_pv = {'no_sun':0, 'sun_less_load':1, 'sun_more_load':2}
            
            v_price = map_price.get(state[0], 0)
            v_pv = map_pv.get(state[5], 0)
            
            # Power values (kW)
            # Note: data is in Watts usually, need to check. 
            # Based on previous code: 'ai_total_power' > 10 is discharge.
            # Let's assume Watts and convert to kW for vector
            
            def get_avg_kw(col):
                if col in df.columns:
                    return df[col].mean() / 1000.0
                return 0.0
                
            v_bat_ai = get_avg_kw('ai_total_power')
            v_bat_self = get_avg_kw('self_total_power')
            v_grid_ai = get_avg_kw('ai_grid_power')
            v_grid_self = get_avg_kw('self_grid_power')
            
            return np.array([v_price, v_bat_ai, v_bat_self, v_grid_ai, v_grid_self, v_pv])

        v1 = get_avg_vector(segment1)
        v2 = get_avg_vector(segment2)
        
        # Weights vector
        weights = np.array([
            Config.W_PRICE, 
            Config.W_BATTERY, 
            Config.W_BATTERY, 
            Config.W_GRID, 
            Config.W_GRID, 
            Config.W_PV
        ])
        
        # Weighted Euclidean Distance
        diff = np.sqrt(np.sum(weights * (v1 - v2)**2))
        
        return diff

    @staticmethod
    def smooth(segments):
        if not segments: return []
        initial_count = len(segments)
        current_segments = segments[:]
        
        while True:
            if len(current_segments) <= Config.TARGET_MIN_SEGMENTS:
                break
                
            # Find segment with minimum 'effective' length
            min_effective_len = float('inf')
            min_idx = -1
            min_real_len = 0
            
            for i, seg in enumerate(current_segments):
                eff_len = Smoother.get_segment_weight(seg)
                if eff_len < min_effective_len:
                    min_effective_len = eff_len
                    min_idx = i
                    min_real_len = len(seg['data'])
            
            # Check termination conditions
            # If we are within max limit, AND the smallest segment satisfies its time-based min duration
            is_day = Smoother.is_daytime(current_segments[min_idx]['data'][0][0])
            min_required = Config.MIN_DURATION_DAY if is_day else Config.MIN_DURATION_NIGHT
            
            if len(current_segments) <= Config.TARGET_MAX_SEGMENTS and min_real_len >= min_required:
                break
                
            # Merge Logic
            merge_idx = -1 
            
            if min_idx == 0:
                merge_idx = 1
            elif min_idx == len(current_segments) - 1:
                merge_idx = len(current_segments) - 2
            else:
                prev_seg = current_segments[min_idx - 1]
                next_seg = current_segments[min_idx + 1]
                curr_seg = current_segments[min_idx]
                
                diff_prev = Smoother.get_state_diff(curr_seg, prev_seg)
                diff_next = Smoother.get_state_diff(curr_seg, next_seg)
                
                if diff_prev < diff_next:
                    merge_idx = min_idx - 1
                elif diff_next < diff_prev:
                    merge_idx = min_idx + 1
                else:
                    # Tie-break: merge into the shorter neighbor (by effective length)
                    eff_prev = Smoother.get_segment_weight(prev_seg)
                    eff_next = Smoother.get_segment_weight(next_seg)
                    if eff_prev < eff_next:
                        merge_idx = min_idx - 1
                    else:
                        merge_idx = min_idx + 1
            
            # Perform merge
            target = current_segments[merge_idx]
            source = current_segments[min_idx]
            
            if merge_idx < min_idx:
                target['data'].extend(source['data'])
            else:
                target['data'] = source['data'] + target['data']
                
            current_segments.pop(min_idx)
        print(f"{initial_count} -> {len(current_segments)} segments.")   
        return current_segments

# =============================================================================
# Output Generation
# =============================================================================
class OutputGenerator:
    @staticmethod
    def calculate_pv_flow(df_seg, mode_prefix):
        FACTOR = 0.25 / 1000 
        pv_total = df_seg['pv_real']
        load_total = df_seg['load_real']
        
        def get_col(name):
            if name in df_seg.columns: return df_seg[name]
            return pd.Series(0.0, index=df_seg.index)

        bat_from_pv = get_col(f'{mode_prefix}battery_from_pv')
        bat_to_load = get_col(f'{mode_prefix}battery_to_load')
        
        pv_to_bat = bat_from_pv
        pv_avail = (pv_total - pv_to_bat).clip(lower=0)
        load_remaining = (load_total - bat_to_load).clip(lower=0)
        pv_to_load = np.minimum(pv_avail, load_remaining)
        pv_to_grid = pv_avail - pv_to_load
        
        return {
            "to_battery_kwh": round(pv_to_bat.sum() * FACTOR, 2),
            "to_load_kwh": round(pv_to_load.sum() * FACTOR, 2),
            "to_grid_kwh": round(pv_to_grid.sum() * FACTOR, 2)
        }

    @staticmethod
    def generate_description(segment, ps_id, date, seg_id):
        data_points = segment['data']
        df_seg = pd.DataFrame([x[1] for x in data_points])
        start_time = data_points[0][0]
        end_time = data_points[-1][0] 
        
        duration_hours = len(data_points) * 0.25
        avg_price = df_seg['price_buy'].mean()
        
        total_load = df_seg['load_real'].sum() * 0.25 / 1000
        total_pv = df_seg['pv_real'].sum() * 0.25 / 1000
        
        def to_avg_kw(kwh_val):
            return round(kwh_val / duration_hours, 2) if duration_hours > 0 else 0

        # AI Actions
        ai_charge = df_seg[df_seg['ai_total_power'] < 0]['ai_total_power'].sum() * 0.25 / 1000
        ai_discharge = df_seg[df_seg['ai_total_power'] > 0]['ai_total_power'].sum() * 0.25 / 1000
        ai_grid_buy = df_seg[df_seg['ai_grid_power'] > 0]['ai_grid_power'].sum() * 0.25 / 1000
        ai_grid_sell = df_seg[df_seg['ai_grid_power'] < 0]['ai_grid_power'].sum() * 0.25 / 1000
        
        # Self Actions
        self_charge = df_seg[df_seg['self_total_power'] < 0]['self_total_power'].sum() * 0.25 / 1000
        self_discharge = df_seg[df_seg['self_total_power'] > 0]['self_total_power'].sum() * 0.25 / 1000
        self_grid_buy = df_seg[df_seg['self_grid_power'] > 0]['self_grid_power'].sum() * 0.25 / 1000
        self_grid_sell = df_seg[df_seg['self_grid_power'] < 0]['self_grid_power'].sum() * 0.25 / 1000
        
        state = segment['state']
        
        ai_pv = OutputGenerator.calculate_pv_flow(df_seg, 'ai_')
        self_pv = OutputGenerator.calculate_pv_flow(df_seg, 'self_')

        # SOC
        ai_soc_start = df_seg['ai_soc'].iloc[0] if 'ai_soc' in df_seg.columns else 0
        ai_soc_end = df_seg['ai_soc'].iloc[-1] if 'ai_soc' in df_seg.columns else 0
        self_soc_start = df_seg['self_soc'].iloc[0] if 'self_soc' in df_seg.columns else 0
        self_soc_end = df_seg['self_soc'].iloc[-1] if 'self_soc' in df_seg.columns else 0

        return {
            "metadata": {
                "ps_id": ps_id,
                "date": date,
                "seg_id": seg_id
            },
            "context": {
                "time_range": f"{start_time} - {end_time}",
                "duration_minutes": len(data_points) * 15,
                "price_state": state[0],
                "avg_price": round(avg_price, 4),
                "pv_state": state[5], # Index 5 is PV in new state tuple
                "total_load_kwh": round(total_load, 2),
                "avg_load_kw": to_avg_kw(total_load),
                "total_pv_kwh": round(total_pv, 2),
                "avg_pv_kw": to_avg_kw(total_pv)
            },
            "action": {
                "ai_mode": {
                    "battery": {
                        "soc_start": round(ai_soc_start, 2),
                        "soc_end": round(ai_soc_end, 2),
                        "charge_kwh": round(abs(ai_charge), 2),
                        "discharge_kwh": round(ai_discharge, 2),
                        "total_kwh": round(ai_charge + ai_discharge, 2),
                        "avg_kw": to_avg_kw(ai_charge + ai_discharge)
                    },
                    "grid": {
                        "buy_kwh": round(ai_grid_buy, 2),
                        "sell_kwh": round(abs(ai_grid_sell), 2),
                        "total_kwh": round(ai_grid_buy + ai_grid_sell, 2),
                        "avg_kw": to_avg_kw(ai_grid_buy + ai_grid_sell)
                    },
                    "pv": ai_pv
                },
                "self_mode": {
                    "battery": {
                        "soc_start": round(self_soc_start, 2),
                        "soc_end": round(self_soc_end, 2),
                        "charge_kwh": round(abs(self_charge), 2),
                        "discharge_kwh": round(self_discharge, 2),
                        "total_kwh": round(self_charge + self_discharge, 2),
                        "avg_kw": to_avg_kw(self_charge + self_discharge)
                    },
                    "grid": {
                        "buy_kwh": round(self_grid_buy, 2),
                        "sell_kwh": round(abs(self_grid_sell), 2),
                        "total_kwh": round(self_grid_buy + self_grid_sell, 2),
                        "avg_kw": to_avg_kw(self_grid_buy + self_grid_sell)
                    },
                    "pv": self_pv
                }
            }
        }

# =============================================================================
# Main Execution
# =============================================================================
def main():
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        
    files = [f for f in os.listdir(Config.INPUT_DIR) if f.endswith('.json')]
    
    for file in files:
        path = os.path.join(Config.INPUT_DIR, file)
        print(f"Processing {file}...",end=' ')
        
        try:
            data = DataLoader.load_json(path)
            df = DataLoader.prepare_dataframe(data)
            
            if df.empty:
                print(f"Skipping {file}: Empty dataframe")
                continue
                
            segmenter = Segmenter(df)
            raw_segments = segmenter.initial_segmentation()
            segments = Smoother.smooth(raw_segments)
            
            ps_id = data.get('ps_id', 'unknown')
            date = data.get('date', file.replace('.json', ''))
            
            for i, seg in enumerate(segments):
                seg_id = f"{i+1:02d}"
                desc = OutputGenerator.generate_description(seg, ps_id, date, seg_id)
                
                out_name = f"{date}_{seg_id}.json"
                DataLoader.save_json(desc, os.path.join(Config.OUTPUT_DIR, out_name))
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    import shutil
    import os
    import draw
    import PIL.Image
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    main()
    # Call the draw function to show the first graph
    draw.main()
    image = PIL.Image.open('./series_graph/2025-06-12.png')
    image.show()


    
