import os
import json
import re
import sys
import time
from datetime import datetime

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api import ChatClient

def generate_series_data():
    slices_dir = r"c:\Users\22638\Desktop\sungrow\prepare\AI模式欧洲典型电站分析20250717\slices"
    output_dir = r"c:\Users\22638\Desktop\sungrow\prepare\series"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = ChatClient()
    
    # Get all files in slices dir
    all_files = os.listdir(slices_dir)
    
    # Group by date
    # Filename format: 5096060_2025-06-12_empirical_result_plot_slice_1.png
    date_files = {}
    for f in all_files:
        if not f.endswith(".png"): continue
        match = re.search(r"(\d{4}-\d{2}-\d{2})", f)
        if match:
            date_str = match.group(1)
            if date_str not in date_files:
                date_files[date_str] = {}
            
            # Determine slice number
            if "slice_1" in f: date_files[date_str][1] = f
            elif "slice_2" in f: date_files[date_str][2] = f
            elif "slice_3" in f: date_files[date_str][3] = f
            elif "slice_4" in f: date_files[date_str][4] = f
            elif "slice_5" in f: date_files[date_str][5] = f

    for date_str, slices in date_files.items():
        output_path = os.path.join(output_dir, f"{date_str}.json")
        if os.path.exists(output_path):
            print(f"Skipping {date_str}, already exists.")
            continue
            
        if len(slices) != 5:
            print(f"Skipping {date_str}, incomplete slices (found {len(slices)}).")
            continue

        print(f"Processing {date_str}...")
        
        # Data storage for this date
        # We will store by time key to merge later
        merged_data = {} # "00:00": { ... }
        
        # Initialize time keys
        for h in range(24):
            for m in [0, 15, 30, 45]:
                t_str = f"{h:02d}:{m:02d}"
                merged_data[t_str] = {}

        # Process each slice
        try:
            # Slice 1: PV & Load
            process_slice(client, os.path.join(slices_dir, slices[1]), date_str, 1, merged_data)

            # Slice 2: Price
            process_slice(client, os.path.join(slices_dir, slices[2]), date_str, 2, merged_data)
            
            # Slice 3: AI Mode
            process_slice(client, os.path.join(slices_dir, slices[3]), date_str, 3, merged_data)
            
            # Slice 4: Self Mode
            process_slice(client, os.path.join(slices_dir, slices[4]), date_str, 4, merged_data)
            
            # Slice 5: Grid Power
            process_slice(client, os.path.join(slices_dir, slices[5]), date_str, 5, merged_data)
            
            # Construct final JSON
            datasets = {
                "pv_load": [],
                "price": [],
                "ai_mode": [],
                "self_mode": [],
                "grid_power": []
            }
            
            sorted_times = sorted(merged_data.keys())
            
            for t in sorted_times:
                d = merged_data[t]
                
                # Helper to get value safely
                def g(k): return d.get(k, 0.0)
                
                datasets["pv_load"].append({
                    "time": t,
                    "pv_forecast": g("PV_Forecast"), 
                    "pv_real": g("PV_Real"),
                    "load_forecast": g("Load_Forecast"),
                    "load_real": g("Load_Real")
                })
                
                datasets["price"].append({
                    "time": t,
                    "price_buy": g("Price_Buy"),
                    "price_sell": g("Price_Sell")
                })
                
                # AI Mode
                ai_soc = g("AI_SOC")
                ai_bat_from_pv = g("AI_Bat_From_PV")
                ai_bat_from_grid = g("AI_Bat_From_Grid")
                ai_bat_to_load = g("AI_Bat_To_Load")
                ai_bat_to_grid = g("AI_Bat_To_Grid")
                ai_total_power = (ai_bat_to_load + ai_bat_to_grid) - (ai_bat_from_pv + ai_bat_from_grid)

                datasets["ai_mode"].append({
                    "time": t,
                    "soc": ai_soc,
                    "battery_from_pv": ai_bat_from_pv,
                    "battery_from_grid": ai_bat_from_grid,
                    "battery_to_load": ai_bat_to_load,
                    "battery_to_grid": ai_bat_to_grid,
                    "ai_total_power": ai_total_power
                })
                
                # Self Mode
                self_soc = g("Self_SOC")
                self_bat_from_pv = g("Self_Bat_From_PV")
                self_bat_from_grid = g("Self_Bat_From_Grid")
                self_bat_to_load = g("Self_Bat_To_Load")
                self_bat_to_grid = g("Self_Bat_To_Grid")
                self_total_power = (self_bat_to_load + self_bat_to_grid) - (self_bat_from_pv + self_bat_from_grid)

                datasets["self_mode"].append({
                    "time": t,
                    "soc": self_soc,
                    "battery_from_pv": self_bat_from_pv,
                    "battery_from_grid": self_bat_from_grid,
                    "battery_to_load": self_bat_to_load,
                    "battery_to_grid": self_bat_to_grid,
                    "self_total_power": self_total_power
                })
                
                datasets["grid_power"].append({
                    "time": t,
                    "ai_grid_power": g("AI_Grid_Power"),
                    "self_grid_power": g("Self_Grid_Power")
                })

            final_json = {
                "ps_id": "5096060",
                "date": date_str,
                "datasets": datasets
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
            print(f"Saved {output_path}")

        except Exception as e:
            print(f"Error processing {date_str}: {e}")

def process_slice(client, image_path, date_str, slice_num, merged_data):
    # Define prompts based on slice_num
    if slice_num == 1:
        prompt = f"""
        You are an expert data extractor. Extract time-series data from this PV & Load Power chart for date {date_str}.
        Time range: 00:00 to 23:45 (15-min intervals, 96 points).
        
        Lines:
        - PV Real Power: Dim blue line with GREEN area.
        - Load Real Power: Red line with ORANGE area.
        - PV Forecast: Green line.
        - Load Forecast: Pink line.
        
        Return CSV with headers: Time,PV_Real,PV_Forecast,Load_Real,Load_Forecast
        Values in Watts (W).
        """
        headers = ["Time", "PV_Real", "PV_Forecast", "Load_Real", "Load_Forecast"]
        
    elif slice_num == 2:
        prompt = f"""
        You are an expert data extractor. Extract time-series data from this Electricity Price chart for date {date_str}.
        Time range: 00:00 to 23:45 (15-min intervals, 96 points).
        
        Lines:
        - Buying Price: Orange line.
        - Selling Price: Dim blue line.
        
        Return CSV with headers: Time,Price_Buy,Price_Sell
        Values in Currency/kWh.
        """
        headers = ["Time", "Price_Buy", "Price_Sell"]
        
    elif slice_num == 3:
        prompt = f"""
        You are an expert data extractor. Extract time-series data from this AI Mode Battery Actions chart for date {date_str}.
        Time range: 00:00 to 23:45 (15-min intervals, 96 points).
        
        - SOC: Red curve (0.0 to 1.0).
        - Battery Charging from PV: GREEN area BELOW axis.
        - Battery Charging from Grid: YELLOW area BELOW axis.
        - Battery Discharging to Load: ORANGE area ABOVE axis.
        - Battery Discharging to Grid: BLUE area ABOVE axis.
        
        Return CSV with headers: Time,AI_SOC,AI_Bat_From_PV,AI_Bat_From_Grid,AI_Bat_To_Load,AI_Bat_To_Grid
        Power values in Watts (W). SOC is 0.0-1.0.
        """
        headers = ["Time", "AI_SOC", "AI_Bat_From_PV", "AI_Bat_From_Grid", "AI_Bat_To_Load", "AI_Bat_To_Grid"]

    elif slice_num == 4:
        prompt = f"""
        You are an expert data extractor. Extract time-series data from this Self Mode Battery Actions chart for date {date_str}.
        Time range: 00:00 to 23:45 (15-min intervals, 96 points).
        
        - SOC: Red curve (0.0 to 1.0).
        - Battery Charging from PV: GREEN area BELOW axis.
        - Battery Charging from Grid: YELLOW area BELOW axis.
        - Battery Discharging to Load: ORANGE area ABOVE axis.
        - Battery Discharging to Grid: BLUE area ABOVE axis.
        
        Return CSV with headers: Time,Self_SOC,Self_Bat_From_PV,Self_Bat_From_Grid,Self_Bat_To_Load,Self_Bat_To_Grid
        Power values in Watts (W). SOC is 0.0-1.0.
        """
        headers = ["Time", "Self_SOC", "Self_Bat_From_PV", "Self_Bat_From_Grid", "Self_Bat_To_Load", "Self_Bat_To_Grid"]

    elif slice_num == 5:
        prompt = f"""
        You are an expert data extractor. Extract time-series data from this Grid Power Actions chart for date {date_str}.
        Time range: 00:00 to 23:45 (15-min intervals, 96 points).
        
        - AI Mode Grid Power: ORANGE area/line.
        - Self Mode Grid Power: GREEN area/line.
        
        Coordinate System:
        - Values ABOVE the axis (Positive) represent BUYING from Grid (Import).
        - Values BELOW the axis (Negative) represent SELLING to Grid (Export).
        
        Return CSV with headers: Time,AI_Grid_Power,Self_Grid_Power
        Values in Watts (W).
        """
        headers = ["Time", "AI_Grid_Power", "Self_Grid_Power"]

    # Call API with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.generate(prompt, image_path=image_path, json_format=False)
            
            # Parse CSV
            csv_content = response
            if "```csv" in response:
                csv_content = response.split("```csv")[1].split("```")[0]
            elif "```" in response:
                csv_content = response.split("```")[1].split("```")[0]
            
            lines = csv_content.strip().split('\n')
            # Skip header
            if "Time" in lines[0]:
                lines = lines[1:]
                
            for line in lines:
                parts = line.split(',')
                if len(parts) < len(headers): continue
                
                vals = [p.strip() for p in parts]
                t = vals[0]
                
                # Normalize time format if needed (e.g. 0:00 -> 00:00)
                if len(t) == 4 and t[1] == ':': t = '0' + t
                
                if t in merged_data:
                    for i in range(1, len(headers)):
                        key = headers[i]
                        try:
                            val = float(vals[i]) if vals[i] else 0.0
                        except:
                            val = 0.0
                        merged_data[t][key] = val
            
            print(f"  Slice {slice_num} processed.")
            return
            
        except Exception as e:
            print(f"  Error slice {slice_num} attempt {attempt}: {e}")
            time.sleep(5)
            
    raise Exception(f"Failed to process slice {slice_num}")

if __name__ == "__main__":
    generate_series_data()
