import os
import json
import re
import sys
from datetime import datetime

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api import ChatClient

def generate_series_data():
    input_dir = r"c:\Users\22638\Desktop\sungrow\prepare\AI模式欧洲典型电站分析20250717\2025-06-01_2025-06-30_5096060_empirical_result"
    output_dir = r"c:\Users\22638\Desktop\sungrow\prepare\series"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = ChatClient()
    
    # List all png files
    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    
    for file in files:
        # Extract date from filename: 5096060_2025-06-06_empirical_result_plot.png
        match = re.search(r"(\d{4}-\d{2}-\d{2})", file)
        if not match:
            print(f"Skipping {file}, no date found.")
            continue
            
        date_str = match.group(1)
        output_path = os.path.join(output_dir, f"{date_str}.json")
        
        if os.path.exists(output_path):
            print(f"Skipping {date_str}, already exists.")
            continue
            
        print(f"Processing {date_str} from {file}...")
        image_path = os.path.join(input_dir, file)
        
        prompt = f"""
        You are an expert data extractor. I will provide you with an image containing several charts showing the operation of a photovoltaic energy storage system for the date {date_str}.
        
        Please extract the time-series data from these charts for the entire day (00:00 to 23:45) at 15-minute intervals (96 points).
        
        The charts likely show:
        1. PV Power and Load Power.
        2. Electricity Price.
        3. Battery SOC (AI Mode and Self Mode).
        4. Battery Power and Grid Power.
        
        Return the data in CSV format with the following headers:
        Time,PV_Real,Load_Real,Price_Buy,Price_Sell,AI_SOC,AI_Bat_In,AI_Bat_Out,AI_Grid_In,AI_Grid_Out,Self_SOC,Self_Bat_In,Self_Bat_Out,Self_Grid_In,Self_Grid_Out
        
        Where:
        - Time: HH:MM (00:00, 00:15, ... 23:45)
        - PV_Real, Load_Real: Power in Watts (or kW, convert to Watts if needed, check axis).
        - Price_Buy, Price_Sell: Currency/kWh.
        - AI_SOC, Self_SOC: 0.0 to 1.0.
        - AI_Bat_In: Power charging the battery in AI mode (Watts).
        - AI_Bat_Out: Power discharging the battery in AI mode (Watts).
        - AI_Grid_In: Power bought from grid in AI mode (Watts).
        - AI_Grid_Out: Power sold to grid in AI mode (Watts).
        - Self_...: Same for Self mode.
        
        IMPORTANT:
        - Strictly 96 rows of data.
        - Estimate values from the curves.
        - Return ONLY the CSV data, no markdown, no explanations.
        """
        
        try:
            response = client.generate(prompt, image_path=image_path, json_format=False)
            
            # Clean up response
            csv_content = response
            if "```csv" in response:
                csv_content = response.split("```csv")[1].split("```")[0]
            elif "```" in response:
                csv_content = response.split("```")[1].split("```")[0]
            
            lines = csv_content.strip().split('\n')
            headers = lines[0].split(',')
            
            datasets = {
                "pv_load": [],
                "price": [],
                "ai_mode": [],
                "self_mode": [],
                "grid_power": []
            }
            
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) < 15: continue
                
                # Parse values (handle potential whitespace)
                vals = [p.strip() for p in parts]
                t = vals[0]
                
                try:
                    # Helper to safe float
                    def f(x): return float(x) if x else 0.0
                    
                    datasets["pv_load"].append({
                        "time": t,
                        "pv_forecast": 0, # Not in CSV, assume 0
                        "pv_real": f(vals[1]),
                        "load_forecast": 0,
                        "load_real": f(vals[2])
                    })
                    
                    datasets["price"].append({
                        "time": t,
                        "price_buy": f(vals[3]),
                        "price_sell": f(vals[4])
                    })
                    
                    # AI Mode
                    # Infer flow directions if needed, but here we asked for In/Out explicitly
                    datasets["ai_mode"].append({
                        "time": t,
                        "battery_from_pv": f(vals[6]), # Simplified assumption: Bat_In comes from PV? 
                        # Actually Bat_In could be from Grid too. 
                        # But for now let's map Bat_In to battery_from_pv and 0 to from_grid unless we have more info.
                        # Wait, the prompt asked for AI_Bat_In. 
                        # Let's assume AI_Bat_In is total charge. 
                        # We need to split it. 
                        # If Price is low, maybe from grid? 
                        # For simplicity, let's put it all in battery_from_pv for now, or split if we can.
                        # Actually, let's just map it directly for now.
                        "battery_from_pv": f(vals[6]), 
                        "battery_from_grid": 0, # Hard to distinguish without more logic
                        "battery_to_load": f(vals[7]),
                        "battery_to_grid": 0,
                        "soc": f(vals[5])
                    })
                    
                    datasets["self_mode"].append({
                        "time": t,
                        "soc": f(vals[10]),
                        "battery_from_pv": f(vals[11]),
                        "battery_to_load": f(vals[12]),
                        "battery_from_grid": 0,
                        "battery_to_grid": 0
                    })
                    
                    datasets["grid_power"].append({
                        "time": t,
                        "grid_power": f(vals[8]) - f(vals[9]) # Net grid? In - Out
                    })
                    
                except ValueError as e:
                    print(f"Error parsing line {line}: {e}")
                    continue

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

if __name__ == "__main__":
    generate_series_data()
