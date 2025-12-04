import json
import os

series_dir = 'd:\\Projects\\temp\\series'
if os.path.exists(series_dir):
    for filename in os.listdir(series_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(series_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                changed = False
                if 'datasets' in data:
                    datasets = data['datasets']
                    
                    if 'ai_mode' in datasets:
                        for item in datasets['ai_mode']:
                            if 'ai_total_power' not in item:
                                to_load = item.get('battery_to_load', 0)
                                to_grid = item.get('battery_to_grid', 0)
                                from_pv = item.get('battery_from_pv', 0)
                                from_grid = item.get('battery_from_grid', 0)
                                item['ai_total_power'] = (to_load + to_grid) - (from_pv + from_grid)
                                changed = True
                    
                    if 'self_mode' in datasets:
                        for item in datasets['self_mode']:
                            if 'self_total_power' not in item:
                                to_load = item.get('battery_to_load', 0)
                                to_grid = item.get('battery_to_grid', 0)
                                from_pv = item.get('battery_from_pv', 0)
                                from_grid = item.get('battery_from_grid', 0)
                                item['self_total_power'] = (to_load + to_grid) - (from_pv + from_grid)
                                changed = True
                
                if changed:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"Updated {filename}")
                else:
                    print(f"No changes needed for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
else:
    print(f"Directory {series_dir} not found")
