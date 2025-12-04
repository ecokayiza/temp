
## 目录结构

*   `./format/`: 数据格式定义参考
    *   `data_format.json`: 原始时间序列数据格式
    *   `input_format.json`: 模型训练输入格式（已废弃，参考 `series_segments`）
    *   `rule_format.json`: 模型输出规则格式
*   `./series/`: 每天的原始运行数据 (JSON)
*   `./series_segments/`: **[核心]** 经过逻辑切分后的分段数据，作为模型的直接输入
*   `./series_graph/`: 可视化图表（包含原始曲线和分段标记）
*   `./rules/`: 规则库定义
*   `./utils/`: 工具脚本

    *   `api.py`: LLM API 调用工具
    *   `slice_graph` 、 `gen_proba_data.py`: 获取参考数据
    *   `slice_day.py`: **[核心]** 数据分段与特征提取脚本(使用状态向量)
    *   `draw.py`: 数据可视化脚本

## 数据处理流程

1.  **原始数据准备** (`./series/`)
    *   包含每 15 分钟的光伏、负载、电价以及 AI/Self 两种模式的运行数据。

2.  **逻辑分段与特征提取** (`./utils/slice_day.py`)
    *   **切分逻辑**：基于“状态向量”变化进行切分。状态包括：
        *   电价状态 (负电价/低谷/平价/高峰/尖峰)
        *   电池动作 (充电/放电/闲置)
        *   电网交互 (买电/卖电)
        *   光伏状态 (无光/光<负/光>负) 等
    *   **输出产物**：在 `./series_segments/` 下生成以段为单位的 JSON 文件（如 `2025-06-12_05.json`）。

3.  **可视化验证** (`./utils/draw.py`)
    *   读取原始数据和分段信息，生成组合图表，直观展示策略差异。

## 模型输入格式 (Segment Data Format)

每个分段文件包含以下核心字段：

| 字段名称 (Field Hierarchy) | 类型 | 示例值 | 描述 |
| :--- | :--- | :--- | :--- |
| **metadata** | Object | - | **元数据信息** |
| &nbsp;&nbsp;&nbsp;&nbsp;`ps_id` | String | "5096060" | 电站/用户唯一标识 ID |
| &nbsp;&nbsp;&nbsp;&nbsp;`date` | String | "2025-06-12" | 数据所属日期 (YYYY-MM-DD) |
| &nbsp;&nbsp;&nbsp;&nbsp;`seg_id` | String | "01" | 分段序号，标识当天的第几个逻辑片段 |
| **context** | Object | - | **环境上下文状态** |
| &nbsp;&nbsp;&nbsp;&nbsp;`time_range` | String | "00:00 - 00:45" | 该分段的起止时间 |
| &nbsp;&nbsp;&nbsp;&nbsp;`duration_minutes` | Number | 60 | 分段持续时长（分钟） |
| &nbsp;&nbsp;&nbsp;&nbsp;`price_state` | String | "flat" | 电价状态：`negative`, `low`, `flat`, `high`, `peak` |
| &nbsp;&nbsp;&nbsp;&nbsp;`avg_price` | Number | 0.625 | 该时段内的平均购电价格 (元/kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;`pv_state` | String | "no_sun" | 光伏状态：`no_sun`, `sun_less_load`, `sun_more_load` |
| &nbsp;&nbsp;&nbsp;&nbsp;`total_load_kwh` | Number | 1.62 | 累计负载消耗电量 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;`avg_load_kw` | Number | 1.62 | 平均负载功率 (kW) |
| &nbsp;&nbsp;&nbsp;&nbsp;`total_pv_kwh` | Number | 0.0 | 累计光伏发电量 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;`avg_pv_kw` | Number | 0.0 | 平均光伏功率 (kW) |
| **action** | Object | - | **系统行为数据** |
| &nbsp;&nbsp;&nbsp;&nbsp;**ai_mode** | Object | - | **AI 策略模式下的详细数据** |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**battery** | Object | - | 电池行为详情 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`soc_start` | Number | 0.56 | 分段开始时的 SOC (0.0-1.0) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`soc_end` | Number | 0.52 | 分段结束时的 SOC (0.0-1.0) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`charge_kwh` | Number | 0.0 | 累计充电电量 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`discharge_kwh` | Number | 1.35 | 累计放电电量 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`total_kwh` | Number | 1.35 | 净吞吐量 (kWh)，正为放电，负为充电 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`avg_kw` | Number | 1.35 | 净平均功率 (kW)，正为放电，负为充电 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**grid** | Object | - | 电网交互详情 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`buy_kwh` | Number | 0.31 | 累计买电量 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sell_kwh` | Number | 0.0 | 累计卖电量 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`total_kwh` | Number | 0.31 | 净交互量 (kWh)，正为买电，负为卖电 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`avg_kw` | Number | 0.31 | 净平均功率 (kW)，正为买电，负为卖电 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**pv** | Object | - | 光伏能量流向详情 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`to_battery_kwh` | Number | 0.0 | 光伏 -> 电池 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`to_load_kwh` | Number | 0.0 | 光伏 -> 负载 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`to_grid_kwh` | Number | 0.0 | 光伏 -> 电网 (kWh) |
| &nbsp;&nbsp;&nbsp;&nbsp;**self_mode** | Object | - | **传统/自用模式下的详细数据** |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*(结构同 ai_mode)* | ... | ... | 包含 battery, grid, pv 三个子对象，结构完全一致 |