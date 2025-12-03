### 说明

*   `./format` 文件夹下是格式参考
    *   原始数据格式: `data_format.json`
    *   训练模型的输入: `input_format.json`
    *   训练的模型输出: `rule_format.json`

*   `./series` 文件夹下是每天的原始数据文件
*   `./series_graph` 文件夹下是上面原始数据的可视化图表
*   `./series_input` 文件夹下是转换为模型输入格式的文件

*   `./utils` 文件夹下是辅助脚本
    *   `api.py`: api调用 gemini-3-pro preview
    *   `slice_graph.py`: 将原始图分段
    *   `gen_proba_data.py`: 从分段图大致猜原始数据点
    *   `draw.py`: 猜的数据点可视化进行对比
    *   [important] `transfer.py`: 将原始数据转换为模型输入格式的脚本

### 流程

1.  **数据包里的图 -> 大致的数据点**（api生成，仅供格式参考）
    *   每天的每15分钟数据点格式
2.  **原始数据 -> 模型输入格式**
    *   [混合字段，程序计算的数值+利用API生成的描述]
    *   [目前分段是用API分的，需要大量token，以后考虑使用程序进行趋势分析进行分段，或结合两种方法]
    *   [字段定义仅供参考，待后续调整]
3.  **最后模型输出规则**，规则格式仅供参考
    *   `./rules` 下面的是用API生成的参考内容
    *   [建议后续使用 模型+人工+特定规则 方式完善]

### 模型输入格式说明

#### 根层级
*   **`ps_id`**: 电站 ID。
*   **`date`**: 数据对应的日期。
*   **`system_specs`**: 系统硬件参数（电池容量、最大充放电功率、SOC 上下限）。
*   **`profit`**: 当日总收益（单位：元）。
    *   `ai_mode`: AI 模式下的收益。
    *   `self_mode`: 传统自发自用模式下的收益。

#### `segments` (时间分段列表)
这是一个数组，包含一天中划分出的多个时间段。每个时间段包含以下信息：

*   **`time_period`**: 时间范围（例如 "10:00 - 14:00"）。
*   **`period_type`**: 时段类型（例如 "solar_self_consumption" 光伏自用, "valley_charging" 谷电充电）。

*   **`descriptions` (环境与基础数据)**:
    *   **`pv_load`**: 光伏与负载关系。
        *   `avg_pv`: 平均光伏功率 (W)。
        *   `avg_load`: 平均负载功率 (W)。
        *   `trend`: 供需趋势 (surplus 盈余 / deficit 亏缺 / balanced 平衡)。
        *   `description`: AI 对该时段光伏和负载情况的文字描述。
    *   **`price`**: 电价情况。
        *   `avg_value`: 平均电价。
        *   `trend`: 价格趋势 (rising 上升 / falling 下降 / high 高价 / stable 稳定)。
        *   `description`: AI 对该时段电价的文字描述。

*   **`ai_mode` / `self_mode` (策略表现)**:
    这两个对象结构完全相同，分别代表 AI 策略和传统策略在该时间段的表现。
    *   **`pv_action` (光伏去向)**:
        *   `to_battery`: 光伏充入电池的平均功率 (W)。
        *   `to_load`: 光伏直接供负载的平均功率 (W)。
        *   `to_grid`: 光伏余电上网的平均功率 (W)。
        *   `description`: AI 描述光伏电力的主要流向。
    *   **`grid_action` (电网交互)**:
        *   `buy_energy`: 平均买电功率 (W)。
        *   `sell_energy`: 平均卖电功率 (W)。
        *   `description`: AI 描述与电网的交互情况（是买电还是卖电）。
    *   **`battery_soc` (电池状态)**:
        *   `start_soc`: 该时段开始时的电量 (%)。
        *   `end_soc`: 该时段结束时的电量 (%)。
        *   `trend`: 电量变化趋势 (increasing 充电 / decreasing 放电 / stable 保持)。
        *   `description`: AI 描述电池的充放电行为。
