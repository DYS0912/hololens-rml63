# legacy/ —— 历史实验版本（保留备查）

这些程序都曾在实机上运行过单次实验，但已被 `scripts/` 中的当前程序取代：

| 文件 | 历史 | 取代者 |
|---|---|---|
| hololens_hand_step.py | 单次向上步进 v1（上限 5mm，scale 0.2） | hololens_hand_step_v2.py → scripts/hololens_hand_step_5cm.py |
| hololens_hand_step_v2.py | v2（1:1；预览 5cm，实机上限 5mm）；实机完成 5mm（日志 hand_step_v2_1794899805085697543.jsonl） | scripts/hololens_hand_step_5cm.py |
| hololens_single_trigger.py | 单次触发固定 +Z 5mm 后请求停止；实机执行过（日志 single_trigger_1794895213957199167.jsonl） | 停止实验由 5cm 程序内停止逻辑取代 |
| robot_stop_check_v2.py | 单命令诊断（发一次目标后停止） | 历史诊断工具 |
| 单次手控v2_操作说明.md | v2 配套操作说明（含"停止请求≠立即停止"等边界） | — |

**这些是单次实验程序，不是实时跟随：绝不连续重跑累计位移。** 如需实机复现，
先读 scripts/hololens_hand_step_5cm.py 的 docstring 与 docs/交接说明.md 第五节。
