#!/usr/bin/env python3
"""重写 LeRobot 0.6.2 数据集里的图像绝对路径（项目整体迁移后使用）。

背景: LeRobot 0.6.2 的图像列存【绝对路径】（PIL 直接 open，不做相对解析），
转换器因此写出输出目录下的绝对路径。整个项目复制/移动到新位置后，
parquet 中的旧路径会失效。两种修复方式:
  1) 重新生成（推荐）: 在新位置重新跑 scripts/record_dora.py，生成全新的
     绝对路径；
  2) 重建路径（本工具）: 图像文件已随项目一起搬移时，把 parquet 图像列
     的前缀改写到新位置。

用法（任意带 pyarrow 的 Python）:
    python3 scripts/rewrite_image_paths.py --dataset <数据集目录> \
        --new-dataset-root <搬移后的数据集目录> [--write]

默认 dry-run：只打印改写计划，不修改任何文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def plan_rewrite(dataset: Path, new_root: Path) -> list:
    import pyarrow.parquet as pq

    plans = []
    for f in sorted((dataset / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(f)
        touched = False
        new_columns = {}
        for name in table.column_names:
            if not name.startswith("observation.images."):
                continue
            values = []
            for v in table[name].to_pylist():
                if not isinstance(v, str):
                    values.append(v)
                    continue
                if "/images/" in v:
                    rel = v.split("/images/", 1)[1]
                    new_value = f"{new_root}/images/{rel}"
                else:
                    new_value = v  # 已是相对/异常值，保持不动
                if new_value != v:
                    touched = True
                values.append(new_value)
            if touched:
                new_columns[name] = values
        if new_columns:
            plans.append((f, new_columns))
    return plans


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="重写 LeRobot 数据集图像绝对路径")
    parser.add_argument("--dataset", required=True, help="数据集目录（当前位置）")
    parser.add_argument("--new-dataset-root", required=True,
                        help="搬移后的数据集目录（新绝对路径前缀）")
    parser.add_argument("--write", action="store_true", help="实际写回（默认 dry-run）")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        raise SystemExit(f"数据集不存在: {dataset}")
    plans = plan_rewrite(dataset, Path(args.new_dataset_root))
    if not plans:
        print("无需改写（图像列路径已正确或无图像列）")
        return 0

    n_values = 0
    for f, cols in plans:
        print(f"[{'WRITE' if args.write else 'DRY-RUN'}] {f}")
        for name, values in cols.items():
            for old_or_new in values:
                n_values += 1
            print(f"  列 {name}: {len(values)} 条")
        if args.write:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pq.read_table(f)
            for name, values in cols.items():
                table = table.set_column(
                    table.column_names.index(name),
                    name,
                    pa.array(values, type=pa.string()),
                )
            pq.write_table(table, f)

    print(f"共 {len(plans)} 个 parquet 文件、{n_values} 条图像路径"
          f"{'已改写' if args.write else '（dry-run，未修改任何文件）'}")
    if not args.write:
        print("确认无误后加 --write 执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
