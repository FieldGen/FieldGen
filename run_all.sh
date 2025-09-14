#!/usr/bin/env bash
# 统一运行全部分析与绘图脚本
# 可选环境变量:
#   SKIP_DATA=1        跳过数据准备 (create_tele_datasets.py / generate.py)
#   HIDE_ACT=1         在 MainExp1 / MainExp2 绘图中隐藏 ACT 曲线
#   PYTHON=python3     指定 Python 解释器
#   EXTRA_ARGS=...     追加传递给所有绘图脚本的参数（若脚本支持）
# 使用:
#   bash run_all.sh [--hide-act] [--skip-data] [--extra "ARGS"]
# 或通过环境变量控制 (优先级: 命令行参数 > 环境变量 > 默认)
set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

log() { echo -e "[RUN_ALL] $*"; }

usage() {
  cat <<EOF
run_all.sh - 运行项目全部生成与绘图脚本

可用参数:
  --hide-act        隐藏 ACT 曲线 (等效 HIDE_ACT=1)
  --skip-data       跳过数据准备 (等效 SKIP_DATA=1)
  --extra "ARGS"     追加传给所有绘图脚本的附加参数
  -h, --help        显示本帮助

也可使用环境变量:
  HIDE_ACT=1 SKIP_DATA=1 EXTRA_ARGS="--foo" bash run_all.sh
EOF
}

EXTRA_ARGS=${EXTRA_ARGS:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hide-act)
      HIDE_ACT=1; shift;;
    --skip-data)
      SKIP_DATA=1; shift;;
    --extra)
      EXTRA_ARGS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      log "未知参数: $1"; usage; exit 1;;
  esac
done

# CSV 文件名（根据现有脚本硬编码）
CSV_REQUIRED=(
  FieldGenExpData-MainExp1Teleop.csv
  FieldGenExpData-MainExp1Fieldgen.csv
  FieldGenExpData-MainExp2Teleop.csv
  FieldGenExpData-MainExp2Fieldgen.csv
  FieldGenExpData-MainExp3Diversity.csv
  FieldGenExpData-AblaBeta.csv
  FieldGenExpData-AblaCurve.csv
)

missing=()
for f in "${CSV_REQUIRED[@]}"; do
  if [[ ! -f $f ]]; then
    missing+=("$f")
  fi
done
if (( ${#missing[@]} > 0 )); then
  log "⚠ 缺失以下 CSV: ${missing[*]}"
  log "   若需要，请先放入对应文件再运行。继续执行，其它存在的图会正常输出。"
fi

# 可选数据准备步骤
if [[ ${SKIP_DATA:-0} -ne 1 ]]; then
  if [[ -f create_tele_datasets.py ]]; then
    log "Step: create tele datasets (可选)"
    # 示例参数(来自 solve.sh)，若不适用可自行修改或通过环境变量覆盖
    if [[ -n "${TELE_SOURCE:-}" && -n "${TELE_OUTPUT:-}" ]]; then
      $PYTHON create_tele_datasets.py --source "${TELE_SOURCE}" --output "${TELE_OUTPUT}" --seed "${TELE_SEED:-42}" --mode "${TELE_MODE:-symlink}" --force --prefix "${TELE_PREFIX:-fieldgen}" || log "create_tele_datasets.py 运行失败(忽略)"
    else
      log "跳过 create_tele_datasets.py (未提供 TELE_SOURCE/TELE_OUTPUT)"
    fi
  fi
  if [[ -f generate.py ]]; then
    log "Step: generate (可选)"
    $PYTHON generate.py || log "generate.py 执行失败(忽略)"
  fi
else
  log "跳过数据准备步骤 (SKIP_DATA=1)"
fi

# 统一参数: 隐藏 ACT
MAINEXP_HIDE_FLAG=""
if [[ ${HIDE_ACT:-0} -eq 1 ]]; then
  MAINEXP_HIDE_FLAG="--hide-act"
fi

# 创建输出目录
mkdir -p figs pdf

# MainExp1
if [[ -f plot_mainexp1_results.py ]]; then
  log "Step: MainExp1 plots"
  $PYTHON plot_mainexp1_results.py $MAINEXP_HIDE_FLAG ${EXTRA_ARGS:-} || log "MainExp1 生成出错(继续)"
fi
# MainExp2
if [[ -f plot_mainexp2_results.py ]]; then
  log "Step: MainExp2 plots"
  $PYTHON plot_mainexp2_results.py $MAINEXP_HIDE_FLAG ${EXTRA_ARGS:-} || log "MainExp2 生成出错(继续)"
fi
# MainExp3 Diversity
if [[ -f plot_mainexp3_diversity.py ]]; then
  log "Step: MainExp3 diversity"
  $PYTHON plot_mainexp3_diversity.py $MAINEXP_HIDE_FLAG ${EXTRA_ARGS:-} || log "Diversity 生成出错(继续)"
fi
# Ablation Beta
if [[ -f plot_abla_beta.py ]]; then
  log "Step: Abla Beta"
  $PYTHON plot_abla_beta.py "--no-origin" ${EXTRA_ARGS:-} || log "Abla Beta 生成出错(继续)"
fi
# Ablation Curve
if [[ -f plot_abla_curve.py ]]; then
  log "Step: Abla Curve"
  $PYTHON plot_abla_curve.py ${EXTRA_ARGS:-} || log "Abla Curve 生成出错(继续)"
fi

log "全部步骤完成"
