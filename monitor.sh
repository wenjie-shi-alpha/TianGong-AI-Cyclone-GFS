#!/bin/bash
# 全量运行监控脚本
# 用法: bash monitor.sh

LOG="/root/TianGong-AI-Cyclone-GFS/run_fullscale.log"
PIDFILE="/tmp/fullscale_pid.txt"
OUTPUT_DIR="/root/TianGong-AI-Cyclone-GFS/final_single_output"

# 颜色
G='\033[0;32m'; Y='\033[0;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'

echo -e "${B}═══════════════════════════════════════════════════════════════${N}"
echo -e "${B}  🌀 台风GFS全量Pipeline监控面板${N}"
echo -e "${B}═══════════════════════════════════════════════════════════════${N}"

# 1. 进程状态
if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        CPU=$(ps -p "$PID" -o %cpu= 2>/dev/null)
        MEM=$(ps -p "$PID" -o rss= 2>/dev/null)
        MEM_GB=$(echo "scale=1; $MEM/1048576" | bc 2>/dev/null || echo "?")
        ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null)
        echo -e "${G}✅ 进程运行中${N}  PID=$PID  CPU=${CPU}%  MEM=${MEM_GB}GB  运行时间: $ELAPSED"
    else
        echo -e "${R}❌ 进程已结束${N}  PID=$PID"
    fi
else
    echo -e "${Y}⚠️  PID文件不存在${N}"
fi

# 2. 磁盘/内存
echo ""
echo -e "${B}📊 资源使用:${N}"
df -h / /dev/shm 2>/dev/null | awk 'NR>1{printf "   %-10s 已用 %-6s 可用 %-6s (%s)\n", $1, $3, $4, $5}'
echo "   RAM: $(free -h | awk '/Mem:/{printf "已用 %s / 总 %s (%s可用)", $3, $2, $7}')"

# 3. 产出统计
echo ""
N_JSON=$(find "$OUTPUT_DIR" -name "*.json" ! -name "_*" 2>/dev/null | wc -l)
N_TRACK=$(find /root/TianGong-AI-Cyclone-GFS/track_single -name "*.csv" 2>/dev/null | wc -l)
N_NC=$(find /root/TianGong-AI-Cyclone-GFS/data/grib_nc -name "*.nc" 2>/dev/null | wc -l)
N_GRIB=$(find /dev/shm/grib_cache -name "*.f*" 2>/dev/null | wc -l)
echo -e "${B}📁 产出文件:${N}"
echo "   JSON分析:  $N_JSON 个"
echo "   追踪CSV:   $N_TRACK 个"
echo "   NC缓存:    $N_NC 个 (处理后自动删除)"
echo "   GRIB缓存:  $N_GRIB 个 (写NC后自动删除)"

# 4. 最近产出的JSON
echo ""
echo -e "${B}📋 最近5个产出的JSON:${N}"
ls -t "$OUTPUT_DIR"/*.json 2>/dev/null | grep -v "_analysis" | head -5 | while read f; do
    echo "   $(date -r "$f" '+%H:%M:%S')  $(basename "$f")"
done

# 5. 批次进度 (从日志提取)
echo ""
echo -e "${B}📦 批次进度:${N}"
if [[ -f "$LOG" ]]; then
    # 最新的批次完成信息
    grep "批次.*完成:" "$LOG" | tail -3 | while read line; do
        echo "   $line"
    done
    echo ""
    # 最后10行日志
    echo -e "${B}📝 最新日志:${N}"
    tail -8 "$LOG" | while read line; do
        echo "   $line"
    done
else
    echo "   日志文件不存在"
fi

echo ""
echo -e "${B}═══════════════════════════════════════════════════════════════${N}"
echo -e "  停止运行: kill \$(cat $PIDFILE)"
echo -e "  查看日志: tail -f $LOG"
echo -e "  再次监控: bash monitor.sh"
echo -e "${B}═══════════════════════════════════════════════════════════════${N}"
