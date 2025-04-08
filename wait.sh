sleep_seconds=$((5 * 60 * 60))

echo "sleep $sleep_seconds seconds"
# 休眠5小时
sleep $sleep_seconds

# 执行目标命令
bash script/match/match_sample_7b.sh 5,6 8009