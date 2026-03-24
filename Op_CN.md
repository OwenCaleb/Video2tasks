# Video2Tasks 运行手册（精简）

## 1. 单配置文件
项目已统一为单配置文件：
- config.yaml

关键字段：
- run.task_type: segment | vqa | cot
- prompt.segment_task_id / prompt.vqa_task_id / prompt.cot_task_id
- datasets[].video_subset: 视频子集（segment/cot 使用）
- datasets[].frame_subset: 帧图子集（vqa 使用）
- datasets[].data: 样本筛选（可选）

样本筛选示例（只处理前两个样本）：
```yaml
datasets:
  - root: /path/to/data_root
    video_subset: video_retarget_head
    frame_subset: frame_retarget_head
    data: [0, 1]
```

data 支持：
- 整数：按排序后的样本索引
- 字符串：按样本 id 精确匹配

## 2. 启动 Segment
```bash
v2t-server -c config.yaml --mode segment
v2t-worker -c config.yaml --mode segment
```

输出：
- runs/<video_subset>/<run_id>/samples/<sample_id>/segments.json

## 3. 启动 VQA
```bash
v2t-server -c config.yaml --mode vqa
v2t-worker -c config.yaml --mode vqa
```

输出：
- runs/<frame_subset>/<run_id>/vqa/<sample_id>/<qtype>.jsonl

## 4. 启动 CoT
CoT 依赖 segment 产物（segments.json）。

```bash
v2t-server -c config.yaml --mode cot
v2t-worker -c config.yaml --mode cot
```

输出：
- runs/<video_subset>/<run_id>/cot/<sample_id>/cot_results.json

## 5. CoT 视觉证据对齐（已更新）
CoT 不再使用固定 frames_per_segment。
当前按每个 segment 的时间段对齐采样：
- cot.sample_hz
- cot.min_frames_per_segment
- cot.max_frames_per_segment

这保证了：
- t 段 instruction + t 段视频证据 + t 段 cot 对齐
- 基于 subtask 反推该时间段 cot

## 6. 预览完整 Prompt
```bash
python -m video2tasks.prompt.preview_prompt --mode seg --config config.yaml --n-images 16
python -m video2tasks.prompt.preview_prompt --mode vqa --config config.yaml --qtype spatial --max-questions 2
python -m video2tasks.prompt.preview_prompt --mode cot --config config.yaml --n-images 8
```
