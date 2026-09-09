# NPU Timeline Generation Guide
## Prerequisites
- Python environment
- Chrome browser (for visualization)
## Implementation Steps
### 1. Code Modification
#### Register the subscriber
Add the following at the beginning of your program:
```cpp
MsptiMetrics::register_subscriber();
```
#### Add tracing to ACLNN functions (work for msprof as well)
Insert the following macro in your ACLNN functions where you want to measure performance:
```cpp
LLM_MSTX_RANGE();
```
#### Release the subscriber
Add this at the end of your program:
```cpp
MsptiMetrics::release_subscriber();
```
### 2. Log Processing
After running your program, process the generated log file using the timeline script:
```bash
python npu_timeline.py -i custom_log.log -o custom_output.json
```
### 3. Visualization
Open Chrome browser
Navigate to: chrome://tracing
Load the generated JSON file: custom_output.json

## MiniMax H3

Audit an original Ref2VA checkpoint without loading tensor payloads:

```bash
python tools/minimax_h3_checkpoint_audit.py \
  --checkpoint-path /path/to/MiniMax-H3/Ref2VA \
  --output-path /path/to/checkpoint_manifest.json
```

Build the source-to-runtime weight ledger and optionally compare every
converted tensor payload bitwise:

```bash
python tools/minimax_h3_weight_ledger.py \
  --checkpoint-path /path/to/MiniMax-H3/Ref2VA \
  --converted-path /path/to/Ref2VA-diffusers \
  --converter-path /path/to/diffusers/scripts/convert_minimax_h3_to_diffusers.py \
  --output-path /path/to/weight_ledger.jsonl \
  --summary-path /path/to/weight_ledger_summary.json \
  --validate-payload
```

Run a converted Modular Diffusers Ref2VA text-plus-image reference case with
group offloading:

```bash
python tools/minimax_h3_reference.py \
  --checkpoint-path /path/to/Ref2VA-diffusers \
  --image-path /path/to/reference.png \
  --prompt-path /path/to/prompt.txt \
  --output-dir /path/to/artifact \
  --device npu:0
```
