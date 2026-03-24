# Cleanup Procedure for GPU1 Pipeline

## Quick Cleanup

### Method 1: Using cleanup script (recommended)
```bash
bash cleanup_gpu1.sh
```

This script will:
- Kill all running v2t-server and v2t-worker processes for both `vqa` and `cot` modes
- Remove all pid files from the `pids/` directory
- Display remaining logs for inspection
- Show summary of cleanup results

### Method 2: Automatic cleanup on interrupt
When running `run_serial_gpu1.sh`, pressing **Ctrl+C** will automatically:
- Trigger INT/TERM signal handler (trap)
- Call internal `cleanup()` function
- Kill all background processes gracefully
- Clean up pid files
- Exit with code 130

### Method 3: Manual in-script cleanup
If needed, you can source the script and call cleanup directly:
```bash
source run_serial_gpu1.sh
cleanup
```

## Logs Location

All logs are saved in the `logs/` directory:
- `v2t-server.vqa.log` - VQA stage server output
- `v2t-worker.vqa.log` - VQA stage worker output
- `v2t-server.cot.log` - CoT stage server output
- `v2t-worker.cot.log` - CoT stage worker output

View live logs:
```bash
tail -f logs/v2t-server.vqa.log
tail -f logs/v2t-worker.vqa.log
```

## PID Files Location

Process IDs are stored in the `pids/` directory during execution:
- `v2t-server.vqa.pid`
- `v2t-worker.vqa.pid`
- `v2t-server.cot.pid`
- `v2t-worker.cot.pid`

These files are automatically removed by cleanup scripts.

## Manually Kill Processes

If automatic cleanup fails, manually kill by pid:

```bash
# Find and kill all v2t processes
pkill -f "v2t-server|v2t-worker"

# Or kill specific processes
kill -9 $(cat pids/v2t-server.vqa.pid)
kill -9 $(cat pids/v2t-worker.vqa.pid)

# Clean up pid directory
rm -f pids/*.pid
```

## Troubleshooting

### Queue stuck / Data not processing
Before restarting, run cleanup to ensure old processes don't interfere:
```bash
bash cleanup_gpu1.sh
# Then restart with:
bash run_serial_gpu1.sh
```

### Port already in use
If you see "Address already in use" error:
```bash
# First cleanup:
bash cleanup_gpu1.sh

# Then check what's using the port (typically 5000/5001):
lsof -i :5000
lsof -i :5001

# Manually kill if cleanup didn't work:
kill -9 <PID>
```

### Zombie processes
If cleanup shows processes but they won't die:
```bash
# Run cleanup again with force option would require manual:
bash cleanup_gpu1.sh

# If still stuck, use system tools:
ps aux | grep "v2t-"
# Then manually kill with appropriate pid
```
