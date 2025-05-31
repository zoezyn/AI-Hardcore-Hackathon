# benchmark.py
import torch, time, os, json, subprocess, argparse
parser = argparse.ArgumentParser(); parser.add_argument("--model", default="resnet18")
args = parser.parse_args()

model = getattr(torch.hub, "load")("pytorch/vision", args.model, pretrained=False).cuda().eval()
dummy = torch.randn(1, 3, 224, 224, device="cuda")
# warm-up
for _ in range(5): model(dummy)
t0 = time.perf_counter(); model(dummy); torch.cuda.synchronize(); dt = (time.perf_counter()-t0)*1e3
pwr = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"]).decode().strip()
result = dict(model=args.model, runtime_ms=round(dt,2), power_w=float(pwr.split()[0]))
print(json.dumps(result))
