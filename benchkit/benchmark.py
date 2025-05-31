import torch, time, json, subprocess, argparse, platform, re
from torchvision import models
from torchinfo import summary           # pip install torchinfo

def smi(q):  # helper
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]
    ).decode().strip()
def safe_sm_count():
    try:
        return int(smi("multiprocessor_count"))
    except subprocess.CalledProcessError:
        txt = subprocess.check_output(["nvidia-smi","-q"]).decode()
        m   = re.search(r"Multiprocessors\s*:\s*(\d+)", txt)
        return int(m.group(1)) if m else None



parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18")
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

# ── model stats ────────────────────────────────────────────────────────────────
model_cls = getattr(models, args.model)
model = model_cls(weights=None).cuda().eval()
inp = torch.randn(args.batch, 3, 224, 224, device="cuda")
num_params = sum(p.numel() for p in model.parameters())
flops = summary(model, input_data=inp, verbose=0).total_mult_adds  # 1 MAC = 2 FLOPs

# ── warm-up & timing ───────────────────────────────────────────────────────────
for _ in range(10): model(inp)
t0 = time.perf_counter(); model(inp); torch.cuda.synchronize()
runtime_ms = (time.perf_counter() - t0) * 1e3

# ── power & GPU descriptors ───────────────────────────────────────────────────
power_w  = float(smi("power.draw"))
gpu_name = smi("name")
vram_gb  = int(smi("memory.total"))
sm_clock = int(smi("clocks.sm"))
sm_count = safe_sm_count()  # safer than smi("multiprocessor_count")

# CUDA & driver
cuda_ver   = torch.version.cuda
driver_ver = smi("driver_version")

# ── CPU info ──────────────────────────────────────────────────────────────────
cpu_cores = int(subprocess.check_output(["nproc"]).decode())
cpu_model = platform.processor()

# ── dump one rich JSON record ─────────────────────────────────────────────────
print(json.dumps({
    "model": args.model,
    "batch": args.batch,
    "params": num_params,
    "flops": flops,
    "runtime_ms": round(runtime_ms, 2),
    "power_w": power_w,
    "gpu_name": gpu_name,
    "vram_GB": vram_gb,
    "sm_count": sm_count,
    "sm_clock_MHz": sm_clock,
    "cuda_ver": cuda_ver,
    "driver_ver": driver_ver,
    "cpu_model": cpu_model,
    "cpu_cores": cpu_cores
}, separators=(",",":")))
