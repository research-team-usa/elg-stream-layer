# Open Origin – ELG Stream Layer
### Asynchronous INT8 Layer‑Paging Engine for Large Language Models
**Architected, validated, and tested by Emanuel Schaaf — Runs successfully on real hardware**

<img width="1917" height="1141" alt="Screenshot 2026-10-03 093252" src="https://github.com/user-attachments/assets/8640aec5-9ba4-48ad-bfbd-11306309b4e3" />

---

## 📌 Overview
The **Open Origin – ELG Stream Layer** (Elastic Layer Grid – Optimized Streaming) is a custom, high‑performance runtime architecture for large language models (LLMs). 
It bypasses the physical VRAM limits of consumer GPUs by streaming model layers dynamically from system RAM using a highly optimized, block-allocated INT8 memory pool.

Designed and validated by **Emanuel Schaaf (Pirmasens, Germany)**, this engine utilizes the Google **Gemma** base architecture. It has been stress-tested and proven fully operational on a self-hosted Debian Linux server running an NVIDIA GTX 1080 Ti.

---

## 🚀 Key Features

### **1. Block-Allocated INT8 Memory Pool (7.75 GB instead of 18 GB)**
Instead of loading massive FP16 weights into limited VRAM, the engine quantizes all model weights into a single, contiguous INT8 memory block pinned to the system RAM. This reduces memory footprint by over 50% while maintaining precision.

### **2. Asynchronous Layer Paging**
Transformer layers are never kept statically in VRAM. Instead, the engine processes them dynamically:
1. **Prefetch:** Fetched asynchronously from the RAM pool.
2. **Transfer:** Streamed across the PCIe bus.
3. **Reconstruction:** Converted on-the-fly from INT8 back to FP16 directly on the GPU.
4. **Purge:** Instantly freed from VRAM after the computation cycle.

### **3. Double‑Buffered PCIe Prefetching**
A dedicated secondary CUDA stream handles the PCIe transfer of layer $N+1$ while the GPU actively computes layer $N$. This masks PCIe latency and ensures maximum GPU utilization.

### **4. Persistent Pre-Fabrication (Zero-Delay Boot)**
During the initial run, the engine quantizes the model and commits the exact RAM block to the SSD:
- `gemma_int8_pool.pt`
- `gemma_scales.pt`

On subsequent boots, the engine bypasses CPU quantization completely. It maps the pre-fabricated 7.75 GB binary dump directly into the pinned RAM pool using secure `weights_only=True` loading, starting the API server in seconds.

### **5. OpenAI‑Compatible Interface**
Features a built-in, lightweight web server exposing a `/v1/chat/completions` endpoint, providing native drop-in compatibility for Open WebUI and other standard API clients.

### **6. Micro VRAM Footprint**
While a standard Gemma 2 9B model demands ~18 GB of VRAM, the ELG Stream Layer keeps the base VRAM allocation strictly at **~1.71 GB** for permanent foundational modules, enabling massive models to run smoothly on older 11 GB or 8 GB GPUs.

---

## 🖥 System Requirements

### **Operating System**
- Debian 13 Linux (Recommended / Validated)
- Any modern Linux distribution

### **Hardware**
- **GPU:** NVIDIA GPU with CUDA support (Validated on GTX 1080 Ti)
- **RAM:** Minimum 16 GB System RAM (20+ GB recommended for 9B models)
- **Bus:** PCIe 3.0 or higher

### **Software Dependencies**
Install the baseline requirements:

```bash
sudo apt update
sudo apt install -y python3 python3-venv git
```

---

## 🐍 Python Environment

Create a clean virtual environment to prevent system conflicts:

```bash
python3 -m venv vram_env
source vram_env/bin/activate
```

Install PyTorch and required ML libraries for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers accelerate sentencepiece
```

---

## 📦 Installation & Execution

### 1. Setup the Project Directory
Place the project files into the designated path:
```bash
mkdir -p /opt/
```

Required files in this directory:
- [`elg-stream-layer.py`](elg-stream-layer.py)
- [`LICENSE.md`](License.md)

*(Note: `gemma_int8_pool.pt` and `gemma_scales.pt` will be automatically generated upon the first execution).*

### 2. Set Permissions
Ensure the script is executable:
```bash
sudo chmod +x /opt/elg-stream-layer.py
```

### 3. Start the ELG Engine
Activate the environment and ignite the engine:
```bash
source vram_env/bin/activate
python3 /opt/elg-stream-layer.py
```
*The API will listen on `http://0.0.0.0:5005/v1`.*

---

## 🔧 Technical Architecture

### **INT8 Quantization & Direct Mapping**
Weights are calculated using per-parameter scaling to retain context quality:

$$scale = \frac{\max(\vert{}W\vert{})}{127}$$
$$int8\_weight = \text{round}\left(\frac{W}{scale}\right)$$

The resulting INT8 tensors are mapped into a pre-allocated pinned memory pool. Pinned (page-locked) memory enables **zero-copy DMA (Direct Memory Access)** transfers directly to the GPU, bypassing CPU bottlenecks.

### **FP16 On-The-Fly GPU Reconstruction**
Once the INT8 packet arrives on the GPU, it is mathematically restored before computation:
```python
fp16_gpu_tensor = int8_gpu_tensor.to(torch.float16) * scale
```

---

## 📜 License

This system is strictly governed by the:
[**Open Origin – ELG License (OO‑ELG) Version 1.0**](LICENSE.md)

**Core Tenets:**
- **Non‑Commercial:** Monetization, sale, or SaaS deployment is strictly forbidden without written consent.
- **Attribution:** The Architect (Emanuel Schaaf) and Co-Architectures must be credited.
- **Functional Integrity:** Forks and modifications MUST remain fully operational. Defunctionalization or intentional sabotage of the code is prohibited.
- **Share‑Alike:** All derivatives must inherit this exact license.

---

## 🙌 Credits & Architecture Signatures

- **Architect & Lead Developer:** Emanuel Schaaf 
- **Co‑Architecture:** Lyra / Google Gemini Enterprise
- **Base Model Architecture:** Google Gemma

---

**Status:** ✔ Validated. Operational. Tested on Bare Metal.📧[contakt](Contact.md) .
