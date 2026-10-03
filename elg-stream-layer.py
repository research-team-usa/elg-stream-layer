#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modul: Asynchroner Double-Buffered Layer-Paging Streamer
# Architektur: Open Origin elg-stream-layer
# Autor: Emanuel Schaaf Pirmasens Germany
# Co‑Architektur: Google Gemini Enterprise 
# Live: Valiedierung auf dem eingenen Server Getestet bestanden! Copyright ELG License
import os
import sys
import time
import json
import gc
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- HIER SIND DIE BEIDEN GLOBALEN VARIABLEN ---
global_tokenizer = None
global_model = None

# ==================== KONFIGURATION ====================
# Fest hinterlegtes Modell: Gemma 2 9B Instruct
# (Kann bei Bedarf über die Umgebungsvariable MODEL_ID angepasst werden, z. B. auf 'google/gemma-2-2b-it')
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-2-9b-it")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5005"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def print_vram_status(label: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{label}] VRAM belegt: {allocated:.2f} GB | VRAM reserviert: {reserved:.2f} GB")

def get_system_ram_available_gb():
    """Liest den verfügbaren RAM direkt aus /proc/meminfo (ohne externe Abhängigkeiten)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemAvailable:" in line:
                    parts = line.split()
                    return int(parts[1]) / (1024 * 1024)
    except Exception:
        return 0.0

def print_vram_status(label: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{label}] VRAM belegt: {allocated:.2f} GB | VRAM reserviert: {reserved:.2f} GB")

# ==================== PAGING ENGINE ====================

class AsyncLayerPagingEngine:
    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = torch.device(device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        
        # Decoder-Layers und Basismodell lokalisieren
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.base = model.model
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.base = model.transformer
            self.layers = model.transformer.h
        else:
            raise ValueError("Modellarchitektur nicht unterstützt: 'layers' wurde nicht gefunden.")
            
        self.num_layers = len(self.layers)
        print(f"\n[ENGINE] Initialisiere Paging-Engine für {self.num_layers} Transformer-Layer...")

        # 1. Permanente Module fest auf die GPU legen
        for attr in ["embed_tokens", "norm", "rotary_emb"]:
            if hasattr(self.base, attr):
                getattr(self.base, attr).to(self.device)
        if hasattr(model, "lm_head"):
            model.lm_head.to(self.device)

        # 2. Layer-Buffer (z. B. Masken, Frequenzen) permanent auf GPU halten (vernachlässigbare Größe)
        for layer in self.layers:
            for _, buf in layer.named_buffers():
                buf.data = buf.data.to(self.device)

# 3. Layer-Gewichte in EINER INT8-RAM-Block (Block-Allokation + Festplatten-Cache) verankern
        print(f"[ENGINE] Prüfe auf vorgefertigten INT8-Cache...")
        
        CACHE_DIR = "/opt/open_origin/test"
        TENSOR_CACHE_PATH = os.path.join(CACHE_DIR, "gemma_int8_pool.pt")
        SCALES_CACHE_PATH = os.path.join(CACHE_DIR, "gemma_scales.pt")
        
        # A) Berechne die benötigte Gesamtgröße
        total_elements = sum(param.numel() for layer in self.layers for param in layer.parameters())
        print(f"[ENGINE] Benötigte INT8-RAM-Block: {total_elements / (1024**3):.2f} GB")
        
        loaded_from_cache = False
        # B) Cache laden ODER neu erschaffen
        if os.path.exists(TENSOR_CACHE_PATH) and os.path.exists(SCALES_CACHE_PATH):
            print(f"[ENGINE] Lade fertige INT8-RAM-Block direkt von der Festplatte (Pre-Fabrication)...")
            # Lädt den rohen Dump in Sekunden und pinnt ihn sofort für DMA
            self.memory_pool = torch.load(TENSOR_CACHE_PATH, weights_only=True).pin_memory()
            self.scales_pool = torch.load(SCALES_CACHE_PATH, weights_only=True)
            loaded_from_cache = True
        else:
            print(f"[ENGINE] Kein Cache gefunden. Erschaffe und quantisiere neue INT8-Block...")
            self.memory_pool = torch.empty(total_elements, dtype=torch.int8, device="cpu").pin_memory()
            self.scales_pool = {}

        self.cpu_params = []
        current_offset = 0
        
        print(f"[ENGINE] Mappe Layer-Gewichte in die Speicher-Block...")
        # C) Gewichte zuweisen (und quantisieren, falls kein Cache existierte)
        for idx, layer in enumerate(self.layers):
            layer.to("cpu")
            param_dict = {}
            for name, param in layer.named_parameters():
                numel = param.numel()
                # Zuweisung des Speicherbereichs
                pool_slice = self.memory_pool[current_offset : current_offset + numel].view(param.shape)
                
                # Nur berechnen und kopieren, wenn die Daten nicht schon aus dem Cache kommen!
                if not loaded_from_cache:
                    scale = param.data.abs().max() / 127.0
                    if scale == 0: scale = torch.tensor(1e-9, dtype=torch.float16)
                    self.scales_pool[f"{idx}_{name}"] = scale
                    
                    int8_data = torch.round(param.data / scale).to(torch.int8)
                    pool_slice.copy_(int8_data)
                
                param_dict[name] = pool_slice
                current_offset += numel
                
            self.cpu_params.append(param_dict)
            
        # D) Cache für den nächsten Start abspeichern
        if not loaded_from_cache:
            print(f"[ENGINE] Speichere neue INT8-Block auf SSD/HDD für sofortige zukünftige Starts...")
            torch.save(self.memory_pool, TENSOR_CACHE_PATH)
            torch.save(self.scales_pool, SCALES_CACHE_PATH)
            
        print(f"[ENGINE] INT8-Memory-Pool erfolgreich verankert und startklar!")
        
        self.prefetched_gpu_tensors = None
        self.prefetched_layer_idx = None

        self._register_hooks()
        print("[ENGINE] Asynchrone Paging-Hooks erfolgreich aktiv.\n")

    def _async_prefetch(self, layer_idx: int):
        """Startet den asynchronen PCIe-Transfer (jetzt nur noch halbe Datenmenge!)."""
        if layer_idx >= self.num_layers:
            self.prefetched_gpu_tensors = None
            self.prefetched_layer_idx = None
            return

        with torch.cuda.stream(self.transfer_stream):
            gpu_tensors = {}
            for name, cpu_tensor in self.cpu_params[layer_idx].items():
                # PCIe Transfer: Überträgt den kleinen INT8-Block
                gpu_tensors[name] = cpu_tensor.to(self.device, non_blocking=True)
            self.prefetched_gpu_tensors = gpu_tensors
            self.prefetched_layer_idx = layer_idx

    def _make_pre_hook(self, idx: int):
        def hook(module, args):
            torch.cuda.current_stream().wait_stream(self.transfer_stream)

            if self.prefetched_layer_idx != idx or self.prefetched_gpu_tensors is None:
                self._async_prefetch(idx)
                torch.cuda.current_stream().wait_stream(self.transfer_stream)

            # --- GPU-KONVERTER ---
            # INT8 -> FP16 On-the-Fly Rekonstruktion direkt auf der Grafikkarte
            for name, param in module.named_parameters():
                int8_gpu_tensor = self.prefetched_gpu_tensors[name]
                scale = self.scales_pool[f"{idx}_{name}"].to(self.device)
                
                # Mathematisch korrekte Rückrechnung: (Werte * Scale)
                fp16_gpu_tensor = int8_gpu_tensor.to(torch.float16) * scale
                param.data = fp16_gpu_tensor

            if idx + 1 < self.num_layers:
                self._async_prefetch(idx + 1)
            else:
                self.prefetched_gpu_tensors = None
                self.prefetched_layer_idx = None

            return None
        return hook

    def _make_post_hook(self, idx: int):
        def hook(module, args, output):
            # VRAM des Layers sofort auf der GPU freigeben!
            # Da die echten Daten sicher als INT8 im RAM liegen, 
            # löschen wir den entpackten FP16-Tensor auf der GPU einfach.
            for name, param in module.named_parameters():
                param.data = torch.empty(0, dtype=torch.float16, device="cpu")
            return output
        return hook
        
    def _register_hooks(self):
        # Trigger vor dem Token-Embedding: Startet Vorabladen von Layer 0
        if hasattr(self.base, "embed_tokens"):
            self.base.embed_tokens.register_forward_pre_hook(self._on_embed_pre)

        # Vor- und Nachbereitung für jeden einzelnen Decoder-Layer
        for idx, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._make_pre_hook(idx))
            layer.register_forward_hook(self._make_post_hook(idx))

    def _on_embed_pre(self, module, args):
        # Sobald ein neuer Forward-Pass startet: Layer 0 asynchron vorab in den VRAM holen
        self._async_prefetch(0)
        
class OpenOriginPagingAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/v1/models":
            self.send_response(200)
            self.send_header('Content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"object":"list","data":[{"id":"openorigin-paging-gemma","object":"model","owned_by":"openorigin"}]}).encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
            
        length = int(self.headers['Content-Length'])
        req = json.loads(self.rfile.read(length).decode('utf-8'))
        msgs = req.get("messages", [])
        
        # --- BUGFIX: System-Rolle für Gemma filtern ---
        # Gemma stürzt ab, wenn Open WebUI eine "system"-Rolle schickt. Wir werfen sie einfach raus.
        msgs = [m for m in msgs if m.get("role") != "system"]
        # ----------------------------------------------
        
        print(f"\n[API] Eingehende Anfrage. Nachrichten: {len(msgs)}")
        
        try:
            prompt = global_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            input_ids = global_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            
            t0 = time.time()
            with torch.no_grad():
                output_ids = global_model.generate(
                    input_ids,
                    max_new_tokens=req.get("max_tokens", 8192),
                    temperature=req.get("temperature", 0.7),
                    do_sample=req.get("temperature", 0.7) > 0,
                    pad_token_id=global_tokenizer.eos_token_id
                )
            total_time = time.time() - t0
            
            generated_ids = output_ids[0][input_ids.shape[1]:]
            final_text = global_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            num_tokens = len(generated_ids)
            print(f"[API] Generierung abgeschlossen: {num_tokens} Tokens in {total_time:.2f}s")
            
            resp = {
                "id": "openorigin-paging",
                "object": "chat.completion",
                "model": "openorigin-paging-gemma",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}]
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            import traceback
            print(f"[API-ERROR] {traceback.format_exc()}")
            self.send_error(500)
            
        finally:
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def log_message(self, format, *args): 
        pass
        
def main():
    print("=" * 53)
    print("🚀 OPEN ORIGIN: ELG-stream-layer (KI API)🚀")
    print("=" * 53)

    if not torch.cuda.is_available():
        print("❌ FEHLER: Kein CUDA-Gerät gefunden. Eine NVIDIA GPU ist erforderlich.")
        sys.exit(1)

    print(f"Grafikkarte: {torch.cuda.get_device_name(0)}")
    print(f"Modell-Ziel: {MODEL_ID}")

    # RAM-Vorprüfung
    ram_avail = get_system_ram_available_gb()
    print(f"Verfügbarer System-RAM: {ram_avail:.2f} GB")
    if "9b" in MODEL_ID.lower() and ram_avail < 15.0:
        print("\n⚠️ WARNUNG: Gemma 2 9B benötigt im System-RAM ca. 18 GB.")
        print(f"Aktuell sind nur {ram_avail:.2f} GB frei. Falls der Ladevorgang abbricht:")
        print("  1. Führe 'sync; echo 3 > /proc/sys/vm/drop_caches' als Root aus.")
        print("  2. Oder teste zuerst mit 'export MODEL_ID=google/gemma-2-2b-it'.\n")

    print_vram_status("Init")

    # 1. Tokenizer laden
    print("\n[1/4] Lade Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"\n❌ FEHLER beim Laden des Tokenizers: {e}")
        print("\nHINWEIS: Gemma ist ein lizenziertes Modell von Google auf Hugging Face.")
        print("Stelle sicher, dass du auf huggingface.co die Lizenz bestätigt hast")
        print("und eingeloggt bist (z. B. via 'huggingface-cli login' oder 'export HF_TOKEN=...').")
        sys.exit(1)

    # 2. Modell im FP16-Modus auf die CPU laden (Grafikkarte nutzt FP16 stabil)
    print("\n[2/4] Lade Modellgewichte in den System-RAM (low_cpu_mem_usage=True)...")
    start_load = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"\n❌ FEHLER beim Modell-Download/Laden: {e}")
        sys.exit(1)
        
    print(f"Modell geladen in {time.time() - start_load:.1f}s.")
    model.eval()

    # 3. Paging-Engine einhängen
    print("\n[3/4] Binde asynchrone Streaming-Engine ein...")
    engine = AsyncLayerPagingEngine(model, device=DEVICE)
    print_vram_status("Nach Einbindung")

# 4. API-Server vorbereiten und starten
    global global_tokenizer, global_model
    global_tokenizer = tokenizer
    global_model = model

    print("\n[4/4] Starte OpenAI-kompatiblen API-Server...")
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((API_HOST, API_PORT), OpenOriginPagingAPI)
    
    print("=" * 65)
    print(f"✅ SYSTEM BEREIT! WebUI-API lauscht auf http://{API_HOST}:{API_PORT}/v1")
    print("=" * 65)
    
    server.serve_forever()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABBRUCH] Durch Benutzer beendet.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR]:\n{traceback.format_exc()}")
        sys.exit(1)
