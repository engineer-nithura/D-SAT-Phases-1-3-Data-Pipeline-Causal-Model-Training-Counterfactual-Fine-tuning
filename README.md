# D-SAT: Dynamic Scene-Action Transformer
### A Causal World Model for Video Understanding

D-SAT is an independent research project building a world model that learns *why* things change in video, not just *what* changes. Rather than classifying actions or describing scenes in isolation, D-SAT explicitly models causal state transitions — given a scene graph **Gₜ** and an action, predict the resulting scene graph **Gₜ₊₁**.

---

## Motivation

Current video understanding frameworks have a shared blind spot: they learn correlations, not causes.

- **Action recognition** models identify the verb ("cutting") but ignore the agent, the object, and what changes as a result.
- **Scene graph generators** capture spatial relationships in a single frame but can't model how those relationships evolve.
- **Vision-language models** (VLMs) produce rich descriptions from learned patterns, but have no explicit mechanism for causal reasoning — they can't reliably answer "what would happen if...?"

D-SAT is built to close this gap by learning the state transition function:

```
G_{t+1} = p(G_t, action)
```

---

## Framework Overview

The pipeline is split into three conceptual components:

**1. Perception Module (frozen)**
A pre-trained DINOv2 ViT backbone with a graph generator head converts raw video frames into structured JSON scene graphs. This module is treated as a fixed, off-the-shelf component.

**2. Causal Transition Model (trainable)**
A LoRA-adapted Gemma 3 model that takes the current scene graph and an action description as text input, and autoregressively generates the predicted next scene graph. Trained with cross-entropy loss against ground-truth graph sequences.

**3. Counterfactual Reasoning Layer**
A fine-tuning stage on manually curated counterfactual examples that pushes the model from pattern-matching toward genuine causal understanding.

---

## Project Status

This project is being built independently in three completed phases, with more to come.

### Phase 1 — Automated Causal Dataset Generation (YouCook2 Pipeline)
> *Notebook: `DSAT_Phase1_YouCook2.ipynb`*

Built an end-to-end data pipeline to bootstrap a structured causal dataset from cooking videos:

- Loads YouCook2 annotations (via HuggingFace `lmms-lab/YouCook2`) — 3,180 captioned segments across 414 videos
- Downloads video segments with `yt-dlp` and extracts start/end frame pairs using `ffmpeg`
- Calls **Gemini 2.0 Flash** as a Teacher VLM to generate structured JSON scene graphs (Gₜ and Gₜ₊₁) from each frame pair
- Runs consistency filtering to discard malformed or causally inconsistent triplets
- Saves the final `(Gₜ, action_text, Gₜ₊₁)` triplets as a `.jsonl` dataset

**Output:** `triplets.jsonl` — a structured causal dataset ready for training.

---

### Phase 2 — Causal Transition Model Training
> *Notebook: `DSAT_Phase2_Training.ipynb`*

Fine-tuned Gemma 3 (2B instruct) on the Phase 1 dataset to learn graph-to-graph causal translation:

- Formats each triplet into a text prompt: current graph + action → predicted next graph
- Attaches **LoRA adapters** (via `peft`) for parameter-efficient fine-tuning on an A100 GPU
- Trains using cross-entropy loss on the predicted graph token sequence
- Evaluates checkpoints using **Graph Edit Distance (GED)** on held-out examples
- Includes an inference demo and saves the final LoRA adapter checkpoint

**Output:** `lora_adapter/` — a fine-tuned causal transition model checkpoint.

---

### Phase 3 — Counterfactual Fine-tuning
> *Notebook: `DSAT_Phase3_Counterfactual.ipynb`*

Loads the best Phase 2 checkpoint and fine-tunes on a curated set of counterfactual examples:

- Counterfactuals test "what if the action were different?" — e.g., same starting scene, but "add salt" vs. "add sugar" should produce distinct, correctly differing outcomes
- Manually written triplets designed to expose and fix pattern-matching shortcuts
- Evaluates on both **counterfactual accuracy** and original GED to ensure no regression
- Saves the final counterfactually-augmented checkpoint

**Output:** `lora_adapter_cf/` — the most causally-aware model checkpoint to date.

---

## Roadmap

The following phases are planned to complete the full D-SAT MVP:

### Phase 4 — Scale Dataset Generation
Run the Phase 1 pipeline at full scale across the complete YouCook2 training split (and potentially additional video datasets) to produce a significantly larger and more diverse triplet dataset for a full training run.

### Phase 5 — Full Training Run & Evaluation
Train the Causal Transition Model on the scaled dataset with a proper train/validation/test split. Run a comprehensive final evaluation covering graph similarity metrics, causal accuracy, and counterfactual robustness benchmarks.

### Phase 6 — Perception Module Integration
Connect the frozen DINOv2-based Perception Module to the trained Causal Transition Model, enabling end-to-end inference directly from raw video frames rather than pre-extracted graphs.

### Phase 7 — Demo & Report
Build an interactive demo of the full pipeline (video in → predicted future state graph out) and write up the complete a final report covering methodology, results, and limitations.

## Repository Structure

```
dsat/
├── DSAT_Phase1_YouCook2.ipynb       # Data pipeline
├── DSAT_Phase2_Training.ipynb       # Model training
├── DSAT_Phase3_Counterfactual.ipynb # Counterfactual fine-tuning
```

---

## Background & References

- **YouCook2 Dataset** — [lmms-lab/YouCook2](https://huggingface.co/datasets/lmms-lab/YouCook2)
- **Gemma 3** — [google/gemma-3-2b-it](https://huggingface.co/google/gemma-3-2b-it)
- **DINOv2** — Oquab et al., 2023
- **LoRA** — Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
- **Graph Edit Distance** — used as the primary structural similarity metric for scene graph evaluation
