# Personality-Adaptive Squat Coach: Fine-Tuning Llama 3.2 3B with SFT

Fine-tune **Llama 3.2 3B Instruct** (QLoRA, via [Unsloth](https://github.com/unslothai/unsloth)) to act as a short-form squat coach. Given posture scores from pose estimation and the user's **Big Five personality traits**, the model returns a brief spoken-style cue (at most 18 words) whose tone fits the person.

This repo is part of human-robot interaction (HRI) research on personality-adaptive robotic exercise coaching. The exported model is small enough to run locally (GGUF, Q4_K_M), for example on a robot or edge device.

## How it works

```
Pose estimation  ->  Spine / Knee / Ankle scores (0-1) + detected issues
Big Five survey  ->  O, C, E, A, N traits (0-1)
                         |
                         v
          Fine-tuned Llama 3.2 3B (LoRA, 4-bit)
                         |
                         v
   "Great consistency! Just bend forward a bit more."
```

### Example

**Input**

```
Spine: 0.8, Knee: 1, Ankle: 1. Openness: 0.7, Conscientiousness: 0.8,
Extraversion: 0.6, Agreeableness: 0.7, Neuroticism: 0.3. Issues: Bend forward.
```

**Output**

```
Great consistency! Just bend forward a bit more.
```

## Repository contents

| File | Description |
|---|---|
| `finetune1.ipynb` | End-to-end Colab notebook: install, load model, format data, train, test, export to GGUF |
| `perdata.json` | Training set of 697 instruction/input/output examples (Alpaca-style) |
| `README.md` | This file |

## Dataset

`perdata.json` is a JSON list of records with three fields:

| Field | Content |
|---|---|
| `instruction` | The task prompt, identical in every record: *"You are a concise squat coach... Max 18 words."* |
| `input` | Spine, Knee and Ankle scores (0-1), the five Big Five traits (0-1), and a comma-separated list of detected issues |
| `output` | The coaching cue (2-16 words, average about 7) |

**Issue labels** (counts across the 697 examples): knee crossing toe (343), shallow squat (257), bend forward (230), deep squat (211), bend backward (167), good form (29). One example can carry several issues.

**Things to know**
- 648 unique input/output pairs; the rest are duplicates.
- No train/validation/test split is included. The notebook trains on all 697 examples.

## Training setup

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| Method | QLoRA supervised fine-tuning (`trl.SFTTrainer`) |
| LoRA | r = 32, alpha = 32, dropout = 0.05, all attention and MLP projections |
| Max sequence length | 2048 |
| Batch size | 2 per device x 4 gradient accumulation = 8 effective |
| Steps / LR | 300 steps (about 3.4 epochs), 1e-4, cosine schedule, 10 warmup steps |
| Optimizer | `adamw_8bit`, weight decay 0.05 |
| Seed | 3407 |
| Hardware | Free Colab T4 GPU (about 15 GB) |
| Export | GGUF, `q4_k_m` |

## Quick start

1. Open `finetune1.ipynb` in [Google Colab](https://colab.research.google.com/) (File -> Upload notebook).
2. Select a GPU runtime: **Runtime -> Change runtime type -> T4 GPU**.
3. Upload `perdata.json` to the Colab file panel, in the same folder as the notebook.
4. Run all cells in order. The notebook will:
   - install Unsloth and dependencies,
   - load Llama 3.2 3B with LoRA adapters,
   - format the data with the Llama 3 chat template,
   - train for 300 steps,
   - run a sample inference,
   - export a quantized GGUF file and download it.

### Run the exported model locally

The GGUF file works with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.com/) or LM Studio. With llama.cpp:

```bash
llama-cli -m Llama-3.2-3B-Instruct.Q4_K_M.gguf \
  -sys "You are a concise squat coach. Given spine, knee, and ankle conditions (0-1), Big Five personality traits (0-1), and a list of issues, identify the priority area and give 1-2 short cues. Max 18 words." \
  -p "Spine: 0.3, Knee: 0.9, Ankle: 0.9. Openness: 0.68, Conscientiousness: 0.7, Agreeableness: 0.7, Extraversion: 0.9, Neuroticism: 0.7. Issues: Bend forward, Shallow squat, knee crossing toe."
```

## Limitations

- **Small, narrow dataset.** About 700 examples, many near-duplicates, covering one exercise and five issue types.
- **No quantitative evaluation yet.** There is no held-out test set, so the repo does not claim measured accuracy or personality-fit results.
- **Prompt format differs between training and testing.** Training wraps the instruction and input together in the user turn, while the sample inference cell sends the instruction as the system message. Outputs may therefore differ from what the training format would give.
- **Personality adaptation is learned from the examples**, not validated against user outcomes such as trust or adherence.
- **Not medical or physiotherapy advice.** The model gives general form cues only.

## Roadmap

- [ ] Use one identical prompt format for training and inference
- [ ] Train/validation/test split and an evaluation script (word-limit compliance, issue coverage, tone consistency)
- [ ] Compare against the un-tuned base model and a personality-agnostic baseline
- [ ] Publish the LoRA adapter or GGUF on Hugging Face
- [ ] Integrate with a live pose-estimation pipeline

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for fast QLoRA training
- [Meta Llama 3.2](https://www.llama.com/) base model (use is subject to the Llama 3.2 Community License)
- Hugging Face `transformers`, `trl`, `peft`, `datasets`

## License

Add a license for your own code and data (for example MIT), and keep the Llama 3.2 Community License terms for any distributed model weights.

## Citation

If this work is useful to your research, please cite it:

```bibtex
@misc{squatcoach_llama32_sft,
  title  = {Personality-Adaptive Squat Coach: Fine-Tuning Llama 3.2 3B with SFT},
  author = {YOUR NAME},
  year   = {2026},
  url    = {https://github.com/YOUR-USERNAME/YOUR-REPO}
}
```
