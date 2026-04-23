# 🧠 PFA College LLM

A synthetic data generation and fine-tuning pipeline to build a **Psychological First Aid (PFA) peer support chatbot** tailored for college students in distress. The system simulates realistic counselling conversations using LLMs, evaluates their quality, and fine-tunes a lightweight model for deployment.

---

## 🎯 Project Goal

College students face unique, high-intensity stressors — placements, mid-sems, hostel conflicts, financial pressure — that often go unsupported. This project builds a domain-specific LLM that responds to student distress using **WHO-aligned PFA principles**: validating feelings, staying non-clinical, focusing on the present, and gently linking to support systems.

---

## 🗂️ Project Structure

```
pfa-college-llm/
│
├── data/
│   ├── raw/
│   │   └── college_stressors.csv       # Seed stressor scenarios (Category, Scenario, Intensity)
│   ├── synthetic_raw/                  # Auto-generated simulated conversations (JSON)
│   ├── synthetic_filtered/             # LLM-judge filtered, high-quality conversations
│   └── final_training/                 # HuggingFace-ready fine-tuning dataset
│
├── prompts/
│   ├── system_pfa_agent.txt            # System prompt for the PFA responder agent
│   ├── system_client.txt               # System prompt for the simulated distressed student
│   └── judge_rubric.txt                # Rubric used by the LLM judge for evaluation
│
├── scripts/
│   ├── generator.py                    # Runs the multi-turn conversation simulation
│   ├── evaluator.py                    # Scores conversations against the PFA rubric
│   └── format_hf.py                    # Converts filtered data to HuggingFace format
│
├── notebooks/
│   ├── 01_data_generation.ipynb        # Interactive notebook for data generation
│   ├── 02_llm_evaluation.ipynb         # Interactive notebook for LLM-as-judge evaluation
│   └── 03_unsloth_finetuning.ipynb     # Unsloth fine-tuning on Google Colab (GPU)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Pipeline Overview

```
college_stressors.csv
        │
        ▼
[ 1. Data Generation ]  ←  generator.py
   Two-agent simulation: PFA Responder ↔ Distressed Student (Groq / LLaMA 3.3 70B)
        │
        ▼  synthetic_raw/simulated_conversations.json
        │
[ 2. LLM Evaluation ]  ←  evaluator.py
   LLM-as-judge scores each conversation against PFA rubric; filters low-quality
        │
        ▼  synthetic_filtered/
        │
[ 3. Format for HuggingFace ]  ←  format_hf.py
   Converts to instruction-tuning format (system / user / assistant turns)
        │
        ▼  final_training/
        │
[ 4. Fine-Tuning ]  ←  03_unsloth_finetuning.ipynb
   Unsloth fine-tuning on a small base model (Colab GPU)
```

---

## 🤖 Agent Design

### PFA Responder Agent (`prompts/system_pfa_agent.txt`)
Follows strict WHO Psychological First Aid guidelines:
- **Listen & Validate** — acknowledges feelings without judgment
- **No Diagnosis** — avoids clinical labels like "anxiety disorder" or "depression"
- **No Probing** — never asks "why" questions or forces explanation
- **Present Focus** — keeps the conversation grounded in the immediate situation
- **Link to Support** — gently surfaces the student's own support systems

### Simulated Student Agent (`prompts/system_client.txt`)
Realistic college student persona driven by a stressor row from the CSV:
- Uses authentic college vocabulary: *placements, mid-sems, hostel, GPA*
- Emotional intensity mapped to `Intensity` field: **High** → panicked short sentences; **Medium** → venting, frustrated
- Requires 3–4 turns of validation before showing any relief — preventing artificially easy conversations
- Keeps responses short (≤ 3 sentences) to mimic real messaging patterns

---

## 📊 Stressor Seed Data

The seed file `data/raw/college_stressors.csv` contains 8 scenario archetypes across 5 categories:

| Category | Example Scenario | Intensity |
|---|---|---|
| Academic | Failing a core engineering module before mid-sems | High |
| Career | Rejected from a major tech internship while peers got selected | High |
| Social | Feeling isolated in the hostel, struggling to make friends | Medium |
| Financial | Stress about paying next semester tuition fees | High |
| Burnout | Exhaustion from constant coding and maintaining GPA | Medium |

New scenarios can be added directly to this CSV to expand dataset coverage.

---

## 🚀 Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/Aviral0105/pfa-college-llm.git
cd pfa-college-llm
pip install -r requirements.txt
```

### 2. Set up your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Generate synthetic conversations

```python
from scripts.generator import run_simulation
import os
from dotenv import load_dotenv

load_dotenv()
run_simulation(api_key=os.getenv("GROQ_API_KEY"), num_turns=3)
```

Or use the interactive notebook: `notebooks/01_data_generation.ipynb`

### 4. Evaluate and filter

Run `notebooks/02_llm_evaluation.ipynb` to score conversations with the LLM judge and filter out low-quality samples.

### 5. Fine-tune

Open `notebooks/03_unsloth_finetuning.ipynb` in **Google Colab** (GPU runtime required) to fine-tune using Unsloth.

---

## 📦 Requirements

```
google-generativeai
groq
pandas
python-dotenv
```

Fine-tuning dependencies (Colab):
```
unsloth
trl
transformers
datasets
```

---

## 🔑 API Keys Required

| Service | Purpose | Get Key |
|---|---|---|
| [Groq](https://console.groq.com) | Fast LLaMA 3.3 70B inference for simulation & evaluation | Free tier available |
| [Google Generative AI](https://aistudio.google.com) | Optional: alternative model for evaluation | Free tier available |

---

## 📝 Notes

- **Test mode** in `generator.py` currently processes only the first 4 stressors (`if i < 4`). Remove this condition to run all scenarios.
- Synthetic data directories (`synthetic_raw/`, `synthetic_filtered/`, `final_training/`) are excluded from Git via `.gitignore` and should be synced via Google Drive.
- The `judge_rubric.txt` and `evaluator.py` are scaffolded and ready to be populated with scoring criteria aligned with the PFA directives.

---

## 🤝 Contributing

Contributions to expand stressor categories, improve PFA prompt alignment, or add evaluation metrics are welcome. Please open an issue before submitting a PR.

---

## ⚠️ Disclaimer

This project is a research prototype. The generated chatbot is **not a substitute for professional mental health support**. If you or someone you know is in crisis, please contact a qualified mental health professional or a crisis helpline.
