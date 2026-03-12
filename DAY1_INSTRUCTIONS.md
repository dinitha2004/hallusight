# HalluSight — Day 1 Instructions

## What to do RIGHT NOW (copy-paste each command)

---

### STEP 1 — Open Terminal on your Mac
Press `Cmd + Space`, type **Terminal**, press Enter.

---

### STEP 2 — Install Homebrew (Mac package manager)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
When it asks for your Mac password, type it (you won't see it typing — that's normal).

---

### STEP 3 — Install Python 3.10
```bash
brew install python@3.10
python3 --version
```
You should see: `Python 3.10.x`

---

### STEP 4 — Install VS Code
Go to: **https://code.visualstudio.com**
Download the Mac version. Drag it to your Applications folder.

---

### STEP 5 — Copy the project folder to your Mac
Download or copy the `hallusight` folder to your Desktop, then:
```bash
cd ~/Desktop/hallusight
```

---

### STEP 6 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
You will see `(venv)` appear at the start of your Terminal line.
This means the virtual environment is active. ✓

> **IMPORTANT:** Every time you open a NEW Terminal window, run:
> `source venv/bin/activate` before any other command.

---

### STEP 7 — Install all libraries
```bash
bash setup.sh
```
This installs everything automatically. It takes 5-10 minutes.

---

### STEP 8 — Login to HuggingFace (for LLaMA)
If you are using LLaMA (recommended):
1. Go to **https://huggingface.co** and create a free account
2. Go to **https://huggingface.co/meta-llama/Llama-3.2-1B** and request access
3. Go to Settings > Access Tokens and create a token
4. Run:
```bash
huggingface-cli login
```
Paste your token when prompted.

**If you can't get LLaMA access today:**
Open `model/llm_loader.py` in VS Code and change line:
```python
MODEL_NAME = "facebook/opt-1.3b"
```
This works identically — no login needed.

---

### STEP 9 — Run the Day 1 test
```bash
python test_day1.py
```

**First run:** Will download ~2-3 GB model weights (one time only).
**After that:** Loads in ~30 seconds from your Mac cache.

---

### WHAT SUCCESS LOOKS LIKE

```
======================================================
  HalluSight — Day 1 Test
======================================================
TEST 1: Checking all imports...
  ✓ torch
  ✓ numpy
  ✓ scikit-learn
  ✓ scipy
  ✓ spacy
  ✓ flask

TEST 2: Checking spaCy English model...
  ✓ spaCy loaded
  ✓ Test sentence entities: [('Albert Einstein', 'PERSON'), ('1879', 'DATE'), ('Germany', 'GPE')]

TEST 3: Loading the LLM model...
  ✓ Model loaded successfully!

TEST 4: Generating text with hidden state extraction...
  ✓ Full response:   'Paris'
  ✓ Tokens list:     ['Paris', ...]

TEST 5: Checking hidden state vectors...
  ✓ Shape of first hidden state: (2048,)
  ✓ Hidden states count matches token count

TEST 6: Running HiddenStateExtractor analysis...
  ✓ HiddenStateExtractor works correctly!

🎉 DAY 1 COMPLETE — ALL TESTS PASSED!
```

---

### COMMON ERRORS AND FIXES

| Error | Fix |
|-------|-----|
| `No module named 'transformers'` | Run `source venv/bin/activate` first |
| `LLaMA access denied` | Change MODEL_NAME to `"facebook/opt-1.3b"` |
| `Killed` (Mac runs out of memory) | Change MODEL_NAME to `"facebook/opt-125m"` |
| `spacy model not found` | Run `python -m spacy download en_core_web_sm` |
| `huggingface_hub not found` | Run `pip install huggingface_hub` |

---

### FILES CREATED TODAY

```
hallusight/
├── setup.sh                          ← Run once to install everything
├── requirements.txt                  ← List of all libraries
├── test_day1.py                      ← Run to verify Day 1 works ✓
├── model/
│   ├── __init__.py
│   └── llm_loader.py                 ← Loads LLaMA + collects hidden states ✓
└── modules/
    ├── __init__.py
    └── hidden_state_extractor.py     ← Analyses hidden state vectors ✓
```

---

When `test_day1.py` passes all tests → **Day 1 is done. Start Day 2!**
