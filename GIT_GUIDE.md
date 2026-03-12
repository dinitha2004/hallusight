# HalluSight — Git Guide & Commit History Plan

## First Time Setup (do this ONCE)

```bash
# 1. Install git (if not already installed)
brew install git

# 2. Set your identity (use your real name and university email)
git config --global user.name "Chalani"
git config --global user.email "your@email.com"

# 3. Go to your project folder
cd ~/Desktop/hallusight

# 4. Initialise git repository
git init

# 5. Add everything and make your FIRST commit
git add .
git commit -m "🎉 Initial commit: HalluSight project structure and Day 1 setup"
```

---

## Connect to GitHub (do this ONCE)

```bash
# 1. Go to https://github.com → click "New Repository"
# 2. Name it: hallusight
# 3. Set to Private (for your research)
# 4. Do NOT add README or .gitignore (you already have them)
# 5. Copy the URL it gives you, then run:

git remote add origin https://github.com/YOUR_USERNAME/hallusight.git
git branch -M main
git push -u origin main
```

---

## DAY 1 — Commits to Make

```bash
# After setup.sh runs successfully
git add setup.sh requirements.txt .gitignore
git commit -m "⚙️ Day 1: Add project setup script and requirements"

# After llm_loader.py is working
git add model/
git commit -m "🤖 Day 1: Add LLM loader with hidden state extraction (TBG module)"

# After hidden_state_extractor.py is working
git add modules/hidden_state_extractor.py
git commit -m "🧠 Day 1: Add HiddenStateExtractor - variance, magnitude, similarity analysis"

# After test_day1.py passes ALL tests
git add test_day1.py
git commit -m "✅ Day 1 COMPLETE: All tests passing - LLaMA loads, hidden states extracted"

# Push everything to GitHub
git push
```

---

## DAY 2 — Commits to Make

```bash
# After SEP probe is written
git add modules/semantic_entropy_probe.py
git commit -m "🔬 Day 2: Add Semantic Entropy Probe (SEP) - logistic regression classifier"

# After HalluShift is written
git add modules/distribution_shift.py
git commit -m "📊 Day 2: Add Distribution Shift Analyzer (HalluShift) - cosine + Wasserstein"

# After Feature Clipping + TSV
git add modules/feature_clipping.py
git commit -m "✂️ Day 2: Add Feature Clipping and TSV preprocessing module"

# After Token Risk Scorer
git add modules/token_risk_scorer.py
git commit -m "⚠️ Day 2: Add Token Risk Scorer - combines SEP and HalluShift scores"

# After EAT Detector
git add modules/eat_detector.py
git commit -m "🎯 Day 2: Add EAT Detector - identifies factual tokens (names, dates, numbers)"

# After Span Aggregator
git add modules/span_aggregator.py
git commit -m "🔗 Day 2: Add Span Aggregator - groups risky tokens into hallucinated spans"

# After Overall Scorer
git add modules/overall_scorer.py
git commit -m "📈 Day 2: Add Overall Hallucination Score Calculator"

# After probe training script
git add train/
git commit -m "🏋️ Day 2: Add probe training script with TruthfulQA dataset support"

# After test_day2.py passes
git add test_day2.py
git commit -m "✅ Day 2 COMPLETE: All detection modules built and tested"

git push
```

---

## DAY 3 — Commits to Make

```bash
# After pipeline.py connects all modules
git add modules/pipeline.py
git commit -m "🔄 Day 3: Add full detection pipeline - connects all 10 modules end-to-end"

# After Flask API
git add api/
git commit -m "🌐 Day 3: Add Flask REST API - /detect and /health endpoints"

# After HTML frontend
git add frontend/
git commit -m "🎨 Day 3: Add web frontend with real-time hallucination highlighting UI"

# After evaluation metrics
git add evaluation/
git commit -m "📏 Day 3: Add evaluation script - precision, recall, F1, latency metrics"

# After full system demo works end-to-end
git add .
git commit -m "🚀 Day 3 COMPLETE: Full HalluSight system working - demo ready"

git push
```

---

## Everyday Commands (use these often)

```bash
# See what files have changed
git status

# See all your commits so far (your history)
git log --oneline

# See what changed inside a file
git diff modules/semantic_entropy_probe.py

# Save all current changes with a message
git add .
git commit -m "your message here"

# Upload to GitHub
git push

# If you broke something and want to go back to last commit
git checkout -- filename.py
```

---

## Good Commit Message Rules

| Prefix | When to use |
|--------|-------------|
| `🎉 Initial commit:` | Very first commit |
| `✅ Day X COMPLETE:` | After passing daily test |
| `🤖` | LLM / model related code |
| `🧠` | Detection / analysis modules |
| `🔬` | Machine learning / probe code |
| `📊` | Statistics / scoring |
| `🌐` | API code |
| `🎨` | Frontend / UI |
| `🐛 Fix:` | Fixing a bug |
| `📝 Docs:` | Updating README or comments |
| `⚙️` | Setup / configuration |
| `🚀` | Final / deployment |

---

## After Your Project is Done — Final Commits

```bash
# Add your README
git add README.md
git commit -m "📝 Add project README with research description and setup guide"

# Add your evaluation results
git add results/
git commit -m "📊 Add evaluation results - precision/recall/F1 scores"

# Final clean-up
git add .
git commit -m "🏁 Final: Research prototype complete - HalluSight v1.0"

git push
```

---

## View Your Beautiful Commit History

```bash
git log --oneline --graph --all
```

This shows a tree of all your commits — great to include as a screenshot in your research report!
