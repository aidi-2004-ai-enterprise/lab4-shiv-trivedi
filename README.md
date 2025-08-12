# Lab 4 — Bankruptcy Prediction (README) 🏢💰

**Author:** Shivkumar Trivedi 
**Lab 4:** Bankruptcy Prediction through Binary Classification 📊🔍  

---

## 📦 What's in this folder

- `assets/` — folder containing charts and visualizations 📊
- `data.csv` — 6,819 companies with ~96 financial features 📈
- `dataset-snapshot.py` — quick scan: shape, class balance, extreme-value flags + two display-only charts 🔍
- `lab4-report.md` — my graded answers to the decision areas (WHY-focused) 📝
- `psi_results.csv` — PSI numbers exported during my run 📄
- `PSI-analysis.py` — simple PSI check (train vs test) with one bar chart 📊
- `README.md` — this file 📖
- `requirements.txt` — Python dependencies for the project 🐍

---

## 🎯 What I investigated

- 🤖 Picking a baseline + two stronger models  
- 🔧 How to prep data for each model (keep it minimal)  
- ⚖️ What to do about the tiny positive class  
- 📊 Outliers vs. real distress signals  
- 🎯 Train/test representativeness (PSI)  
- 📏 Metrics that match rare-event reality  
- 🔍 Interpretability that business can follow

---

## 🔎 Things the data made obvious

### ⚠️ Severe imbalance
- **220 bankrupt** vs **6,599 non-bankrupt** → ~**3.2%** positive (~**1:29** ratio) 📉
- Accuracy alone is useless here; recall and calibrated probabilities matter ⚖️

### 🚨 A few "impossible" values
- **13 features** with absurd maximums (e.g., **Current Ratio ~ 2.75B**, **Revenue/Share ~ 3.02B**) 💸  
- These look like entry errors; most other ratios sit in sensible ranges ✅

### ✅ Train/test looks like the same population
- PSI across tested features was **< 0.1** (no meaningful drift) 📊  
- Stratified sampling preserved the base rate 🎯

---

## ✅ Decisions (driven by evidence)

### 🤖 Model Selection
**Models:** Logistic Regression (baseline), Random Forest, XGBoost  
**Why:** Baseline is explainable; trees/boosting capture thresholds & interactions common in financial ratios

### 🔧 Pre-processing Strategy  
**Approach:** Scale **only** for Logistic Regression; same simple pipeline for fairness  
**Why:** Trees don't need scaling; consistent prep keeps comparisons honest

### ⚖️ Imbalance Handling
**Strategy:** Stratified splits; class weighting; SMOTE only if needed (within train folds)  
**Why:** Raise recall without fabricating too much noise

### 📊 Outlier Treatment
**Approach:** Cap only obviously wrong values; keep legitimate extremes  
**Why:** Distress looks extreme—don't delete the signal we care about

### 🎯 Sampling Bias Assessment
**Method:** PSI train vs test and monitor over time  
**Why:** Validate today's split and catch drift before performance drops

### 📏 Evaluation Metrics
**Strategy:** Use **predict_proba**; primary **PR-AUC**; also ROC-AUC, F1; check calibration  
**Why:** Imbalanced reality + decisions use probabilities, not just labels

### 🔍 Explainability Approach
**Methods:** SHAP for the best tree model; coefficients for the baseline  
**Why:** Clear, case-level reasons for stakeholders and regulators

---

## 📸 Screens I captured for the report

- 📊 Class distribution bar chart (shows the ~1:29 split)  
- 🚨 "Implausible extremes — top features" bar chart  
- ✅ PSI table + PSI bar chart (all green / below 0.1)  
- 💻 Terminal snippets: dataset shape and flagged features

---

## 🧠 Evidence → Action (quick map)

| Evidence from analysis 🔍            | What it told me 💡                     | Action I took ⚡                            |
|--------------------------------------|-----------------------------------------|----------------------------------------------|
| ~1:29 class ratio                    | Rare positives; accuracy is misleading  | Stratified CV; class weights; SMOTE if needed|
| 13 absurd max values                 | Entry errors, not finance reality       | Cap only those; keep real extremes           |
| PSI < 0.1 (train vs test)            | Split is representative                 | Proceed with stratified K-fold               |
| Ratios mostly in sensible ranges     | Minimal prep is enough                  | Scale LR only; trees on raw ratios           |

---

## 🎓 Key Learning Outcomes

### 📊 Data Understanding
- Learned to identify and handle severe class imbalance in financial data
- Developed skills in detecting data quality issues vs. legitimate extreme values
- Gained experience with Population Stability Index (PSI) for drift detection

### 🤖 Model Strategy
- Understood when to use different preprocessing approaches for different model types
- Learned the importance of starting simple before adding complexity
- Developed intuition for model selection based on problem characteristics

### 📏 Evaluation Insights
- Recognized why accuracy is misleading for imbalanced datasets
- Learned to prioritize probability-based metrics over simple classification
- Understood the business need for model explainability in financial contexts

---

## 🚀 Future Improvements

### 🔄 Model Enhancement
- Experiment with ensemble methods combining all three models
- Test advanced sampling techniques (ADASYN, BorderlineSMOTE)
- Explore threshold optimization techniques for cost-sensitive learning

### 📊 Data Engineering
- Implement automated outlier detection pipelines
- Develop real-time PSI monitoring for production deployment
- Create feature engineering based on domain expertise

### 🏭 Production Readiness
- Build model monitoring dashboard for performance tracking
- Implement A/B testing framework for model comparison
- Develop automated retraining pipelines based on drift detection

---

## 🙋 Final note

I kept the pipeline small 🔧, the "why" front-and-center 💡, and the visuals as proof I actually ran the analysis 📊. If something changes (new data or drift), PSI and calibration checks are my early warning system ⚠️.

**Remember:** In bankruptcy prediction, it's better to be cautious and flag potential risks than to miss actual bankruptcies! 🛡️💼