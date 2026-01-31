# 🎓 Interview Kit: User Engagement & Churn Prediction

## 🚀 Resume Project Summary (Copy-Paste Ready)
**Project Title:** End-to-End Customer Churn Prediction Platform (MLOps & Full Stack)

**Description:**
Designed and deployed a full-stack Machine Learning platform to predict customer churn in the Telecommunications sector. Architected a decoupled inference system using **FastAPI** and **React-style Vanilla JS**, powered by an **XGBoost/Logistic Regression** backend.
-   **Engineered** a robust data pipeline with **Stratified Sampling** to handle 74:26 class imbalance and **Pandera** for strict schema validation.
-   **Optimized** model for **Recall (0.79)** to minimize revenue loss from missed churners (False Negatives), achieving an **ROC-AUC of 0.84**.
-   **Implemented** a modular "Champion-Challenger" training framework and exposed real-time predictions via a REST API with latency <100ms.
-   **Deployed** an interactive dashboard categorizing users into clear Risk Tiers (Low/Medium/High) for actionable business insights.

**Tech Stack:** Python, Scikit-Learn, FastAPI, Pandas, Docker (Ready), JavaScript, HTML5.

---

## 🗣️ Technical Talking Points (Deep Dive)

### 1. Handling Class Imbalance
**Interviewer:** "The dataset is imbalanced. How did you handle that?"
**Answer:** 
"I used a multi-layered approach. First, I implemented **Stratified Splitting** in the data pipeline to ensure the test set represents the real-world distribution (26% churn). For the model itself, I used `class_weight='balanced'` in Logistic Regression, which inversely weights classes, and `scale_pos_weight` in XGBoost. This forced the model to pay more attention to the minority class, significantly boosting Recall from ~0.5 to ~0.79."

### 2. Architecture Decisions
**Interviewer:** "Why did you separate the Inference Engine from the API?"
**Answer:**
"I followed MLOps maturity principles. By creating a dedicated `src.inference.predictor` class, I decoupled the core ML logic from the FastAPI web layer. This makes testing easier (I can unit test the predictor without mocking HTTP requests) and allows me to swap the backend framework (e.g., to Flask or Django) without touching the ML code."

### 3. Business Metric Selection
**Interviewer:** "Why did you optimize for Recall?"
**Answer:**
"In Churn prediction, a False Negative (missing a churner) is much more expensive than a False Positive. If we miss a churner, we lose their Lifetime Value (LTV). If we falsely flag a loyal user, the cost is just a small retention discount. Therefore, I tuned the model threshold to maximize Recall, accepting a slight drop in Precision as a calculated business trade-off."

### 4. Handling Missing Data (TotalCharges)
**Interviewer:** "How did you handle the missing values in TotalCharges?"
**Answer:**
"I discovered that `TotalCharges` was blank only for customers with **0 tenure**. Functionally, this means they are brand new and haven't been billed yet. Instead of dropping them (which loses data) or imputing with the mean (which is factually wrong), I imputed them with **0**. This reflects the reality that they have paid nothing yet."

---

## 🌟 Future Improvements (If asked)
-   **Model Monitoring:** Integrating tools like **Prometheus/Grafana** to track data drift over time.
-   **CI/CD:** Automating the training pipeline via **GitHub Actions** on new data commits.
-   **A/B Testing:** Deploying the Challenger (XGBoost) alongside the Baseline to compare real-world performance.
