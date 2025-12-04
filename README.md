# SAP EWA Traffic-Light Chatbot (System A1C)

This project automates analysis of SAP EarlyWatch Alert (EWA) reports for system **A1C**.

Features:

- Convert `.DOC` → `.DOCX`
- Parse traffic-light colors (overall and per section) from EWA Word reports
- Build a historical color matrix (Jan–Nov 2025)
- Train & test a baseline ML model (RandomForest) to predict overall status
- Compute deviations between latest historical report and a newly uploaded report
- Risk scoring (Low / Medium / High) based on deteriorations
- Streamlit web app with:
  - History table
  - Trend charts
  - File upload for new EWA report
  - Deviation view
  - Chatbot using OpenAI GPT to answer questions and explain risks

## 1. Setup

```bash
git clone <this-repo>
cd ewa-ewa-bot
python -m venv .venv
.\.venv\Scripts\activate   # on Windows
pip install -r requirements.txt
