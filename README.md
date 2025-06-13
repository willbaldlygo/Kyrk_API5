# Spine Copilot – Single‑File Edition

Minimal Streamlit app that lets you chat with Summer Spine race data using OpenAI.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run spine_app.py
```

Then upload:
* Race results CSV (required)
* Course records CSV (optional)
* Facebook gallery HTML (optional)