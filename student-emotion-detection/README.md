# EduSense

EduSense is a Streamlit app for exploring student facial-expression classification
and classroom-level engagement summaries.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
python -m streamlit run app.py
```

Open <http://localhost:8501>. The app needs a trained model checkpoint at
`artifacts/models/emotion_cnn.pt` for meaningful predictions. Without one, it
shows a warning and uses untrained weights.

## Deploy to Render

The repository-root `render.yaml` defines the Streamlit web service. In Render,
create a Blueprint from this repository and select the branch containing that
file. Render builds from `student-emotion-detection`, installs
`requirements-cpu.txt`, and starts Streamlit on Render's assigned port. The
free service may spin down while idle.

Once the service is live at
`https://edusense-student-emotion-detection.onrender.com`, the Vercel project
redirects visitors to the full Streamlit app. The standalone Vercel preview
does not run Streamlit or access a camera.
