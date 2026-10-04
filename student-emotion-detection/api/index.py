# Vercel entry point for Streamlit app
from streamlit.web import bootstrap

def app(environ, start_response):
    bootstrap.run(
        "app.py",
        command_line=None,
        args=[],
        flag_options={},
        is_hello=False,
    )

# For Vercel Python runtime (WSGI)
application = app
handler = app
