"""Launch the local demo with installed dependencies or this workspace's staged copy."""
import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if importlib.util.find_spec('streamlit') is None:
    staged = root / '.local-deps'
    if not (staged / 'streamlit').is_dir():
        raise SystemExit('Install dependencies first: python -m pip install -r requirements.txt')
    sys.path.insert(0, str(staged))

from streamlit.web import cli

if __name__ == '__main__':
    sys.argv = ['streamlit', 'run', str(root/'app.py'), '--global.developmentMode=false',
                '--server.headless=true', '--server.address=127.0.0.1', '--browser.gatherUsageStats=false']
    cli.main()
