import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from web_igt import app

# For Vercel
app = app 