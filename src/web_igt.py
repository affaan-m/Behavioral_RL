from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from igt_env import IGTEnvironment
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize IGT environment
env = IGTEnvironment()

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IGT Experiment</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; text-align: center; }
            #experiment { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Iowa Gambling Task</h1>
        <div id="experiment">
            <!-- Content will be loaded dynamically -->
        </div>
        <script src="/static/js/main.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/start")
async def start_experiment(data: Dict[str, Any]):
    session_id = f"{datetime.now().isoformat()}_{data.get('age')}_{data.get('gender')}"
    state = env.reset()[0]
    
    return {
        'session_id': session_id,
        'state': state.tolist() if hasattr(state, 'tolist') else state,
        'total_money': env.total_money
    }

@app.post("/api/step")
async def step(data: Dict[str, Any]):
    action = data.get('action')
    if action is None:
        return JSONResponse(
            status_code=400,
            content={'error': 'No action provided'}
        )
    
    state, reward, done, _, info = env.step(action)
    
    return {
        'state': state.tolist() if hasattr(state, 'tolist') else state,
        'reward': reward,
        'done': done,
        'total_money': info['total_money']
    }

@app.post("/api/save")
async def save_results(data: Dict[str, Any]):
    try:
        return {
            'success': True,
            'message': 'Data saved successfully'
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': str(e)
            }
        )

if __name__ == '__main__':
    uvicorn.run("web_igt:app", host="0.0.0.0", port=8000, reload=True) 