from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from igt_env import IGTEnvironment
import json
from datetime import datetime
from pathlib import Path

app = FastAPI()

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize IGT environment
env = IGTEnvironment()

@app.get("/")
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IGT Experiment</title>
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
async def start_experiment(request: Request):
    data = await request.json()
    session_id = f"{datetime.now().isoformat()}_{data.get('age')}_{data.get('gender')}"
    state = env.reset()[0]
    
    return JSONResponse({
        'session_id': session_id,
        'state': state.tolist() if hasattr(state, 'tolist') else state,
        'total_money': env.total_money
    })

@app.post("/api/step")
async def step(request: Request):
    data = await request.json()
    action = data.get('action')
    if action is None:
        return JSONResponse({'error': 'No action provided'}, status_code=400)
    
    state, reward, done, _, info = env.step(action)
    
    return JSONResponse({
        'state': state.tolist() if hasattr(state, 'tolist') else state,
        'reward': reward,
        'done': done,
        'total_money': info['total_money']
    })

@app.post("/api/save")
async def save_results(request: Request):
    try:
        data = await request.json()
        return JSONResponse({
            'success': True,
            'message': 'Data saved successfully'
        })
    except Exception as e:
        return JSONResponse({
            'success': False,
            'message': str(e)
        }, status_code=500)

if __name__ == '__main__':
    uvicorn.run("web_igt:app", host="0.0.0.0", port=8000, reload=True) 