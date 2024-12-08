from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
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

# Mount static files if they exist
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Simple state management (for testing)
class SimpleIGTEnv:
    def __init__(self):
        self.total_money = 2000
        self.step_count = 0
        
    def reset(self):
        self.total_money = 2000
        self.step_count = 0
        return np.zeros(4), {}
        
    def step(self, action):
        self.step_count += 1
        # Simplified reward structure
        rewards = {
            0: (-100, 50),   # Deck A
            1: (-250, 100),  # Deck B
            2: (-50, 50),    # Deck C
            3: (-50, 50)     # Deck D
        }
        loss, gain = rewards[action]
        reward = gain if np.random.random() > 0.5 else loss
        self.total_money += reward
        done = self.step_count >= 100
        return np.zeros(4), reward, done, False, {"total_money": self.total_money}

# Initialize environment
env = SimpleIGTEnv()

@app.get("/")
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IGT Experiment</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; text-align: center; }
            #experiment { margin-top: 20px; text-align: center; }
            button { margin: 10px; padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h1>Iowa Gambling Task</h1>
        <div id="experiment">
            <div id="money">Total Money: $2000</div>
            <div id="decks">
                <button onclick="selectDeck(0)">Deck A</button>
                <button onclick="selectDeck(1)">Deck B</button>
                <button onclick="selectDeck(2)">Deck C</button>
                <button onclick="selectDeck(3)">Deck D</button>
            </div>
            <div id="feedback"></div>
        </div>
        <script>
            async function selectDeck(deck) {
                try {
                    const response = await fetch('/api/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ action: deck })
                    });
                    const data = await response.json();
                    document.getElementById('money').textContent = `Total Money: $${data.total_money}`;
                    document.getElementById('feedback').textContent = 
                        `Reward: ${data.reward > 0 ? '+' : ''}${data.reward}`;
                    if (data.done) {
                        alert('Experiment complete! Thank you for participating.');
                    }
                } catch (error) {
                    console.error('Error:', error);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/start")
async def start_experiment(data: Dict[str, Any]):
    try:
        session_id = f"{datetime.now().isoformat()}"
        state, _ = env.reset()
        return {
            'session_id': session_id,
            'state': state.tolist(),
            'total_money': env.total_money
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/step")
async def step(data: Dict[str, Any]):
    try:
        action = data.get('action')
        if action is None:
            raise HTTPException(status_code=400, detail="No action provided")
        
        state, reward, done, _, info = env.step(action)
        return {
            'state': state.tolist(),
            'reward': reward,
            'done': done,
            'total_money': info['total_money']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("web_igt:app", host="0.0.0.0", port=8000, reload=True) 