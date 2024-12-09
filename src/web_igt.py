from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json
import os
import logging
from supabase import Client, create_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")

if not supabase_url or not supabase_key:
    logger.error("Supabase credentials missing!")
    logger.info(f"SUPABASE_URL present: {'SUPABASE_URL' in os.environ}")
    logger.info(f"SUPABASE_KEY present: {'SUPABASE_KEY' in os.environ}")
else:
    logger.info(f"Initializing Supabase with URL: {supabase_url[:20]}...")

try:
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Supabase client: {str(e)}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
active_sessions = {}

class IGTEnv:
    def __init__(self):
        self.total_money = 2000
        self.step_count = 0
        self.history = {
            'deck_choices': [],
            'rewards': [],
            'total_money': [self.total_money],
            'reaction_times': [],
            'timestamps': []
        }
        self.deck_properties = {
            0: {'win': 100, 'loss': -250, 'loss_prob': 0.4},  # Deck A
            1: {'win': 100, 'loss': -1250, 'loss_prob': 0.1}, # Deck B
            2: {'win': 50, 'loss': -50, 'loss_prob': 0.5},    # Deck C
            3: {'win': 50, 'loss': -250, 'loss_prob': 0.1}    # Deck D
        }
        
    def reset(self):
        self.total_money = 2000
        self.step_count = 0
        self.history = {
            'deck_choices': [],
            'rewards': [],
            'total_money': [self.total_money],
            'reaction_times': [],
            'timestamps': []
        }
        return np.zeros(4), {}
        
    def step(self, action, reaction_time=None):
        self.step_count += 1
        self.history['deck_choices'].append(action)
        self.history['timestamps'].append(datetime.now().isoformat())
        if reaction_time:
            self.history['reaction_times'].append(reaction_time)
            
        deck = self.deck_properties[action]
        if np.random.random() < deck['loss_prob']:
            reward = deck['loss']
        else:
            reward = deck['win']
            
        self.total_money += reward
        self.history['rewards'].append(reward)
        self.history['total_money'].append(self.total_money)
        
        done = self.step_count >= 100
        return np.zeros(4), reward, done, False, {
            "total_money": self.total_money,
            "metrics": self.calculate_metrics() if done else None
        }
    
    def calculate_metrics(self):
        choices = self.history['deck_choices']
        rewards = self.history['rewards']
        
        # Calculate advantageous choices (C+D) ratio
        advantageous = sum(1 for c in choices[-20:] if c in [2, 3]) / 20
        
        # Calculate risk-seeking after losses
        risk_seeking = []
        for i in range(1, len(choices)):
            if rewards[i-1] < 0:  # After a loss
                risk_seeking.append(1 if choices[i] in [0, 1] else 0)
        risk_seeking_ratio = np.mean(risk_seeking) if risk_seeking else 0
        
        # Calculate deck preferences
        total_choices = len(choices)
        deck_preferences = {
            f"deck_{chr(65+i)}": choices.count(i) / total_choices
            for i in range(4)
        }
        
        # Calculate learning metrics
        first_20_choices = choices[:20]
        last_20_choices = choices[-20:]
        learning_progress = {
            'early_advantageous': sum(1 for c in first_20_choices if c in [2, 3]) / 20,
            'late_advantageous': sum(1 for c in last_20_choices if c in [2, 3]) / 20
        }
        
        return {
            'total_money': self.total_money,
            'advantageous_ratio': advantageous,
            'risk_seeking_after_loss': risk_seeking_ratio,
            'deck_preferences': deck_preferences,
            'mean_reaction_time': np.mean(self.history['reaction_times']) if self.history['reaction_times'] else 0,
            'learning_progress': learning_progress
        }

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
            .deck-button { 
                margin: 10px; 
                padding: 20px 40px; 
                font-size: 18px;
                cursor: pointer;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            .deck-button:hover { background-color: #45a049; }
            #money { font-size: 24px; margin: 20px; }
            #feedback { margin: 20px; font-size: 18px; }
            #stats { margin-top: 30px; text-align: left; }
        </style>
    </head>
    <body>
        <h1>Iowa Gambling Task</h1>
        <div id="experiment">
            <div id="money">Total Money: $2000</div>
            <div id="decks">
                <button class="deck-button" onclick="selectDeck(0)">Deck A</button>
                <button class="deck-button" onclick="selectDeck(1)">Deck B</button>
                <button class="deck-button" onclick="selectDeck(2)">Deck C</button>
                <button class="deck-button" onclick="selectDeck(3)">Deck D</button>
            </div>
            <div id="feedback"></div>
            <div id="stats"></div>
        </div>
        <script>
            let startTime;
            let trialCount = 0;
            let sessionId = null;
            
            async function initSession() {
                try {
                    const response = await fetch('/api/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    });
                    const data = await response.json();
                    sessionId = data.session_id;
                    startTime = new Date();
                } catch (error) {
                    console.error('Error initializing session:', error);
                }
            }
            
            async function selectDeck(deck) {
                if (!sessionId) {
                    console.error('No active session');
                    return;
                }
                
                if (!startTime) startTime = new Date();
                const reactionTime = (new Date() - startTime) / 1000;
                trialCount++;
                
                try {
                    const response = await fetch('/api/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            session_id: sessionId,
                            action: deck,
                            reaction_time: reactionTime
                        })
                    });
                    const data = await response.json();
                    document.getElementById('money').textContent = `Total Money: $${data.total_money}`;
                    document.getElementById('feedback').textContent = 
                        `Reward: ${data.reward > 0 ? '+' : ''}${data.reward}`;
                    
                    if (data.metrics) {
                        const stats = document.getElementById('stats');
                        stats.innerHTML = `
                            <h3>Performance Summary:</h3>
                            <p>Advantageous Choices (C+D): ${(data.metrics.advantageous_ratio * 100).toFixed(1)}%</p>
                            <p>Risk-Seeking After Loss: ${(data.metrics.risk_seeking_after_loss * 100).toFixed(1)}%</p>
                            <p>Average Reaction Time: ${data.metrics.mean_reaction_time.toFixed(2)}s</p>
                            <p>Deck Preferences:</p>
                            <ul>
                                ${Object.entries(data.metrics.deck_preferences)
                                    .map(([deck, pref]) => `<li>${deck}: ${(pref * 100).toFixed(1)}%</li>`)
                                    .join('')}
                            </ul>
                        `;
                        
                        if (data.done) {
                            alert('Experiment complete! Thank you for participating.');
                            sessionId = null;  // End session
                        }
                    }
                    
                    startTime = new Date();  // Reset for next trial
                } catch (error) {
                    console.error('Error:', error);
                }
            }
            
            // Initialize session when page loads
            initSession();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/start")
async def start_session():
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    active_sessions[session_id] = IGTEnv()
    logger.info(f"New session started: {session_id}")
    state, _ = active_sessions[session_id].reset()
    return {
        'session_id': session_id,
        'state': state.tolist(),
        'total_money': active_sessions[session_id].total_money
    }

@app.post("/api/step")
async def step(data: Dict[str, Any]):
    try:
        session_id = data.get('session_id')
        if not session_id or session_id not in active_sessions:
            raise HTTPException(status_code=400, detail="Invalid or expired session")
            
        action = data.get('action')
        reaction_time = data.get('reaction_time')
        
        if action is None:
            raise HTTPException(status_code=400, detail="No action provided")
        
        env = active_sessions[session_id]
        state, reward, done, _, info = env.step(action, reaction_time)
        
        response_data = {
            'state': state.tolist(),
            'reward': reward,
            'done': done,
            'total_money': info['total_money']
        }
        
        if done:
            metrics = info['metrics']
            experiment_data = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'history': env.history,
                'metrics': metrics
            }
            
            logger.info(f"Session {session_id} complete. Saving data...")
            
            try:
                result = supabase.table('participants').insert(experiment_data).execute()
                logger.info(f"Data saved for session {session_id}")
                response_data['metrics'] = metrics
                # Clean up session
                del active_sessions[session_id]
            except Exception as e:
                logger.error(f"Error saving to Supabase: {str(e)}")
                response_data['metrics'] = metrics
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in step endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("web_igt:app", host="0.0.0.0", port=8000, reload=True) 