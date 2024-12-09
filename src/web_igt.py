import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
    def step(self, action: int, reaction_time: Optional[float] = None) -> tuple:
        try:
            if action not in self.deck_properties:
                raise ValueError(f"Invalid action: {action}")
                
            self.step_count += 1
            self.history['deck_choices'].append(action)
            self.history['timestamps'].append(datetime.now().isoformat())
            if reaction_time is not None:
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
        except Exception as e:
            logger.error(f"Error in IGTEnv.step: {str(e)}")
            raise
    
    def calculate_metrics(self):
        try:
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
        except Exception as e:
            logger.error(f"Error in calculate_metrics: {str(e)}")
            raise

class SessionManager:
    def __init__(self, timeout: int = 300):  # 5 minutes timeout
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.timeout = timeout
        self.last_cleanup = time.time()
    
    def create_session(self) -> tuple[str, IGTEnv]:
        self.cleanup_expired()
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        env = IGTEnv()
        self.sessions[session_id] = {
            'env': env,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        return session_id, env
    
    def get_session(self, session_id: str) -> Optional[IGTEnv]:
        if session_id not in self.sessions:
            return None
        self.sessions[session_id]['last_activity'] = time.time()
        return self.sessions[session_id]['env']
    
    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired(self):
        current_time = time.time()
        if current_time - self.last_cleanup < 60:  # Only cleanup once per minute
            return
            
        expired = []
        for sid, session in self.sessions.items():
            if current_time - session['last_activity'] > self.timeout:
                expired.append(sid)
        
        for sid in expired:
            logger.info(f"Removing expired session {sid}")
            del self.sessions[sid]
        
        self.last_cleanup = current_time
    
    def get_active_sessions(self) -> Dict[str, Any]:
        self.cleanup_expired()
        return {
            sid: {
                'total_money': session['env'].total_money,
                'step_count': session['env'].step_count,
                'history_length': len(session['env'].history['deck_choices']),
                'age': time.time() - session['created_at'],
                'last_activity': time.time() - session['last_activity']
            }
            for sid, session in self.sessions.items()
        }

# Create session manager
session_manager = SessionManager()

class StepRequest(BaseModel):
    session_id: str
    action: int
    reaction_time: Optional[float] = None

@app.get("/api/debug")
async def debug_info():
    """Endpoint to check active sessions and their state"""
    try:
        sessions = session_manager.get_active_sessions()
        return {
            "active_sessions": len(sessions),
            "session_ids": list(sessions.keys()),
            "session_states": sessions
        }
    except Exception as e:
        logger.error(f"Error in debug_info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start")
async def start_session():
    try:
        session_id, env = session_manager.create_session()
        logger.info(f"New session started: {session_id}")
        state, _ = env.reset()
        return {
            'session_id': session_id,
            'state': state.tolist(),
            'total_money': env.total_money
        }
    except Exception as e:
        logger.error(f"Error in start_session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/step")
async def step(data: StepRequest):
    try:
        session_id = data.session_id
        logger.info(f"Step request for session {session_id}")
        
        env = session_manager.get_session(session_id)
        if not env:
            logger.error(f"Invalid session: {session_id}")
            raise HTTPException(status_code=400, detail="Invalid or expired session")
        
        action = data.action
        reaction_time = data.reaction_time
        
        logger.info(f"Processing action {action} for session {session_id}")
        state, reward, done, _, info = env.step(action, reaction_time)
        
        logger.info(f"Step result: reward={reward}, done={done}, total_money={info['total_money']}")
        
        response_data = {
            'state': state.tolist(),
            'reward': reward,
            'done': done,
            'total_money': info['total_money']
        }
        
        if done:
            logger.info(f"Session {session_id} complete")
            response_data['metrics'] = info['metrics']
            session_manager.remove_session(session_id)
        
        return response_data
        
    except ValueError as e:
        logger.error(f"Value error in step: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def index():
    """Serve the main IGT interface"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>IGT Experiment</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }
                h1 { 
                    color: #333; 
                    text-align: center; 
                    margin-bottom: 30px;
                }
                #experiment { 
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .deck-button { 
                    margin: 10px; 
                    padding: 20px 40px; 
                    font-size: 18px;
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    transition: all 0.3s ease;
                }
                .deck-button:hover { 
                    background-color: #45a049; 
                    transform: translateY(-2px);
                }
                .deck-button:disabled {
                    background-color: #cccccc;
                    cursor: not-allowed;
                    transform: none;
                }
                #money { 
                    font-size: 24px; 
                    margin: 20px;
                    text-align: center;
                    color: #2196F3;
                }
                #feedback { 
                    margin: 20px; 
                    font-size: 18px;
                    text-align: center;
                    min-height: 27px;
                }
                #debug { 
                    margin: 20px; 
                    font-size: 14px; 
                    color: #666; 
                    text-align: left;
                    background-color: #f8f8f8;
                    padding: 10px;
                    border-radius: 5px;
                    max-height: 200px;
                    overflow-y: auto;
                }
                #stats { 
                    margin-top: 30px; 
                    text-align: left;
                    background-color: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                }
                .debug-message {
                    margin: 5px 0;
                    font-family: monospace;
                }
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
                <div id="debug"></div>
                <div id="stats"></div>
            </div>

            <script>
                let startTime = new Date();
                let trialCount = 0;
                let sessionId = null;
                let isProcessing = false;
                let totalMoney = 2000;
                
                function updateDebug(message) {
                    const debugDiv = document.getElementById('debug');
                    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'debug-message';
                    messageDiv.textContent = `[${timestamp}] ${message}`;
                    debugDiv.appendChild(messageDiv);
                    debugDiv.scrollTop = debugDiv.scrollHeight;
                    console.log(`[${timestamp}] ${message}`);
                }
                
                function updateDisplay() {
                    const moneyDisplay = document.getElementById('money');
                    moneyDisplay.textContent = `Total Money: $${totalMoney}`;
                    updateDebug(`Updated display: Total Money = $${totalMoney}`);
                }
                
                function updateButtons(enabled) {
                    const buttons = document.querySelectorAll('.deck-button');
                    buttons.forEach(btn => {
                        btn.disabled = !enabled;
                    });
                    updateDebug(`Buttons ${enabled ? 'enabled' : 'disabled'}`);
                }
                
                async function checkServerState() {
                    try {
                        const response = await fetch('/api/debug');
                        const data = await response.json();
                        updateDebug(`Server state: ${JSON.stringify(data)}`);
                        
                        if (sessionId) {
                            if (!data.session_states[sessionId]) {
                                updateDebug('Session expired or not found! Reinitializing...');
                                await initSession();
                                return;
                            }
                            
                            const serverMoney = data.session_states[sessionId].total_money;
                            if (serverMoney !== totalMoney) {
                                updateDebug(`Money mismatch! Local: ${totalMoney}, Server: ${serverMoney}`);
                                totalMoney = serverMoney;
                                updateDisplay();
                            }
                        }
                    } catch (error) {
                        updateDebug(`Error checking server state: ${error.message}`);
                    }
                }
                
                async function initSession() {
                    updateDebug('Initializing session...');
                    updateButtons(false);
                    
                    try {
                        const response = await fetch('/api/start', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({})
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        sessionId = data.session_id;
                        totalMoney = data.total_money;
                        updateDebug(`Session initialized: ${sessionId}`);
                        updateDisplay();
                        startTime = new Date();
                        updateButtons(true);
                        await checkServerState();
                    } catch (error) {
                        updateDebug(`Error initializing: ${error.message}`);
                        document.getElementById('feedback').textContent = 'Error starting game. Please refresh the page.';
                    }
                }
                
                async function selectDeck(deck) {
                    updateDebug(`Selecting deck ${deck}...`);
                    if (!sessionId) {
                        updateDebug('No active session! Reinitializing...');
                        await initSession();
                        return;
                    }
                    if (isProcessing) {
                        updateDebug('Still processing previous action');
                        return;
                    }
                    
                    isProcessing = true;
                    updateButtons(false);
                    
                    const reactionTime = (new Date() - startTime) / 1000;
                    trialCount++;
                    
                    try {
                        updateDebug(`Sending choice: deck ${deck}, reaction time: ${reactionTime}s`);
                        const response = await fetch('/api/step', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ 
                                session_id: sessionId,
                                action: deck,
                                reaction_time: reactionTime
                            })
                        });
                        
                        if (!response.ok) {
                            if (response.status === 400 && (await response.text()).includes('Invalid or expired session')) {
                                updateDebug('Session expired, reinitializing...');
                                await initSession();
                                return;
                            }
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        updateDebug(`Received response: ${JSON.stringify(data)}`);
                        
                        totalMoney = data.total_money;
                        updateDisplay();
                        
                        const feedback = document.getElementById('feedback');
                        feedback.textContent = `Reward: ${data.reward > 0 ? '+' : ''}${data.reward}`;
                        feedback.style.color = data.reward >= 0 ? 'green' : 'red';
                        
                        if (data.metrics) {
                            const stats = document.getElementById('stats');
                            stats.innerHTML = `
                                <h3>Performance Summary:</h3>
                                <p>Trials Completed: ${trialCount}</p>
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
                                updateDebug('Experiment complete');
                                alert('Experiment complete! Thank you for participating.');
                                sessionId = null;
                                return;
                            }
                        }
                        
                        startTime = new Date();
                        await checkServerState();
                        
                    } catch (error) {
                        updateDebug(`Error: ${error.message}`);
                        document.getElementById('feedback').textContent = 'Error processing choice. Please try again.';
                        
                        // Check if session is still valid
                        const stateResponse = await fetch('/api/debug');
                        const stateData = await stateResponse.json();
                        if (!stateData.session_states[sessionId]) {
                            updateDebug('Session lost, reinitializing...');
                            await initSession();
                        }
                    } finally {
                        isProcessing = false;
                        if (sessionId) {
                            updateButtons(true);
                        }
                    }
                }
                
                // Initialize session when page loads
                window.onload = function() {
                    updateDebug('Page loaded, initializing...');
                    initSession();
                    
                    // Set up periodic state check
                    setInterval(checkServerState, 2000);  // Check more frequently
                };
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("web_igt:app", host="0.0.0.0", port=8000, reload=True) 