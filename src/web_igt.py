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
import json
from pathlib import Path
from supabase import create_client, Client
import uuid
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

logger.info("Checking Supabase credentials...")
logger.info(f"SUPABASE_URL present: {'SUPABASE_URL' in os.environ}")
logger.info(f"SUPABASE_KEY present: {'SUPABASE_KEY' in os.environ}")

supabase = None
if not supabase_url or not supabase_key:
    logger.error("Supabase credentials missing!")
else:
    try:
        logger.info(f"Initializing Supabase with URL: {supabase_url[:20]}...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Test connection
        test_result = supabase.table('participants').select("*").limit(1).execute()
        if hasattr(test_result, 'data'):
            logger.info("Supabase connection test successful")
        else:
            logger.error("Supabase connection test failed: unexpected response format")
            supabase = None
    except Exception as e:
        logger.error(f"Error initializing Supabase client: {str(e)}")
        supabase = None

if not supabase:
    logger.warning("Supabase integration disabled due to initialization errors")

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
            metrics = self.calculate_metrics() if done else None
            
            return np.zeros(4), reward, done, False, {
                "total_money": self.total_money,
                "metrics": metrics,
                "step_count": self.step_count
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
    def __init__(self, timeout: int = 3600):  # 1 hour timeout
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.timeout = timeout
        self.last_cleanup = time.time()
        logger.info("Session manager initialized")
    
    def create_session(self) -> tuple[str, IGTEnv]:
        """Create a new session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        env = IGTEnv()
        self._sessions[session_id] = {
            'env': env,
            'created_at': time.time(),
            'last_activity': time.time(),
            'uuid': str(uuid.uuid4()),  # Generate UUID for Supabase
            'lock': threading.Lock(),  # Add thread lock for synchronization
            'is_saving': False  # Flag to prevent duplicate saves
        }
        logger.info(f"Created new session: {session_id} with UUID: {self._sessions[session_id]['uuid']}")
        return session_id, env
    
    def get_session(self, session_id: str) -> Optional[IGTEnv]:
        """Get an existing session with thread safety"""
        session = self._sessions.get(session_id)
        if session is None:
            logger.error(f"Session not found: {session_id}")
            logger.info(f"Active sessions: {list(self._sessions.keys())}")
            return None
            
        with session['lock']:
            current_time = time.time()
            if current_time - session['last_activity'] > self.timeout and not session['is_saving']:
                logger.error(f"Session {session_id} expired")
                self.remove_session(session_id)
                return None
                
            session['last_activity'] = current_time
            logger.info(f"Retrieved session: {session_id}, step count: {session['env'].step_count}")
            return session['env']
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            logger.info(f"Removing session: {session_id}, final step count: {session['env'].step_count}")
            del self._sessions[session_id]
        else:
            logger.warning(f"Attempted to remove non-existent session: {session_id}")
    
    def list_sessions(self) -> list[Dict[str, Any]]:
        """List all active sessions with their details"""
        current_time = time.time()
        return [{
            'id': sid,
            'step_count': session['env'].step_count,
            'total_money': session['env'].total_money,
            'age': current_time - session['created_at'],
            'last_activity': current_time - session['last_activity']
        } for sid, session in self._sessions.items()]
    
    def cleanup_expired(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired = []
        for sid, session in self._sessions.items():
            if current_time - session['last_activity'] > self.timeout:
                expired.append(sid)
        
        for sid in expired:
            logger.info(f"Cleaning up expired session: {sid}")
            self.remove_session(sid)

# Create global session manager
session_manager = SessionManager()

class StepRequest(BaseModel):
    session_id: str
    action: int
    reaction_time: Optional[float] = None

@app.get("/api/debug")
async def debug_info():
    """Get debug information about active sessions"""
    try:
        sessions = session_manager.list_sessions()
        return {
            "active_sessions": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error in debug_info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start")
async def start_session():
    """Start a new session"""
    try:
        # Clean up any existing sessions for this client
        session_id, env = session_manager.create_session()
        logger.info(f"Starting new session: {session_id}")
        state, _ = env.reset()
        
        response_data = {
            'session_id': session_id,
            'state': state.tolist(),
            'total_money': env.total_money,
            'step_count': 0
        }
        logger.info(f"Session started successfully: {session_id}, data: {response_data}")
        return response_data
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/step")
async def step(data: StepRequest):
    """Process a step in the session"""
    try:
        session_id = data.session_id
        logger.info(f"Step request for session {session_id}")
        
        # Clean up expired sessions
        session_manager.cleanup_expired()
        
        # Get session
        session = session_manager._sessions.get(session_id)
        if session is None:
            logger.error(f"Invalid session: {session_id}")
            raise HTTPException(
                status_code=400, 
                detail="Invalid or expired session"
            )
        
        with session['lock']:
            env = session['env']
            # Process action
            action = data.action
            reaction_time = data.reaction_time
            logger.info(f"Processing action {action} for session {session_id} at step {env.step_count}")
            
            try:
                state, reward, done, _, info = env.step(action, reaction_time)
                logger.info(f"Step completed: reward={reward}, done={done}, money={info['total_money']}, step={info['step_count']}")
            except Exception as e:
                logger.error(f"Error in step execution: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing step: {str(e)}")
            
            # Prepare response
            response_data = {
                'state': state.tolist(),
                'reward': reward,
                'done': done,
                'total_money': info['total_money'],
                'step_count': info['step_count']
            }
            
            # Handle completion
            if done and not session['is_saving']:
                session['is_saving'] = True  # Prevent duplicate saves
                logger.info(f"Session {session_id} complete at step {info['step_count']}")
                try:
                    metrics = info.get('metrics', {})
                    if not metrics:
                        logger.warning(f"No metrics available for completed session {session_id}")
                    
                    response_data['metrics'] = metrics
                    
                    # Save to Supabase using UUID
                    if supabase:
                        try:
                            session_data = {
                                'id': session['uuid'],  # Use UUID for Supabase
                                'session_id': session_id,  # Store original session ID as separate field
                                'timestamp': datetime.now().isoformat(),
                                'metrics': json.dumps(metrics),
                                'history': json.dumps(env.history)
                            }
                            
                            logger.info(f"Saving session data to Supabase: {session_id} (UUID: {session['uuid']})")
                            result = supabase.table('participants').insert(session_data).execute()
                            if hasattr(result, 'data'):
                                logger.info(f"Successfully saved session {session_id} to Supabase")
                                response_data['save_status'] = 'success'
                            else:
                                logger.error(f"Unexpected Supabase response for session {session_id}: {result}")
                                response_data['save_status'] = 'error'
                                response_data['save_error'] = 'Unexpected response from Supabase'
                        except Exception as e:
                            logger.error(f"Error saving to Supabase: {str(e)}")
                            response_data['save_status'] = 'error'
                            response_data['save_error'] = str(e)
                    else:
                        logger.warning("Supabase client not initialized, skipping data save")
                        response_data['save_status'] = 'error'
                        response_data['save_error'] = 'Supabase client not initialized'
                except Exception as e:
                    logger.error(f"Error handling session completion: {str(e)}")
                    response_data['save_status'] = 'error'
                    response_data['save_error'] = str(e)
                finally:
                    # Clean up session only after successful save
                    if response_data.get('save_status') == 'success':
                        try:
                            session_manager.remove_session(session_id)
                            logger.info(f"Session {session_id} cleaned up after successful save")
                        except Exception as e:
                            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
                    else:
                        logger.warning(f"Keeping session {session_id} active due to save failure")
                        session['is_saving'] = False  # Reset save flag if save failed
            
            return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in step: {str(e)}")
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
                #trial-counter {
                    font-size: 20px;
                    margin: 15px;
                    color: #666;
                }
                #feedback { 
                    margin: 20px; 
                    font-size: 18px;
                    text-align: center;
                    min-height: 27px;
                }
                #completion-message {
                    display: none;
                    margin: 20px;
                    padding: 20px;
                    background-color: #e8f5e9;
                    border-radius: 5px;
                    text-align: center;
                    font-size: 20px;
                    color: #2e7d32;
                }
                #stats { 
                    margin-top: 30px; 
                    text-align: left;
                    background-color: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>Iowa Gambling Task</h1>
            <div id="experiment">
                <div id="money">Total Money: $2000</div>
                <div id="trial-counter">Trials Remaining: 100</div>
                <div id="decks">
                    <button class="deck-button" onclick="selectDeck(0)">Deck A</button>
                    <button class="deck-button" onclick="selectDeck(1)">Deck B</button>
                    <button class="deck-button" onclick="selectDeck(2)">Deck C</button>
                    <button class="deck-button" onclick="selectDeck(3)">Deck D</button>
                </div>
                <div id="feedback"></div>
                <div id="completion-message"></div>
                <div id="stats"></div>
            </div>

            <script>
                let startTime = new Date();
                let trialCount = 0;
                let trialsRemaining = 100;
                let sessionId = null;
                let isProcessing = false;
                let totalMoney = 2000;
                let isComplete = false;
                
                async function initSession() {
                    updateButtons(false);
                    document.getElementById('feedback').textContent = 'Initializing...';
                    
                    try {
                        console.log('Requesting new session...');
                        const response = await fetch('/api/start', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({})
                        });
                        
                        if (!response.ok) {
                            const errorData = await response.text();
                            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData}`);
                        }
                        
                        const data = await response.json();
                        console.log('Session initialized with data:', data);
                        
                        if (!data.session_id) {
                            throw new Error('No session ID received from server');
                        }
                        
                        sessionId = data.session_id;
                        totalMoney = data.total_money;
                        trialCount = data.step_count;
                        trialsRemaining = 100 - trialCount;
                        isComplete = false;
                        updateDisplay();
                        startTime = new Date();
                        updateButtons(true);
                        document.getElementById('feedback').textContent = '';
                        
                        console.log(`Session ${sessionId} initialized successfully`);
                    } catch (error) {
                        console.error('Error initializing session:', error);
                        document.getElementById('feedback').textContent = 'Error starting game. Please refresh the page.';
                        updateButtons(false);
                    }
                }
                
                function updateDisplay() {
                    const moneyDisplay = document.getElementById('money');
                    moneyDisplay.textContent = `Total Money: $${totalMoney}`;
                    const counterDisplay = document.getElementById('trial-counter');
                    counterDisplay.textContent = `Trials Remaining: ${trialsRemaining}`;
                }
                
                function updateButtons(enabled) {
                    const buttons = document.querySelectorAll('.deck-button');
                    buttons.forEach(btn => {
                        btn.disabled = !enabled || isComplete;
                    });
                }
                
                async function selectDeck(deck) {
                    if (!sessionId) {
                        console.log('No session, initializing...');
                        await initSession();
                        return;
                    }
                    
                    if (isProcessing || isComplete) {
                        return;
                    }
                    
                    isProcessing = true;
                    updateButtons(false);
                    
                    const reactionTime = (new Date() - startTime) / 1000;
                    
                    try {
                        console.log(`Sending choice for session ${sessionId}:`, { deck, reactionTime, trialCount });
                        const response = await fetch('/api/step', {
                            method: 'POST',
                            headers: { 
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ 
                                session_id: sessionId,
                                action: deck,
                                reaction_time: reactionTime
                            })
                        });
                        
                        if (!response.ok) {
                            const errorData = await response.text();
                            if (response.status === 400 && errorData.includes('Invalid or expired session')) {
                                console.log('Session expired, reinitializing...');
                                await initSession();
                                return;
                            }
                            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData}`);
                        }
                        
                        const data = await response.json();
                        console.log(`Received response for session ${sessionId}:`, data);
                        
                        totalMoney = data.total_money;
                        trialCount = data.step_count;
                        trialsRemaining = 100 - trialCount;
                        updateDisplay();
                        
                        const feedback = document.getElementById('feedback');
                        feedback.textContent = `Reward: ${data.reward > 0 ? '+' : ''}${data.reward}`;
                        feedback.style.color = data.reward >= 0 ? 'green' : 'red';
                        
                        if (data.done) {
                            isComplete = true;
                            const completionDiv = document.getElementById('completion-message');
                            completionDiv.style.display = 'block';
                            
                            let message = '<h2>Experiment Complete!</h2>';
                            if (data.save_status === 'success') {
                                message += '<p>Thank you for participating! Your data has been saved successfully.</p>';
                            } else if (data.save_status === 'error') {
                                message += `<p>Thank you for participating!</p><p style="color: red;">Note: There was an error saving your data: ${data.save_error}</p>`;
                                console.error('Error saving data:', data.save_error);
                            } else {
                                message += '<p>Thank you for participating!</p>';
                            }
                            completionDiv.innerHTML = message;
                            
                            if (data.metrics) {
                                const stats = document.getElementById('stats');
                                stats.innerHTML = `
                                    <h3>Performance Summary:</h3>
                                    <p>Final Money: $${totalMoney}</p>
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
                            }
                            updateButtons(false);
                            sessionId = null;  // Clear session only after successful completion
                        }
                        
                        startTime = new Date();
                        
                    } catch (error) {
                        console.error('Error processing choice:', error);
                        document.getElementById('feedback').textContent = 'Error processing choice. Please try again.';
                        updateDisplay();
                    } finally {
                        isProcessing = false;
                        if (sessionId && !isComplete) {
                            updateButtons(true);
                        }
                    }
                }
                
                // Initialize when page loads
                window.onload = async function() {
                    console.log('Page loaded, initializing session...');
                    await initSession();
                };
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

def save_session_data(session_id: str, data: dict):
    """Save session data to a JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = data_dir / f"session_{timestamp}_{session_id}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved session data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving session data: {e}")
        return False

if __name__ == '__main__':
    uvicorn.run("web_igt:app", host="0.0.0.0", port=8000, reload=True) 