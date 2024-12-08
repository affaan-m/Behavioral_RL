import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from igt_env import IGTEnvironment

class IGTExperiment:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Iowa Gambling Task Experiment")
        self.root.geometry("800x600")
        
        self.env = IGTEnvironment()
        self.state = None
        self.history = {
            'deck_choices': [],
            'rewards': [],
            'total_money': [],
            'reaction_times': []
        }
        self.start_time = None
        self.participant_info = {}
        
        self.setup_gui()
        
    def setup_gui(self):
        # Welcome and instructions
        self.welcome_frame = ttk.Frame(self.root, padding="20")
        self.welcome_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.welcome_frame, text="Welcome to the Iowa Gambling Task", 
                 font=('Arial', 16, 'bold')).pack(pady=10)
        
        instructions = """
        In this task, you will be presented with 4 decks of cards (A, B, C, and D).
        
        On each turn:
        1. Select a deck by clicking on it
        2. You will win or lose money based on the card drawn
        3. Try to win as much money as possible
        
        Some decks are better than others.
        You are free to switch from one deck to another at any time.
        
        The task will continue for 100 trials.
        """
        ttk.Label(self.welcome_frame, text=instructions, wraplength=600).pack(pady=20)
        
        # Participant info collection
        info_frame = ttk.LabelFrame(self.welcome_frame, text="Participant Information", padding="10")
        info_frame.pack(pady=20)
        
        ttk.Label(info_frame, text="Age:").grid(row=0, column=0, padx=5, pady=5)
        self.age_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.age_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Gender:").grid(row=1, column=0, padx=5, pady=5)
        self.gender_var = tk.StringVar()
        ttk.Combobox(info_frame, textvariable=self.gender_var,
                    values=['Male', 'Female', 'Other', 'Prefer not to say']).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(self.welcome_frame, text="Start Experiment", 
                  command=self.start_experiment).pack(pady=20)
        
        # Game frame (initially hidden)
        self.game_frame = ttk.Frame(self.root, padding="20")
        
        # Status display
        self.status_frame = ttk.LabelFrame(self.game_frame, text="Status", padding="10")
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.money_label = ttk.Label(self.status_frame, text="Total Money: $2000")
        self.money_label.pack(side=tk.LEFT, padx=20)
        
        self.trials_label = ttk.Label(self.status_frame, text="Trials Remaining: 100")
        self.trials_label.pack(side=tk.RIGHT, padx=20)
        
        # Deck buttons
        self.decks_frame = ttk.Frame(self.game_frame, padding="20")
        self.decks_frame.pack(expand=True)
        
        self.deck_buttons = []
        for i, deck in enumerate(['A', 'B', 'C', 'D']):
            btn = ttk.Button(self.decks_frame, text=f"Deck {deck}",
                           width=20, command=lambda x=i: self.select_deck(x))
            btn.pack(side=tk.LEFT, padx=10)
            self.deck_buttons.append(btn)
        
        # Feedback label
        self.feedback_label = ttk.Label(self.game_frame, text="", wraplength=600)
        self.feedback_label.pack(pady=20)
        
    def start_experiment(self):
        if not self.age_var.get() or not self.gender_var.get():
            messagebox.showerror("Error", "Please fill in all participant information")
            return
            
        self.participant_info = {
            'age': self.age_var.get(),
            'gender': self.gender_var.get(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        self.welcome_frame.pack_forget()
        self.game_frame.pack(fill=tk.BOTH, expand=True)
        
        self.state = self.env.reset()[0]
        self.start_time = datetime.now()
    
    def select_deck(self, deck_idx):
        if self.state is None:
            return
            
        # Record reaction time
        reaction_time = (datetime.now() - self.start_time).total_seconds()
        self.history['reaction_times'].append(reaction_time)
        
        # Make selection
        state, reward, done, _, info = self.env.step(deck_idx)
        
        # Update history
        self.history['deck_choices'].append(deck_idx)
        self.history['rewards'].append(reward)
        self.history['total_money'].append(info['total_money'])
        
        # Update display
        self.money_label['text'] = f"Total Money: ${info['total_money']}"
        self.trials_label['text'] = f"Trials Remaining: {100 - len(self.history['deck_choices'])}"
        
        # Show feedback
        if reward >= 0:
            self.feedback_label['text'] = f"You won ${reward}!"
            self.feedback_label['foreground'] = 'green'
        else:
            self.feedback_label['text'] = f"You lost ${-reward}"
            self.feedback_label['foreground'] = 'red'
        
        # Check if experiment is complete
        if done:
            self.complete_experiment()
        else:
            self.state = state
            self.start_time = datetime.now()
    
    def calculate_metrics(self):
        """Calculate behavioral metrics"""
        choices = self.history['deck_choices']
        rewards = self.history['rewards']
        
        # Calculate advantageous choices (C+D) in last 20 trials
        last_20_choices = choices[-20:]
        advantageous = sum(1 for c in last_20_choices if c in [2, 3]) / len(last_20_choices)
        
        # Calculate risk-seeking after losses
        risk_seeking = []
        for i in range(1, len(choices)):
            if rewards[i-1] < 0:  # After a loss
                risk_seeking.append(1 if choices[i] in [0, 1] else 0)
        risk_seeking_ratio = np.mean(risk_seeking) if risk_seeking else 0
        
        # Calculate deck preferences
        total_choices = len(choices)
        deck_preferences = {
            deck: choices.count(i) / total_choices
            for i, deck in enumerate(['A', 'B', 'C', 'D'])
        }
        
        return {
            'total_money': self.history['total_money'][-1],
            'advantageous_ratio': advantageous,
            'risk_seeking_after_loss': risk_seeking_ratio,
            'mean_reaction_time': np.mean(self.history['reaction_times']),
            'deck_preferences': deck_preferences
        }
    
    def complete_experiment(self):
        """Save data and show completion message"""
        metrics = self.calculate_metrics()
        
        # Create results directory
        results_dir = Path('results/human_data')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data = {
            'participant_info': self.participant_info,
            'history': self.history,
            'metrics': metrics
        }
        
        filename = results_dir / f"participant_{self.participant_info['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Show completion message
        messagebox.showinfo("Complete", 
                          f"Experiment complete!\nFinal Money: ${metrics['total_money']}\n\nThank you for participating!")
        self.root.quit()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    experiment = IGTExperiment()
    experiment.run() 