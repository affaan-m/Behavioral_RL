import tkinter as tk
from tkinter import ttk, messagebox
import json
import pandas as pd
from datetime import datetime
from igt_env import IGTEnvironment
import numpy as np
import os

class IGTExperiment:
    def __init__(self, master):
        self.master = master
        self.master.title("Iowa Gambling Task")
        
        # Initialize environment
        self.env = IGTEnvironment()
        self.state = self.env.reset()[0]
        
        # Store game history
        self.history = {
            'total_money': [],
            'deck_choices': [],
            'rewards': [],
            'timestamps': [],
            'reaction_times': []
        }
        
        # Participant info
        self.participant_info = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Instructions
        instructions = """
        Welcome to the Iowa Gambling Task!
        
        You will see 4 decks of cards: A, B, C, and D.
        
        On each turn:
        1. Choose a deck by clicking on it
        2. You will win or lose money based on the card drawn
        3. Try to win as much money as possible
        
        Some decks are better than others.
        You are free to switch from one deck to another at any time.
        
        Click 'Start' when you're ready.
        """
        
        self.instruction_label = ttk.Label(self.master, text=instructions, wraplength=400)
        self.instruction_label.pack(pady=20)
        
        # Participant info frame
        self.info_frame = ttk.Frame(self.master)
        self.info_frame.pack(pady=10)
        
        ttk.Label(self.info_frame, text="Age:").grid(row=0, column=0, padx=5)
        self.age_var = tk.StringVar()
        self.age_entry = ttk.Entry(self.info_frame, textvariable=self.age_var)
        self.age_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(self.info_frame, text="Gender:").grid(row=0, column=2, padx=5)
        self.gender_var = tk.StringVar()
        self.gender_combo = ttk.Combobox(self.info_frame, textvariable=self.gender_var,
                                       values=['Male', 'Female', 'Other', 'Prefer not to say'])
        self.gender_combo.grid(row=0, column=3, padx=5)
        
        # Game frame (initially hidden)
        self.game_frame = ttk.Frame(self.master)
        
        # Money display
        self.money_label = ttk.Label(self.game_frame, text="Total Money: $2000")
        self.money_label.pack(pady=10)
        
        # Deck buttons frame
        self.deck_frame = ttk.Frame(self.game_frame)
        self.deck_frame.pack(pady=20)
        
        # Create deck buttons
        self.deck_buttons = {}
        for i, deck in enumerate(['A', 'B', 'C', 'D']):
            btn = ttk.Button(self.deck_frame, text=f"Deck {deck}",
                           command=lambda d=i: self.choose_deck(d))
            btn.grid(row=0, column=i, padx=10)
            self.deck_buttons[deck] = btn
        
        # History display
        self.history_text = tk.Text(self.game_frame, height=10, width=50)
        self.history_text.pack(pady=10)
        
        # Start button
        self.start_button = ttk.Button(self.master, text="Start",
                                     command=self.start_experiment)
        self.start_button.pack(pady=10)
        
    def start_experiment(self):
        if not self.age_var.get() or not self.gender_var.get():
            messagebox.showerror("Error", "Please fill in all participant information")
            return
            
        self.participant_info = {
            'age': self.age_var.get(),
            'gender': self.gender_var.get(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Hide info frame and show game frame
        self.info_frame.pack_forget()
        self.instruction_label.pack_forget()
        self.start_button.pack_forget()
        self.game_frame.pack()
        
        # Start timing
        self.last_action_time = datetime.now()
        
    def choose_deck(self, deck_idx):
        # Record reaction time
        now = datetime.now()
        if hasattr(self, 'last_action_time'):
            reaction_time = (now - self.last_action_time).total_seconds()
        else:
            reaction_time = 0
        self.last_action_time = now
        
        # Take action in environment
        state, reward, done, _, info = self.env.step(deck_idx)
        
        # Record history
        self.history['total_money'].append(info['total_money'])
        self.history['deck_choices'].append('ABCD'[deck_idx])
        self.history['rewards'].append(info['raw_reward'])
        self.history['timestamps'].append(now.isoformat())
        self.history['reaction_times'].append(reaction_time)
        
        # Update display
        self.money_label.config(text=f"Total Money: ${info['total_money']}")
        self.add_history_entry(deck_idx, info)
        
        if done:
            self.end_experiment()
            
    def add_history_entry(self, deck_idx, info):
        deck = 'ABCD'[deck_idx]
        entry = f"Deck {deck}: {'won' if info['raw_reward'] > 0 else 'lost'} ${abs(info['raw_reward'])}\n"
        self.history_text.insert(tk.END, entry)
        self.history_text.see(tk.END)
        
    def end_experiment(self):
        # Calculate final metrics
        final_metrics = self.calculate_metrics()
        
        # Save data
        self.save_experiment_data(final_metrics)
        
        # Show results
        self.show_results(final_metrics)
        
    def calculate_metrics(self):
        """Calculate behavioral metrics"""
        choices = self.history['deck_choices']
        rewards = self.history['rewards']
        
        # Calculate advantageous choices (C+D) in last 20 trials
        last_20_choices = choices[-20:]
        advantageous = sum(1 for c in last_20_choices if c in ['C', 'D']) / len(last_20_choices)
        
        # Calculate risk-seeking after losses
        risk_seeking = []
        for i in range(1, len(choices)):
            if rewards[i-1] < 0:  # After a loss
                risk_seeking.append(1 if choices[i] in ['A', 'B'] else 0)
        risk_seeking_ratio = np.mean(risk_seeking) if risk_seeking else 0
        
        return {
            'total_money': self.history['total_money'][-1],
            'advantageous_ratio': advantageous,
            'risk_seeking_after_loss': risk_seeking_ratio,
            'mean_reaction_time': np.mean(self.history['reaction_times'][1:]),  # Exclude first trial
            'deck_preferences': {
                deck: choices.count(deck) / len(choices)
                for deck in 'ABCD'
            }
        }
        
    def save_experiment_data(self, metrics):
        """Save experiment data and metrics"""
        # Create both directories for compatibility
        os.makedirs('results/human_data', exist_ok=True)
        os.makedirs('results/web_results', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to both locations for compatibility
        data = {
            'participant_info': self.participant_info,
            'history': self.history,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        # Save to human_data directory
        filename1 = f'results/human_data/igt_data_{timestamp}.json'
        with open(filename1, 'w') as f:
            json.dump(data, f, indent=4)
            
        # Save to web_results directory
        filename2 = f'results/web_results/participant_{timestamp}.json'
        with open(filename2, 'w') as f:
            json.dump(data, f, indent=4)
            
    def show_results(self, metrics):
        """Show results to participant"""
        result_window = tk.Toplevel(self.master)
        result_window.title("Experiment Complete")
        
        # Final money
        ttk.Label(result_window, 
                 text=f"Final Money: ${metrics['total_money']:.2f}",
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Deck preferences
        prefs_text = "Deck Choices:\n" + "\n".join(
            f"Deck {deck}: {pref*100:.1f}%"
            for deck, pref in metrics['deck_preferences'].items()
        )
        ttk.Label(result_window, text=prefs_text).pack(pady=10)
        
        # Average reaction time
        ttk.Label(result_window,
                 text=f"Average reaction time: {metrics['mean_reaction_time']*1000:.0f}ms").pack(pady=5)
        
        ttk.Button(result_window, text="Close",
                  command=self.master.destroy).pack(pady=10)

def run_igt_experiment():
    root = tk.Tk()
    app = IGTExperiment(root)
    root.mainloop()

if __name__ == "__main__":
    run_igt_experiment() 