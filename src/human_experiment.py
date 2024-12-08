import tkinter as tk
from tkinter import ttk
import json
import pandas as pd
from datetime import datetime
from custom_env import InvestmentGameEnv
import numpy as np
import os

class InvestmentGameGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Investment Game")
        
        # Initialize environment
        self.env = InvestmentGameEnv()
        self.state = self.env.reset()[0]
        
        # Store game history
        self.history = {
            'capitals': [],
            'investment_props': [],
            'returns': [],
            'timestamps': []
        }
        
        # GUI Setup
        self.setup_gui()
        
        # Initialize game state
        self.update_display()
        
    def setup_gui(self):
        # Capital Display
        self.capital_label = ttk.Label(self.master, text="Current Capital: $0")
        self.capital_label.pack(pady=10)
        
        # Investment Slider
        self.slider_label = ttk.Label(self.master, text="Investment Proportion (0-100%):")
        self.slider_label.pack(pady=5)
        self.investment_slider = ttk.Scale(self.master, from_=0, to=100, orient='horizontal')
        self.investment_slider.pack(pady=5)
        
        # Invest Button
        self.invest_button = ttk.Button(self.master, text="Make Investment", command=self.make_investment)
        self.invest_button.pack(pady=10)
        
        # History Display
        self.history_text = tk.Text(self.master, height=10, width=50)
        self.history_text.pack(pady=10)
        
        # End Game Button
        self.end_button = ttk.Button(self.master, text="End Game", command=self.end_game)
        self.end_button.pack(pady=10)
        
    def make_investment(self):
        # Get investment proportion from slider (convert 0-100 to 0-1)
        investment_prop = self.investment_slider.get() / 100
        
        # Convert to discrete action (0-10)
        action = int(investment_prop * 10)
        
        # Take step in environment
        state, reward, done, _, info = self.env.step(action)
        
        # Record history
        self.history['capitals'].append(info['capital'])
        self.history['investment_props'].append(info['investment_proportion'])
        self.history['returns'].append(info['market_return'])
        self.history['timestamps'].append(datetime.now().isoformat())
        
        # Update display
        self.update_display()
        self.add_history_entry(info)
        
        if done:
            self.end_game()
    
    def update_display(self):
        self.capital_label.config(text=f"Current Capital: ${self.env.capital:.2f}")
        
    def add_history_entry(self, info):
        entry = f"Investment: {info['investment_proportion']:.1%}, Return: {info['market_return']:.1%}, New Capital: ${info['capital']:.2f}\n"
        self.history_text.insert(tk.END, entry)
        self.history_text.see(tk.END)
    
    def end_game(self):
        # Save game data
        self.save_game_data()
        
        # Show final results
        final_capital = self.env.capital
        avg_investment = np.mean(self.history['investment_props'])
        total_return = (final_capital - self.env.initial_capital) / self.env.initial_capital
        
        result_window = tk.Toplevel(self.master)
        result_window.title("Game Results")
        
        ttk.Label(result_window, text=f"Final Capital: ${final_capital:.2f}").pack(pady=5)
        ttk.Label(result_window, text=f"Total Return: {total_return:.1%}").pack(pady=5)
        ttk.Label(result_window, text=f"Average Investment: {avg_investment:.1%}").pack(pady=5)
        
        ttk.Button(result_window, text="Start New Game", 
                  command=lambda: [result_window.destroy(), self.reset_game()]).pack(pady=10)
        ttk.Button(result_window, text="Quit", 
                  command=self.master.destroy).pack(pady=5)
    
    def reset_game(self):
        self.state = self.env.reset()[0]
        self.history = {
            'capitals': [],
            'investment_props': [],
            'returns': [],
            'timestamps': []
        }
        self.history_text.delete(1.0, tk.END)
        self.update_display()
    
    def save_game_data(self):
        # Create results directory if it doesn't exist
        os.makedirs('results/human_data', exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/human_data/game_data_{timestamp}.json'
        
        # Calculate additional metrics
        metrics = {
            'final_capital': self.env.capital,
            'total_return': (self.env.capital - self.env.initial_capital) / self.env.initial_capital,
            'avg_investment': np.mean(self.history['investment_props']),
            'investment_after_gains': np.mean([inv for i, inv in enumerate(self.history['investment_props'][1:])
                                            if self.history['returns'][i] > 0]),
            'investment_after_losses': np.mean([inv for i, inv in enumerate(self.history['investment_props'][1:])
                                             if self.history['returns'][i] < 0]),
            'risk_seeking_ratio': np.mean([inv for i, inv in enumerate(self.history['investment_props'][1:])
                                         if self.history['returns'][i] < 0]) /
                                np.mean([inv for i, inv in enumerate(self.history['investment_props'][1:])
                                       if self.history['returns'][i] > 0])
        }
        
        # Combine history and metrics
        data = {
            'history': self.history,
            'metrics': metrics
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

def run_human_experiment():
    root = tk.Tk()
    app = InvestmentGameGUI(root)
    root.mainloop()

if __name__ == "__main__":
    run_human_experiment() 