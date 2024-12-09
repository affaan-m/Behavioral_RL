from supabase import create_client
import os
from datetime import datetime, timedelta
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("Supabase credentials not found in environment variables!")
    logger.info("Please make sure SUPABASE_URL and SUPABASE_KEY are set.")
    exit(1)

logger.info(f"Connecting to Supabase URL: {supabase_url[:20]}...")
supabase = create_client(supabase_url, supabase_key)

def check_data():
    try:
        # Query all entries
        logger.info("Querying Supabase for all participant data...")
        response = supabase.table('participants').select("*").order('timestamp.desc').execute()
        
        if not response.data:
            logger.warning("No data found in the participants table.")
            logger.info("Make sure you completed the experiment and it saved successfully.")
            return
        
        print(f"\nFound {len(response.data)} total entries:")
        for entry in response.data:
            print("\n" + "="*50)
            print(f"Session ID: {entry.get('session_id', 'N/A')}")
            print(f"Timestamp: {entry['timestamp']}")
            
            # Verify data completeness
            history = entry['history']
            choices = history['deck_choices']
            rewards = history['rewards']
            total_money = history['total_money']
            
            print(f"\nData Completeness:")
            print(f"Total Trials: {len(choices)}")
            print(f"Total Rewards Recorded: {len(rewards)}")
            print(f"Money History Length: {len(total_money)}")
            print(f"Reaction Times Recorded: {len(history['reaction_times'])}")
            
            print(f"\nPerformance Metrics:")
            metrics = entry['metrics']
            print(f"Final Money: ${metrics['total_money']}")
            print(f"Advantageous Ratio: {metrics['advantageous_ratio']*100:.1f}%")
            print(f"Mean Reaction Time: {metrics['mean_reaction_time']:.2f}s")
            
            print("\nDeck Preferences:")
            for deck, pref in metrics['deck_preferences'].items():
                print(f"  {deck}: {pref*100:.1f}%")
            
            print("\nLearning Progress:")
            print(f"  Early Advantageous: {metrics['learning_progress']['early_advantageous']*100:.1f}%")
            print(f"  Late Advantageous: {metrics['learning_progress']['late_advantageous']*100:.1f}%")
            
            # Verify session completion
            if len(choices) != 100:
                print(f"\nWARNING: Incomplete session - only {len(choices)} trials completed!")
            
            # Check for any anomalies
            if len(set(choices)) < 2:
                print("\nWARNING: Low deck variety - participant might be stuck on one deck!")
            
            if metrics['mean_reaction_time'] < 0.1:
                print("\nWARNING: Very fast reactions - might indicate automated/bot behavior!")
            
    except Exception as e:
        logger.error(f"Error checking data: {str(e)}")
        logger.info("Try checking the Supabase dashboard directly at: https://supabase.com")

if __name__ == "__main__":
    check_data() 