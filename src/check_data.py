from supabase import create_client
import os
from datetime import datetime, timedelta
import logging

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
        response = supabase.table('participants').select("*").execute()
        
        if not response.data:
            logger.warning("No data found in the participants table.")
            logger.info("Make sure you completed the experiment and it saved successfully.")
            return
        
        print(f"\nFound {len(response.data)} total entries:")
        for entry in response.data:
            print("\nEntry Details:")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Total Money: ${entry['metrics']['total_money']}")
            print(f"Advantageous Ratio: {entry['metrics']['advantageous_ratio']*100:.1f}%")
            print("Deck Preferences:")
            for deck, pref in entry['metrics']['deck_preferences'].items():
                print(f"  {deck}: {pref*100:.1f}%")
            print(f"Mean Reaction Time: {entry['metrics']['mean_reaction_time']:.2f}s")
            print("\nLearning Progress:")
            print(f"  Early Advantageous: {entry['metrics']['learning_progress']['early_advantageous']*100:.1f}%")
            print(f"  Late Advantageous: {entry['metrics']['learning_progress']['late_advantageous']*100:.1f}%")
            
            # Print some history stats
            choices = entry['history']['deck_choices']
            print(f"\nTotal Choices: {len(choices)}")
            print("Choice Distribution:")
            for i, deck in enumerate(['A', 'B', 'C', 'D']):
                count = choices.count(i)
                print(f"  Deck {deck}: {count} times ({count/len(choices)*100:.1f}%)")
            
    except Exception as e:
        logger.error(f"Error checking data: {str(e)}")
        logger.info("Try checking the Supabase dashboard directly at: https://supabase.com")

if __name__ == "__main__":
    check_data() 