import json
from supabase import create_client
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")

supabase = create_client(supabase_url, supabase_key)

def import_data():
    # Load the generated data
    with open('simulated_igt_data.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} participants to import")
    
    # Import in batches to avoid timeouts
    batch_size = 10
    total_imported = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        try:
            result = supabase.table('participants').insert(batch).execute()
            total_imported += len(batch)
            print(f"Imported batch {i//batch_size + 1}, total progress: {total_imported}/{len(data)}")
            
            # Small delay to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error importing batch starting at index {i}: {str(e)}")
            continue
    
    print(f"\nImport complete. Total records imported: {total_imported}")

if __name__ == "__main__":
    import_data() 