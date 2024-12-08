from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

def save_participant_data(data):
    """
    Save participant data to Supabase
    """
    try:
        response = supabase.table('participants').insert(data).execute()
        return response.data
    except Exception as e:
        print(f"Error saving to Supabase: {str(e)}")
        return None

def get_all_participants():
    """
    Retrieve all participant data from Supabase
    """
    try:
        response = supabase.table('participants').select("*").execute()
        return response.data
    except Exception as e:
        print(f"Error retrieving from Supabase: {str(e)}")
        return []

def get_participant_by_id(participant_id):
    """
    Retrieve specific participant data
    """
    try:
        response = supabase.table('participants').select("*").eq('id', participant_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error retrieving participant: {str(e)}")
        return None 