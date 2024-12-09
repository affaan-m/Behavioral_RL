from supabase import create_client
import os
from dotenv import load_dotenv

def get_supabase_client():
    """Get Supabase client using environment variables"""
    load_dotenv()
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
    
    return create_client(supabase_url, supabase_key)

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