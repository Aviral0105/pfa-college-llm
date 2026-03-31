import os
import csv
import json
import time
from datetime import datetime
import google.generativeai as genai

def get_time():
    return datetime.now().strftime("%H:%M:%S")

def safe_send_message(chat_session, message, max_retries=3):
    """Wraps the API call with timestamps and retry logic."""
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(message)
            return response
        except Exception as e:
            if "429" in str(e):
                print(f"      [{get_time()}] ⚠️ Quota Hit. Sleeping 70s to reset window...")
                time.sleep(70)
            else:
                raise e
    return None

def run_simulation(api_key, num_turns=3): # Reduced turns to 3 for the test
    genai.configure(api_key=api_key)
    
    with open('prompts/system_pfa_agent.txt', 'r') as f:
        pfa_system_prompt = f.read()
    with open('prompts/system_client.txt', 'r') as f:
        client_system_template = f.read()
        
    stressors = []
    with open('data/raw/college_stressors.csv', 'r') as f:
        reader = csv.DictReader(f)
        # ONLY TAKE THE FIRST 4 SCENARIOS FOR THIS TEST
        for i, row in enumerate(reader):
            if i < 4: stressors.append(row)
            
    pfa_model = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=pfa_system_prompt)
    results = []
    
    print(f"[{get_time()}] 🚀 Starting 4-Scenario stabilized test run...")
    
    for idx, stressor in enumerate(stressors):
        print(f"\n[{get_time()}] Running Scenario {idx+1}/4: {stressor['Scenario']}")
        
        client_prompt = client_system_template.format(
            Category=stressor['Category'], Scenario=stressor['Scenario'], Intensity=stressor['Intensity']
        )
        client_model = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=client_prompt)
        
        pfa_chat = pfa_model.start_chat(history=[])
        client_chat = client_model.start_chat(history=[])
        
        conversation = []
        
        # 1. Start Conversation
        print(f"   -> [{get_time()}] Client speaking...")
        res = safe_send_message(client_chat, "Start by expressing your distress.")
        conversation.append({"role": "client", "content": res.text.strip()})
        time.sleep(20) # 20 second gap between every single message
        
        # 2. Back and Forth
        for turn in range(num_turns):
            print(f"   -> [{get_time()}] Turn {turn+1}/{num_turns} in progress...")
            
            # Responder
            res_pfa = safe_send_message(pfa_chat, conversation[-1]['content'])
            conversation.append({"role": "responder", "content": res_pfa.text.strip()})
            time.sleep(20)
            
            # Client
            res_cli = safe_send_message(client_chat, conversation[-1]['content'])
            conversation.append({"role": "client", "content": res_cli.text.strip()})
            time.sleep(20)
            
        results.append({"scenario": stressor, "transcript": conversation})
        
        # SAVE IMMEDIATELY
        output_path = 'data/synthetic_raw/simulated_conversations.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[{get_time()}] ✅ Saved Scenario {idx+1}. Resting 30s...")
        time.sleep(30)

    print(f"\n[{get_time()}] 🎉 TEST COMPLETE. Check your Drive for the file.")