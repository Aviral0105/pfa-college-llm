import os
import csv
import json
import time
from datetime import datetime
from groq import Groq

def get_time():
    return datetime.now().strftime("%H:%M:%S")

def run_simulation(api_key, num_turns=3):
    # Initialize the Groq Client
    client = Groq(api_key=api_key)
    
    # 1. Load Prompts
    with open('prompts/system_pfa_agent.txt', 'r') as f:
        pfa_system = f.read()
    with open('prompts/system_client.txt', 'r') as f:
        client_system_template = f.read()
        
    stressors = []
    with open('data/raw/college_stressors.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 4: stressors.append(row) # Test mode: only doing 4

    results = []
    print(f"[{get_time()}] 🚀 Starting High-Speed Groq Simulation...")

    for idx, stressor in enumerate(stressors):
        print(f"\n[{get_time()}] Scenario {idx+1}/4: {stressor['Scenario']}")
        
        # Setup AI instructions
        client_sys = client_system_template.format(**stressor)
        pfa_messages = [{"role": "system", "content": pfa_system}]
        client_messages = [{"role": "system", "content": client_sys}]
        
        # Client speaks first
        print(f"   -> [{get_time()}] Client starting...")
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Start the conversation by expressing your distress."}]
        )
        first_msg = res.choices[0].message.content
        
        # Save the message to both agents' memories
        pfa_messages.append({"role": "user", "content": first_msg})
        client_messages.append({"role": "assistant", "content": first_msg})
        time.sleep(1) # Tiny pause

        for turn in range(num_turns):
            print(f"   -> [{get_time()}] Turn {turn+1} in progress...")
            
            # PFA replies
            res_pfa = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=pfa_messages
            )
            pfa_out = res_pfa.choices[0].message.content
            pfa_messages.append({"role": "assistant", "content": pfa_out})
            client_messages.append({"role": "user", "content": pfa_out})
            time.sleep(1)

            # Client replies
            res_cli = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=client_messages
            )
            cli_out = res_cli.choices[0].message.content
            client_messages.append({"role": "assistant", "content": cli_out})
            pfa_messages.append({"role": "user", "content": cli_out})
            time.sleep(1)

        # Clean up the format for our JSON file
        transcript = []
        for i, msg in enumerate(pfa_messages[1:]): # Skip the system prompt
            role = "client" if i % 2 == 0 else "responder"
            transcript.append({"role": role, "content": msg["content"]})

        results.append({"scenario": stressor, "transcript": transcript})
        
        # Save securely
        os.makedirs('data/synthetic_raw', exist_ok=True)
        with open('data/synthetic_raw/simulated_conversations.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"[{get_time()}] ✅ Saved Scenario {idx+1}.")

    print(f"\n[{get_time()}] 🎉 PIPELINE STABILIZED. High-speed data generated.")