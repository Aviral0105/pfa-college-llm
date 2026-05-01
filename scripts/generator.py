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
    with open('prompts/system_pfa_agent.txt', 'r', encoding='utf-8') as f:
        pfa_system = f.read()
    with open('prompts/system_client.txt', 'r', encoding='utf-8') as f:
        client_system_template = f.read()
        
    stressors = []
    with open('data/raw/college_stressors.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 5: stressors.append(row) # Now set to generate exactly 5 datasets

    results = []
    print(f"[{get_time()}] 🚀 Starting High-Speed Groq Simulation...")

    for idx, stressor in enumerate(stressors):
        # Uses the new lowercase 'scenario' column name securely
        scenario_desc = stressor.get('scenario', 'Unknown Scenario')
        print(f"\n[{get_time()}] Scenario {idx+1}/{len(stressors)}: {scenario_desc}")
        
        # Setup AI instructions
        try:
            # This injects your new CSV columns into the prompt
            client_sys = client_system_template.format(**stressor)
        except KeyError as e:
            print(f"[{get_time()}] ⚠️ WARNING: Your prompts/system_client.txt contains a placeholder {e} that doesn't match your new CSV headers!")
            print(f"[{get_time()}] ⚠️ Using fallback prompt to prevent crash...")
            client_sys = f"You are a college student experiencing this distress: {scenario_desc}. Express your feelings."
        
        pfa_messages = [{"role": "system", "content": pfa_system}]
        client_messages = [{"role": "system", "content": client_sys}]
        
        # Client speaks first
        print(f"   -> [{get_time()}] Client starting...")
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Start the conversation by expressing your distress."}]
            )
            first_msg = res.choices[0].message.content
        except Exception as e:
             print(f"   -> ❌ API Error: {e}")
             continue # Skip to the next one if the API fails
        
        # Save the message to both agents' memories
        pfa_messages.append({"role": "user", "content": first_msg})
        client_messages.append({"role": "assistant", "content": first_msg})
        time.sleep(2) # Increased to 2 seconds to respect Groq rate limits

        for turn in range(num_turns):
            print(f"   -> [{get_time()}] Turn {turn+1} in progress...")
            
            try:
                # PFA replies
                res_pfa = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=pfa_messages
                )
                pfa_out = res_pfa.choices[0].message.content
                pfa_messages.append({"role": "assistant", "content": pfa_out})
                client_messages.append({"role": "user", "content": pfa_out})
                time.sleep(2)

                # Client replies
                res_cli = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=client_messages
                )
                cli_out = res_cli.choices[0].message.content
                client_messages.append({"role": "assistant", "content": cli_out})
                pfa_messages.append({"role": "user", "content": cli_out})
                time.sleep(2)
            except Exception as e:
                print(f"   -> ❌ API Error during turn {turn+1}: {e}")
                break # Save whatever dialogue we have so far and move on

        # Clean up the format for our JSON file
        transcript = []
        for i, msg in enumerate(pfa_messages[1:]): # Skip the system prompt
            role = "client" if i % 2 == 0 else "responder"
            transcript.append({"role": role, "content": msg["content"]})

        # Packaged with all your new Kosha metadata!
        results.append({"metadata": stressor, "transcript": transcript})
        
        # Save securely
        os.makedirs('data/synthetic_raw', exist_ok=True)
        with open('data/synthetic_raw/simulated_conversations.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"[{get_time()}] ✅ Saved Scenario {idx+1}.")

    print(f"\n[{get_time()}] 🎉 PIPELINE STABILIZED. Generated {len(results)} datasets.")
