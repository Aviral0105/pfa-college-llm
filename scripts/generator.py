import os
import csv
import json
import time
import google.generativeai as genai

def run_simulation(api_key, num_turns=4):
    """Runs a multi-agent simulation between a Client LLM and a PFA Responder LLM."""
    genai.configure(api_key=api_key)
    
    # 1. Load the Prompts and Scenarios
    with open('prompts/system_pfa_agent.txt', 'r') as f:
        pfa_system_prompt = f.read()
    with open('prompts/system_client.txt', 'r') as f:
        client_system_template = f.read()
        
    stressors = []
    with open('data/raw/college_stressors.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stressors.append(row)
            
    # 2. Initialize the PFA Model (This persona remains static)
    pfa_model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=pfa_system_prompt
    )

    results = []
    print(f"🚀 Starting simulation for {len(stressors)} college scenarios...")
    
    # 3. The Conversation Loop
    for idx, stressor in enumerate(stressors):
        print(f"Running scenario {idx+1}/{len(stressors)}: {stressor['Scenario']}")
        
        # Inject the specific stressor into the Client's system prompt
        client_prompt = client_system_template.format(
            Category=stressor['Category'],
            Scenario=stressor['Scenario'],
            Intensity=stressor['Intensity']
        )
        
        client_model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=client_prompt
        )
        
        # Start empty chat histories
        pfa_chat = pfa_model.start_chat(history=[])
        client_chat = client_model.start_chat(history=[])
        
        conversation = []
        
        # Force the Client to speak first
        print("   -> Initializing Client distress...")
        client_response = client_chat.send_message("Start the conversation by expressing your current distress.")
        conversation.append({"role": "client", "content": client_response.text.strip()})
        time.sleep(15) # Safe pause
        
        # Pass messages back and forth
        for turn in range(num_turns):
            print(f"   -> Executing Turn {turn+1}/{num_turns}...")
            
            # Responder replies to Client
            pfa_response = pfa_chat.send_message(conversation[-1]['content'])
            conversation.append({"role": "responder", "content": pfa_response.text.strip()})
            time.sleep(15) # Safe pause
            
            # Client replies to Responder
            client_reply = client_chat.send_message(conversation[-1]['content'])
            conversation.append({"role": "client", "content": client_reply.text.strip()})
            time.sleep(15) # Safe pause
            
        # Store the full transcript
        results.append({
            "scenario": stressor,
            "transcript": conversation
        })
        
        print("✅ Scenario complete. Saving and resting before the next one...")
        time.sleep(15)
        
    # 4. Save to the persistent directory
    output_file = 'data/synthetic_raw/simulated_conversations.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Simulation complete. Raw data saved to {output_file}")