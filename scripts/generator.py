import os
import csv
import json
import time
import random
from datetime import datetime
from groq import Groq


def get_time():
    return datetime.now().strftime("%H:%M:%S")


def run_simulation(api_key, num_conversations=4, stressor_limit=5):
    """
    For every stressor row in the CSV, generate `num_conversations` distinct
    conversations, each containing between 3 and 4 full therapist↔client
    exchange turns.
    """
    client = Groq(api_key=api_key)

    # ── Load prompts ──────────────────────────────────────────────────────────
    with open("prompts/system_pfa_agent.txt", "r", encoding="utf-8") as f:
        pfa_system = f.read()
    with open("prompts/system_client.txt", "r", encoding="utf-8") as f:
        client_system_template = f.read()

    # ── Load stressors ────────────────────────────────────────────────────────
    stressors = []
    with open("data/raw/college_stressors.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < stressor_limit:                          # keep the same 5-row cap for now
                stressors.append(row)

    all_results = []
    print(f"[{get_time()}] 🚀 Starting simulation — "
          f"{num_conversations} conversations × {len(stressors)} stressors")

    # ── OUTER LOOP: iterate over every stressor ───────────────────────────────
    for idx, stressor in enumerate(stressors):
        scenario_desc = stressor.get("scenario", "Unknown Scenario")
        print(f"\n[{get_time()}] ── Stressor {idx + 1}/{len(stressors)}: "
              f"{scenario_desc}")

        # Build the client system prompt once per stressor
        try:
            client_sys = client_system_template.format(**stressor)
        except KeyError as e:
            print(f"[{get_time()}] ⚠️  Missing placeholder {e} — using fallback.")
            client_sys = (
                f"You are a college student experiencing this distress: "
                f"{scenario_desc}. Express your feelings authentically."
            )

        # ── CONVERSATION LOOP: 4 distinct conversations per stressor ─────────
        # FIX 1 ▸ New outer loop — previously only 1 conversation was produced.
        for conv_idx in range(num_conversations):
            print(f"   [{get_time()}] Conversation {conv_idx + 1}/{num_conversations}")

            # FIX 2 ▸ Randomise turns between 3 and 4 per conversation.
            # Previously num_turns was a fixed parameter with no per-conversation
            # variation; now each conversation independently draws 3 or 4 turns.
            num_turns = random.randint(3, 4)
            print(f"   [{get_time()}] → This conversation will use {num_turns} turns")

            # Fresh message histories for every new conversation
            # FIX 3 ▸ Histories are reset here (inside the conv loop), not just
            # once per stressor — previously a single history was reused, which
            # would have blended context across conversations.
            pfa_messages    = [{"role": "system", "content": pfa_system}]
            client_messages = [{"role": "system", "content": client_sys}]

            # ── Turn 0: client opens the conversation ─────────────────────────
            try:
                res = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system",  "content": client_sys},
                        {"role": "user",    "content": "Start the conversation by expressing your distress."},
                    ],
                )
                first_msg = res.choices[0].message.content
            except Exception as e:
                print(f"   [{get_time()}] ❌ API error on opening turn: {e}")
                continue           # skip this conversation, try the next

            pfa_messages.append(   {"role": "user",      "content": first_msg})
            client_messages.append({"role": "assistant", "content": first_msg})
            time.sleep(2)

            # ── Exchange turns ─────────────────────────────────────────────────
            for turn in range(num_turns):
                print(f"   [{get_time()}]   turn {turn + 1}/{num_turns} …")

                try:
                    # Therapist replies
                    res_pfa = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=pfa_messages,
                    )
                    pfa_out = res_pfa.choices[0].message.content
                    pfa_messages.append(   {"role": "assistant", "content": pfa_out})
                    client_messages.append({"role": "user",      "content": pfa_out})
                    time.sleep(2)

                    # Client replies
                    res_cli = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=client_messages,
                    )
                    cli_out = res_cli.choices[0].message.content
                    client_messages.append({"role": "assistant", "content": cli_out})
                    pfa_messages.append(   {"role": "user",      "content": cli_out})
                    time.sleep(2)

                except Exception as e:
                    print(f"   [{get_time()}] ❌ API error on turn {turn + 1}: {e}")
                    break          # save whatever turns completed

            # ── Build transcript ───────────────────────────────────────────────
            transcript = []
            for i, msg in enumerate(pfa_messages[1:]):   # skip system prompt
                role = "client" if i % 2 == 0 else "responder"
                transcript.append({"role": role, "content": msg["content"]})

            # FIX 4 ▸ conversation_index added to metadata so every record is
            # uniquely identifiable in the output JSON.
            all_results.append({
                "metadata": {
                    **stressor,
                    "conversation_index": conv_idx + 1,   # 1-based for readability
                },
                "num_turns": num_turns,
                "transcript": transcript,
            })

            print(f"   [{get_time()}] ✅ Conversation {conv_idx + 1} saved "
                  f"({num_turns} turns, {len(transcript)} messages).")

        # ── Persist after every stressor (safe mid-run saves) ─────────────────
        os.makedirs("data/synthetic_raw", exist_ok=True)
        with open(
            "data/synthetic_raw/simulated_conversations.json", "w", encoding="utf-8"
        ) as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        print(f"[{get_time()}] 💾 Saved after stressor {idx + 1} "
              f"({len(all_results)} total conversations so far).")

    print(
        f"\n[{get_time()}] 🎉 PIPELINE COMPLETE — "
        f"{len(all_results)} conversations across {len(stressors)} stressors."
    )
