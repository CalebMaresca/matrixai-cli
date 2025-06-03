#!/usr/bin/env python3
"""
CLI script to run matrix wargame scenarios from the scenarios folder.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path so we can import matrix_ai
sys.path.insert(0, str(Path(__file__).parent / "src"))

from matrix_ai import (
    run_matrix_game,
    stream_matrix_game,
    MatrixGame,
    GameState
)
from matrix_ai.main_game_graph import create_main_game_graph
import uuid

def load_scenario(scenario_name):
    """Load a scenario from the scenarios folder."""
    scenarios_dir = Path(__file__).parent / "scenarios"
    scenario_file = scenarios_dir / f"{scenario_name}.json"
    
    if not scenario_file.exists():
        available_scenarios = [f.stem for f in scenarios_dir.glob("*.json")]
        print(f"❌ Scenario '{scenario_name}' not found.")
        print(f"Available scenarios: {', '.join(available_scenarios)}")
        return None
    
    try:
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
        
        scenario = MatrixGame.model_validate(scenario_data)
        return scenario
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        return None

def list_scenarios():
    """List available scenarios."""
    scenarios_dir = Path(__file__).parent / "scenarios"
    scenario_files = list(scenarios_dir.glob("*.json"))
    
    if not scenario_files:
        print("No scenarios found.")
        return
    
    print("\n📚 Available Scenarios:")
    for scenario_file in sorted(scenario_files):
        try:
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            name = scenario_data.get('name', 'Unknown')
            description = scenario_data.get('description', 'No description')
            category = scenario_data.get('category', 'Unknown')
            actors = len(scenario_data.get('actors', []))
            turns = scenario_data.get('game_length', 'Unknown')
            
            print(f"\n🎯 {scenario_file.stem}")
            print(f"   Title: {name}")
            print(f"   Category: {category}")
            print(f"   Description: {description}")
            print(f"   Actors: {actors} | Turns: {turns}")
            
        except Exception as e:
            print(f"❌ Error reading {scenario_file.stem}: {e}")

def run_scenario_streaming(scenario):
    """Run a scenario with streaming output."""
    print(f"\n🎮 Running: {scenario.name}")
    print(f"📖 {scenario.description}")
    print(f"🎭 Actors: {len(scenario.actors)} | 🕐 Turns: {scenario.game_length}")
    
    # Initialize the game
    try:
        print("\n🔧 Initializing game...")
        graph = create_main_game_graph()
        initial_state = GameState.from_matrix_game_setup(scenario)
        
        config = {
            "thread_id": str(uuid.uuid4()),
            "recursion_limit": 600
        }
        
        print("🚀 Starting simulation...\n")
        
        final_state = None
        update_count = 0
        turn_count = 0
        current_actor = ""
        
        for state_update in graph.stream(initial_state, config=config, stream_mode="values"):
            update_count += 1
            
            # Reconstruct GameState from streaming response
            current_state = None
            
            if hasattr(state_update, 'current_turn') and hasattr(state_update, 'current_phase'):
                current_state = state_update
            elif isinstance(state_update, dict) or hasattr(state_update, 'items'):
                if hasattr(state_update, 'current_turn') or 'current_turn' in state_update:
                    try:
                        current_state = GameState.model_validate(dict(state_update))
                    except Exception:
                        continue
            
            if current_state:
                final_state = current_state
                
                # Show turn progress
                if current_state.current_turn != turn_count:
                    turn_count = current_state.current_turn
                    print(f"⏱️  Turn {turn_count}")
                
                # Show actor actions
                if hasattr(current_state, 'current_actor_definition') and current_state.current_actor_definition:
                    actor_name = current_state.current_actor_definition.actor_name
                    if actor_name != current_actor:
                        current_actor = actor_name
                        print(f"   🎮 {current_actor}'s turn")
                
                # Show completed arguments
                if hasattr(current_state, 'current_actor_state') and current_state.current_actor_state:
                    actor_state = current_state.current_actor_state
                    if hasattr(actor_state, 'argument') and actor_state.argument:
                        arg = actor_state.argument
                        if hasattr(arg, 'is_successful') and arg.is_successful is not None:
                            result = "✅" if arg.is_successful else "❌"
                            print(f"   {result} {arg.action_description}")
        
        print(f"\n🏁 Simulation completed!")
        
        if final_state:
            print(f"\n📊 Final Results:")
            print(f"   Duration: {final_state.current_turn} turns")
            print(f"   Phase: {final_state.current_phase.value}")
            
            if hasattr(final_state, 'game_state_summary'):
                print(f"\n📜 Final Situation:")
                print(f"   {final_state.game_state_summary}")
            
            if hasattr(final_state, 'global_narrative_markers') and final_state.global_narrative_markers:
                print(f"\n🌍 Key Developments:")
                for marker in final_state.global_narrative_markers[-5:]:  # Show last 5
                    print(f"   • {marker}")
        
    except Exception as e:
        print(f"❌ Error running scenario: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        print("🎯 Matrix Wargame Scenario Runner")
        print("\nUsage:")
        print("  python run_scenario.py list              # List available scenarios")
        print("  python run_scenario.py <scenario-name>   # Run a scenario")
        print("\nExample:")
        print("  python run_scenario.py diplomatic-crisis")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_scenarios()
    else:
        scenario = load_scenario(command)
        if scenario:
            run_scenario_streaming(scenario)

if __name__ == "__main__":
    main() 