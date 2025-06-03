from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import uuid
from datetime import datetime

from .schemas import (
    GameState, LogEntry, LogEntryType, SecretArgument, ArgumentStatus,
    GamePhase, CombinedNarrativeAndWorldStateResponse
)

# --- PROMPTS ---

COMBINED_NARRATIVE_AND_WORLD_STATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are both a narrative AI and world state manager for a matrix wargame. Your role is to:

1. Create a compelling narrative description of what happened during an argument's adjudication
2. Determine how the argument's outcome affects the game world

For the NARRATIVE, write an account that describes:
- What the actor did (or attempted to do)
- The immediate consequences of their action
- If the action failed, explain that the actor tried to do this but failed because of specific reasons from the cons
- If any secret arguments were triggered, incorporate their revelation into the narrative

The narrative should be:
- Written in past tense
- Engaging and immersive
- Realistic and grounded in the game world
- Specific about what happened and why
- Include the impact of any revealed secret arguments

For the WORLD STATE UPDATES, determine:
1. New effects to add to the actor's personal effects list
2. Updates to force units (location changes, status updates, etc.) - NOTE: You can update ANY actor's forces, not just the current actor's
3. New global narrative markers that affect the overall game state
4. An updated game state summary incorporating these changes

For the game state summary, provide a comprehensive overview that includes:
- The current overall situation and key developments
- Major ongoing tensions, conflicts, or diplomatic situations
- Recent significant events and their ongoing impact
- The general strategic position of major actors
- Any emerging trends or patterns in the game
- The broader context, not just the most recent action

The summary should give players a clear understanding of the current state of play and help them make informed decisions about their next moves.

Consider:
- The scope and scale of the action
- Realistic consequences and ripple effects
- How this affects other actors and the broader situation
- Both immediate and potential longer-term implications
- The impact of any triggered secret arguments

Game Context:
{game_context}

Current Actor: {actor_name}
Turn: {current_turn}
Current Game State Summary: {current_summary}
Global Narrative Markers: {global_markers}
Triggered Secret Arguments: {triggered_secrets}"""),
    ("human", """Argument Details:
Action: {action_description}
Pros: {pros}
Cons: {cons}
Adjudication Method: {adjudication_method}
Success: {is_successful}
Final Probability: {final_probability}

Current Actor Effects: {current_effects}

All Forces in Game:
{all_forces}

Create both the narrative description and world state updates for this argument's resolution.""")
])

# --- NODE FUNCTIONS ---

def create_narrative_and_update_world_state(state: GameState) -> GameState:
    """Combined node to create the adjudication narrative and update world state in a single LLM call"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for narrative and world state update")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for narrative and world state update")
        return state
    
    current_argument = current_actor_state.argument
    
    # Prepare context
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Situation: {state.game_state_summary}
"""
    
    # Get triggered secrets info
    triggered_secrets = state.triggered_secrets_this_turn
    triggered_secrets_str = "\n".join(triggered_secrets) if triggered_secrets else "None"
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)  # Balanced temperature for both narrative and analysis
    
    # Create combined chain
    combined_chain = COMBINED_NARRATIVE_AND_WORLD_STATE_PROMPT | llm.with_structured_output(CombinedNarrativeAndWorldStateResponse)
    
    try:
        # For secret arguments, we need to be careful about what we reveal in the game state
        action_description = current_argument.action_description
        if isinstance(current_argument, SecretArgument) and current_argument.is_successful:
            # Don't reveal the specific action in the game state summary
            action_description_for_summary = "implemented a covert operation"
        else:
            action_description_for_summary = action_description
        
        # Prepare all forces context for LLM
        all_forces_info = []
        for actor_state in state.actor_states:
            actor_forces = [f"{f.unit_name} at {f.location}" + (f" ({f.details})" if f.details else "") 
                           for f in actor_state.current_forces]
            if actor_forces:
                all_forces_info.append(f"{actor_state.actor_name}: {', '.join(actor_forces)}")
        all_forces_str = "\n".join(all_forces_info) if all_forces_info else "No forces deployed"
        
        combined_response = combined_chain.invoke({
            "game_context": game_context,
            "actor_name": current_actor.actor_name,
            "current_turn": state.current_turn,
            "current_summary": state.game_state_summary,
            "global_markers": state.global_narrative_markers,
            "action_description": action_description,  # Use full description for narrative
            "pros": current_argument.pros,
            "cons": current_argument.cons,
            "adjudication_method": current_argument.adjudication_method.value if current_argument.adjudication_method else "Unknown",
            "is_successful": current_argument.is_successful,
            "final_probability": current_argument.final_probability,
            "current_effects": current_actor_state.effects,
            "all_forces": all_forces_str,
            "triggered_secrets": triggered_secrets_str
        })
        
        # Update argument with narrative
        current_argument.adjudication_narrative = combined_response.adjudication_narrative
        
        # Apply world state updates to actor state
        current_actor_state.effects.extend(combined_response.actor_effects)
        
        # Update force units - search through ALL actors since one actor's actions can affect others
        for force_update in combined_response.force_updates:
            if force_update.unit_name and force_update.actor_name:
                # Find the actor that owns this force
                target_actor_state = None
                for actor_state in state.actor_states:
                    if actor_state.actor_name == force_update.actor_name:
                        target_actor_state = actor_state
                        break
                
                if target_actor_state:
                    # Find and update the specific force unit
                    for force in target_actor_state.current_forces:
                        if force.unit_name == force_update.unit_name:
                            # Update force fields
                            if force_update.location:
                                force.location = force_update.location
                            if force_update.details:
                                force.details = force_update.details
                            break
        
        # Update global state
        state.global_narrative_markers.extend(combined_response.global_narrative_markers)
        state.game_state_summary = combined_response.game_state_summary_update
        
    except Exception as e:
        print(f"Error in combined narrative and world state update: {e}")
        # Apply minimal default updates
        if current_argument.is_successful:
            if isinstance(current_argument, SecretArgument):
                current_argument.adjudication_narrative = f"{current_actor.actor_name} successfully implemented a covert operation"
                current_actor_state.effects.append(f"Successfully implemented a covert operation")
            else:
                current_argument.adjudication_narrative = f"{current_actor.actor_name} successfully executed their planned action: {current_argument.action_description}"
                current_actor_state.effects.append(f"Successfully completed: {current_argument.action_description}")
        else:
            current_argument.adjudication_narrative = f"{current_actor.actor_name} attempted to {current_argument.action_description} but failed due to various challenges and constraints."
            current_actor_state.effects.append(f"Failed attempt: {current_argument.action_description}")
    
    return state

def create_log_entry(state: GameState) -> GameState:
    """Node to create log entries for the argument"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for log entry creation")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for log entry creation")
        return state
    
    current_argument = current_actor_state.argument
    
    # Create log entry for the argument
    log_entry = LogEntry(
        entry_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        turn=state.current_turn,
        phase=state.current_phase,
        entry_type=LogEntryType.ARGUMENT,
        actor_name=current_actor.actor_name,
        content=current_argument,
        summary=f"{current_actor.actor_name}: {current_argument.action_description} - {'Success' if current_argument.is_successful else 'Failure'}"
    )
    
    state.game_log.append(log_entry)
    
    return state

def update_game_phase(state: GameState) -> GameState:
    """Node to update the game phase after scenario update"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if current_actor_state and current_actor_state.argument:
        current_argument = current_actor_state.argument
        
        # If this was a successful secret argument, add it to pending_secret_arguments
        if isinstance(current_argument, SecretArgument) and current_argument.is_successful:
            # Add to pending secret arguments
            current_actor_state.pending_secret_arguments.append(current_argument)
        
        # Clear the current argument since it's been processed
        current_actor_state.argument = None
        
        # Clear any big project feedback since the turn is complete
        current_actor_state.big_project_feedback = None
    
    # Remove triggered secret arguments from all actors' pending_secret_arguments
    for actor_state in state.actor_states:
        # Create a new list without the triggered arguments
        actor_state.pending_secret_arguments = [
            secret_arg for secret_arg in actor_state.pending_secret_arguments
            if not secret_arg.is_triggered
        ]
    
    # Clear triggered secrets for this turn
    state.triggered_secrets_this_turn.clear()
    
    # Set phase for what's coming next (game over check)
    state.current_phase = GamePhase.GAME_OVER_CHECK
    
    return state

# --- GRAPH CONSTRUCTION ---

def create_scenario_update_graph() -> StateGraph:
    """Create the scenario update workflow graph"""
    
    # Create the graph
    workflow = StateGraph(GameState)
    
    # Add nodes
    workflow.add_node("create_narrative_and_update_world_state", create_narrative_and_update_world_state)
    workflow.add_node("create_log_entry", create_log_entry)
    workflow.add_node("update_game_phase", update_game_phase)
    
    # Add edges - now we have one less node in the sequence
    workflow.add_edge(START, "create_narrative_and_update_world_state")
    workflow.add_edge("create_narrative_and_update_world_state", "create_log_entry")
    workflow.add_edge("create_log_entry", "update_game_phase")
    workflow.add_edge("update_game_phase", END)
    
    return workflow.compile() 