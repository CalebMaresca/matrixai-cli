from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
import uuid
from datetime import datetime
from typing import List, Tuple

from .schemas import (
    GameState, ActorState, ArgumentStatus, StandardArgument, SecretArgument,
    ArgumentResponse, SecretArgumentValidationResponse, BigProjectCheckResponse,
    LogEntry, LogEntryType, GamePhase
)

# --- PROMPTS ---

DELIBERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI player in a matrix wargame. Matrix games are a unique form of wargaming where players are free to undertake ANY plausible action during their turn. Unlike traditional wargames with rigid rules, your actions are limited only by what makes narrative sense within the game world.

KEY CONCEPTS:
1. Arguments: Each turn, you propose an "argument" consisting of:
   - An ACTION: What specifically you want to do with a measurable outcome
   - REASONS (Pros): Why this action would succeed (your capabilities, resources, favorable conditions, etc.)
   
2. One Action Per Turn: You can only take ONE major action per turn. This represents your highest priority. Other routine activities continue in the background, but your argument represents the most impactful action you're taking.

3. Turn Length: Each turn represents {turn_length}. Your actions should be realistic for this timeframe.

4. Secret Arguments: Use these ONLY when secrecy provides real advantage (covert ops, surprise attacks, etc.). They must have specific trigger conditions for when they're revealed. Don't use secrets just to avoid opposition.

5. Big Projects: Actions that would realistically take multiple turns should be broken into stages. Each stage becomes a separate argument across multiple turns.

6. Building on Success: Arguments that build on previous successful actions are more likely to succeed. The game rewards coherent narratives.

GAME INFORMATION:
Title: {game_name}
Background: {game_background}

YOUR ROLE:
You are playing: {actor_name}
Your Briefing: {actor_briefing}
Your Objectives: {objectives}

Remember: You're not just executing moves - you're helping create a collaborative narrative. Your actions should be plausible within the game world and advance your objectives while responding to the evolving situation."""),
    MessagesPlaceholder(variable_name="conversation_history")
])

SECRET_VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an umpire AI evaluating whether a proposed secret argument is truly appropriate to remain secret in a matrix wargame.

A secret argument should only be used when:
1. The action genuinely benefits from secrecy (e.g., surprise attacks, covert operations, diplomatic back-channels)
2. The action would be less effective or fail if known to other players
3. There are realistic trigger conditions for when it would be revealed
4. The secrecy is plausible within the game world

A secret argument should NOT be used for:
1. Actions that would naturally be public or observable
2. Actions where secrecy provides no meaningful advantage
3. Attempts to hide actions simply to avoid opposition
4. Actions with vague or unrealistic trigger conditions

Game Context:
{game_context}

Actor: {actor_name}"""),
    ("human", """Proposed Secret Argument:
Action: {action_description}
Trigger Conditions: {trigger_conditions}
Reasoning: {pros}

Evaluate whether this truly warrants being a secret argument or should be reformulated as a standard argument.""")
])

BIG_PROJECT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an umpire AI evaluating whether a proposed action constitutes a "big project" that should be broken down into smaller stages in a matrix wargame.

A "big project" is an action that:
1. Would realistically take multiple turns to complete given THIS SPECIFIC actor's capabilities and resources
2. Has multiple distinct stages or phases that cannot reasonably be compressed into the given timeframe
3. Is too complex or large-scale to achieve meaningful progress in a single turn of {turn_length}
4. Would benefit from being broken down into manageable steps that build upon each other

IMPORTANT CONSIDERATIONS:
- Actor capabilities matter: What a nation-state can accomplish differs vastly from what a small organization can do
- Turn length is crucial: {turn_length} is a significant factor in determining feasibility
- Matrix games encourage ambitious actions, but they should still be plausible within the timeframe
- Focus on whether meaningful progress can be made in one turn, not whether the entire goal can be completed

Evaluate based on:
- The actor's current resources and capabilities
- The complexity and scope of the proposed action
- Whether the action has natural break points or stages
- The realistic timeline for such an action given the actor's position
     
Err on the side of allowing an action to be completed in one turn (that is, not declaring it a big project).

If this is a big project, suggest:
1. What the first stage should be (something achievable in one turn that makes meaningful progress)
2. What the remaining stages might look like (to be saved for future planning)

Game Context:
{game_context}

Actor: {actor_name}
Turn Length: {turn_length}"""),
    ("human", """Proposed Action:
{action_description}

Supporting Reasons:
{pros}

Evaluate whether this is a big project that should be broken down into stages.""")
])

# --- HELPER FUNCTIONS ---

def update_conversation_history(state: GameState, actor_state: ActorState) -> None:
    """Update the actor's conversation history with recent events"""
    
    # Get current actor name
    actor_name = actor_state.actor_name
    
    # Find the last argument by this actor
    last_actor_argument_index = -1
    for i, log in enumerate(state.game_log):
        if (log.entry_type == LogEntryType.ARGUMENT and 
            hasattr(log.content, 'proposing_actor_name') and 
            log.content.proposing_actor_name == actor_name):
            last_actor_argument_index = i
    
    # Check if this is the very first move of the game
    if last_actor_argument_index == -1 and not any(log for log in state.game_log if log.entry_type == LogEntryType.ARGUMENT):
        # First move of the game
        actor_state.conversation_history.append(("human", f"""Turn 1 begins. You have the first move.

Current Game State: {state.game_definition.introduction}

Your Current Forces: {_format_forces(state, actor_name)}

What action do you want to take?"""))
        return
    
    # Collect all argument logs after this actor's last action
    others_actions = []
    if last_actor_argument_index >= 0:
        # Get logs after the actor's last action
        for log in state.game_log[last_actor_argument_index + 1:]:
            if (log.entry_type == LogEntryType.ARGUMENT and 
                hasattr(log.content, 'proposing_actor_name') and
                hasattr(log.content, 'adjudication_narrative') and 
                log.content.adjudication_narrative):
                others_actions.append(f"- {log.content.proposing_actor_name}: {log.content.adjudication_narrative}")
    else:
        # This actor hasn't acted yet, get all argument logs
        for log in state.game_log:
            if (log.entry_type == LogEntryType.ARGUMENT and 
                hasattr(log.content, 'proposing_actor_name') and
                hasattr(log.content, 'adjudication_narrative') and 
                log.content.adjudication_narrative):
                others_actions.append(f"- {log.content.proposing_actor_name}: {log.content.adjudication_narrative}")
    
    # Build the human message
    human_msg = f"Turn {state.current_turn}. "
    
    if others_actions:
        if last_actor_argument_index == -1:
            human_msg += "Other actors have moved before you:\n\n"
        else:
            human_msg += "Since your last action, other actors have moved:\n\n"
        human_msg += "\n".join(others_actions) + "\n\n"
    else:
        if last_actor_argument_index == -1:
            human_msg += "You have the first move.\n\n"
        else:
            human_msg += "You have the first move this turn.\n\n"
    
    human_msg += f"""Current Game State: {state.game_state_summary}
Global Situation: {', '.join(state.global_narrative_markers) if state.global_narrative_markers else 'Situation developing'}

Your Current Forces: {_format_forces(state, actor_name)}
Your Current Effects: {_format_effects(state, actor_name)}

What action do you want to take this turn? Remember, each turn represents {state.game_definition.turn_length}, so suggest an action that is achievable in that time frame. If you have a longer term plan, you can break it down into stages and propose them one at a time, using your scratch pad to save notes."""
    
    actor_state.conversation_history.append(("human", human_msg))


def _format_forces(state: GameState, actor_name: str) -> str:
    """Helper to format forces for a specific actor"""
    actor_state = next((a for a in state.actor_states if a.actor_name == actor_name), None)
    if not actor_state or not actor_state.current_forces:
        return "None"
    
    return ", ".join([f"{f.unit_name} at {f.location}" + (f" ({f.details})" if f.details else "") 
                     for f in actor_state.current_forces])


def _format_effects(state: GameState, actor_name: str) -> str:
    """Helper to format effects for a specific actor"""
    actor_state = next((a for a in state.actor_states if a.actor_name == actor_name), None)
    if not actor_state or not actor_state.effects:
        return "None"
    
    return ", ".join(actor_state.effects)

# --- NODE FUNCTIONS ---

def update_actor_conversation_history(state: GameState) -> GameState:
    """Node to update the current actor's conversation history"""
    
    current_actor_state = state.current_actor_state
    
    if not current_actor_state:
        print("Warning: No current actor state found for conversation history update")
        return state
    
    # Update conversation history
    update_conversation_history(state, current_actor_state)
    
    return state

def player_deliberation(state: GameState) -> GameState:
    """Node for AI player to deliberate and formulate an argument"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state:
        print("Warning: No current actor state found for deliberation")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for deliberation")
        return state
    
    # Don't set phase here - it was already set by the previous node
    
    # Format objectives
    objectives_str = "\n".join([f"- {obj}" for obj in current_actor.objectives])
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    
    # Create deliberation chain
    deliberation_chain = DELIBERATION_PROMPT | llm.with_structured_output(ArgumentResponse)
    
    try:
        argument_response = deliberation_chain.invoke({
            "game_name": state.game_definition.name,
            "game_background": state.game_definition.background_briefing,
            "turn_length": state.game_definition.turn_length,
            "actor_name": current_actor.actor_name,
            "actor_briefing": current_actor.actor_briefing,
            "objectives": objectives_str,
            "conversation_history": current_actor_state.conversation_history
        })
        
        # Add scratchpad notes
        if argument_response.scratchpad_notes.strip():
            current_actor_state.internal_scratchpad.append(f"Turn {state.current_turn}: {argument_response.scratchpad_notes}")
        
        # Create the appropriate argument type
        argument_id = str(uuid.uuid4())
        
        if argument_response.type == "SecretArgument":
            argument = SecretArgument(
                argument_id=argument_id,
                proposing_actor_name=current_actor.actor_name,
                turn_proposed=state.current_turn,
                action_description=argument_response.action_description,
                pros=argument_response.pros,
                trigger_conditions=argument_response.trigger_conditions,
                status=ArgumentStatus.PROPOSED
            )
        else:
            argument = StandardArgument(
                argument_id=argument_id,
                proposing_actor_name=current_actor.actor_name,
                turn_proposed=state.current_turn,
                action_description=argument_response.action_description,
                pros=argument_response.pros,
                status=ArgumentStatus.PROPOSED
            )
        
        # Store the argument in the actor state
        current_actor_state.argument = argument
        
        # Add the assistant's response to conversation history
        pros_str = '\n'.join([f"- {pro}" for pro in argument.pros])
        
        assistant_response = f"""I propose the following action:

Action: {argument.action_description}

Reasons supporting this action:
{pros_str}"""
        
        if argument_response.scratchpad_notes.strip():
            assistant_response += f"\n\nNotes for future planning: {argument_response.scratchpad_notes}"
        
        current_actor_state.conversation_history.append(("assistant", assistant_response))
        
    except Exception as e:
        print(f"Error in player deliberation: {e}")
        # Create a default argument to prevent the game from breaking
        argument = StandardArgument(
            argument_id=str(uuid.uuid4()),
            proposing_actor_name=current_actor.actor_name,
            turn_proposed=state.current_turn,
            action_description="Continue current operations and maintain position",
            pros=["Maintain stability", "Preserve resources"],
            status=ArgumentStatus.PROPOSED
        )
        current_actor_state.argument = argument
    
    return state

def validate_secret_argument(state: GameState) -> GameState:
    """Node to validate if a secret argument is truly appropriate"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for secret validation")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for secret validation")
        return state
    
    current_argument = current_actor_state.argument
    
    # Only validate if it's a secret argument
    if not isinstance(current_argument, SecretArgument):
        return state
    
    # Prepare context
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Turn: {state.current_turn}
Game State: {state.game_state_summary}
"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.)
    
    # Create validation chain
    validation_chain = SECRET_VALIDATION_PROMPT | llm.with_structured_output(SecretArgumentValidationResponse)
    
    try:
        validation_response = validation_chain.invoke({
            "game_context": game_context,
            "actor_name": current_actor.actor_name,
            "action_description": current_argument.action_description,
            "trigger_conditions": current_argument.trigger_conditions,
            "pros": current_argument.pros
        })
        
        # If not a valid secret, convert to standard argument
        if not validation_response.is_valid_secret:
            # Create new standard argument with same details
            new_argument = StandardArgument(
                argument_id=current_argument.argument_id,
                proposing_actor_name=current_argument.proposing_actor_name,
                turn_proposed=current_argument.turn_proposed,
                action_description=current_argument.action_description,
                pros=current_argument.pros,
                status=ArgumentStatus.PROPOSED
            )
            current_actor_state.argument = new_argument
            
            # Log the conversion
            conversion_log = LogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                turn=state.current_turn,
                phase=state.current_phase,
                entry_type=LogEntryType.UMPIRE_RULING,
                actor_name=current_actor.actor_name,
                content=f"Secret argument converted to standard argument: {validation_response.reasoning}",
                summary=f"Secret argument validation: {current_actor.actor_name}"
            )
            state.game_log.append(conversion_log)
        
    except Exception as e:
        print(f"Error in secret argument validation: {e}")
        # If validation fails, default to keeping it as secret
    
    return state

def check_big_project(state: GameState) -> GameState:
    """Node to check if the argument is a big project that should be broken down"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for big project check")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for big project check")
        return state
    
    current_argument = current_actor_state.argument
    
    # Prepare context
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Turn: {state.current_turn}
"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    
    # Create big project check chain
    big_project_chain = BIG_PROJECT_CHECK_PROMPT | llm.with_structured_output(BigProjectCheckResponse)
    
    try:
        big_project_response = big_project_chain.invoke({
            "game_context": game_context,
            "actor_name": current_actor.actor_name,
            "turn_length": state.game_definition.turn_length,
            "action_description": current_argument.action_description,
            "pros": current_argument.pros
        })
        
        # If it's a big project, automatically replace with first stage action
        if big_project_response.is_big_project and big_project_response.first_stage_action:
            # Replace the current argument's action with the first stage action
            original_action = current_argument.action_description
            current_argument.action_description = big_project_response.first_stage_action
            
            # Inject the remaining plan into the scratchpad with appropriate framing
            if big_project_response.remaining_plan:
                scratchpad_note = f"Turn {state.current_turn}: Original ambitious plan was '{original_action}'. Breaking this down into stages - completed first stage this turn. Future plan (may adapt based on other players' actions): {big_project_response.remaining_plan}"
            else:
                scratchpad_note = f"Turn {state.current_turn}: Original ambitious plan was '{original_action}'. Breaking this down into stages - completed first stage this turn. Will reassess next steps based on results and other players' actions."
            
            current_actor_state.internal_scratchpad.append(scratchpad_note)
            
            # Log the automatic breakdown
            breakdown_log = LogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                turn=state.current_turn,
                phase=state.current_phase,
                entry_type=LogEntryType.UMPIRE_RULING,
                actor_name=current_actor.actor_name,
                content=f"Big project automatically broken down into first stage: {big_project_response.reasoning}. Action changed to: {big_project_response.first_stage_action}",
                summary=f"Big project breakdown: {current_actor.actor_name}"
            )
            state.game_log.append(breakdown_log)
        
    except Exception as e:
        print(f"Error in big project check: {e}")
        # Continue without breaking down the project
    
    return state

def finalize_argument(state: GameState) -> GameState:
    """Node to finalize the argument and prepare for adjudication"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for argument finalization")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for argument finalization")
        return state
    
    current_argument = current_actor_state.argument
    
    # All arguments (including secret ones) go through normal adjudication
    current_argument.status = ArgumentStatus.UNDER_REVIEW
    
    # Set phase for what's coming next (adjudication subgraph)
    state.current_phase = GamePhase.ADJUDICATION
    
    return state

# --- CONDITIONAL EDGES ---

def is_secret_argument(state: GameState) -> str:
    """Conditional edge to determine if we need to validate a secret argument"""
    current_actor_state = state.current_actor_state
    if current_actor_state and current_actor_state.argument:
        if isinstance(current_actor_state.argument, SecretArgument):
            return "validate_secret"
        else:
            return "check_big_project"
    return "check_big_project"

# --- GRAPH CONSTRUCTION ---

def create_argumentation_graph() -> StateGraph:
    """Create the argumentation workflow graph"""
    
    # Create the graph
    workflow = StateGraph(GameState)
    
    # Add nodes
    workflow.add_node("update_conversation_history", update_actor_conversation_history)
    workflow.add_node("player_deliberation", player_deliberation)
    workflow.add_node("validate_secret", validate_secret_argument)
    workflow.add_node("check_big_project", check_big_project)
    workflow.add_node("finalize_argument", finalize_argument)
    
    # Add edges
    workflow.add_edge(START, "update_conversation_history")
    workflow.add_edge("update_conversation_history", "player_deliberation")
    
    # After deliberation, check if it's a secret argument
    workflow.add_conditional_edges(
        "player_deliberation",
        is_secret_argument,
        {
            "validate_secret": "validate_secret",
            "check_big_project": "check_big_project"
        }
    )
    
    # After secret validation, check for big project
    workflow.add_edge("validate_secret", "check_big_project")
    
    # After big project check, either restart or finalize
    workflow.add_edge("check_big_project", "finalize_argument")
    
    workflow.add_edge("finalize_argument", END)
    
    return workflow.compile() 