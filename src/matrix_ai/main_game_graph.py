from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import random
import uuid
from datetime import datetime

from .schemas import (
    GameState, GamePhase, LogEntry, LogEntryType, Actor,
    GameOverCheckResponse, EndGameAssessmentResponse
)
from .argumentation import create_argumentation_graph
from .adjudication import create_adjudication_graph  
from .scenario_update import create_scenario_update_graph

# --- PROMPTS ---

GAME_OVER_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an umpire AI evaluating whether a matrix wargame should end based on the current game state and objectives.

Consider:
- Whether any actor has clearly achieved their primary objectives
- Whether the situation has become unwinnable or deadlocked for all parties
- Whether continuing would not meaningfully advance the narrative or objectives
- The current strategic situation and whether further turns would be productive

NOTE: The maximum turn limit check has already been performed - you are evaluating based on objective achievement and game dynamics only.

Game Context:
{game_context}

Current Turn: {current_turn}
Maximum Turns: {max_turns}

For each actor, consider their objectives and current status:
{actor_objectives_status}"""),
    ("human", """Current Game State:
{game_state_summary}

Global Narrative Markers:
{global_markers}

Should the game end based on objective achievement or strategic deadlock? Provide your assessment.""")
])

END_GAME_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an umpire AI providing final assessment for a completed matrix wargame.

Review each actor's performance against their stated objectives and provide:
1. Objective achievement assessment for each actor
2. Overall strategic outcome analysis
3. Key turning points and decisive moments
4. Strategic lessons learned
5. Narrative summary of the game's conclusion

Game Context:
{game_context}

Actor Objectives:
{actor_objectives}"""),
    ("human", """Final Game State:
{game_state_summary}

Global Narrative Markers:
{global_markers}

Game Log Summary:
{game_log_summary}

Provide a comprehensive final assessment of the game outcome.""")
])

# --- NODE FUNCTIONS ---

def establish_turn_order(state: GameState) -> GameState:
    """Node to establish initial turn order (can be random or fixed)"""
    
    if not state.turn_order:
        # Create initial turn order based on actor definitions
        num_actors = len(state.game_definition.actors)
        state.turn_order = list(range(num_actors))
        
        # Could randomize here if desired
        # random.shuffle(state.turn_order)
    
    state.active_player_queue_index = 0
    
    # Create initial log entry
    turn_order_names = [state.game_definition.actors[i].actor_name for i in state.turn_order]
    log_entry = LogEntry(
        entry_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        turn=state.current_turn,
        phase=GamePhase.SETUP,
        entry_type=LogEntryType.GAME_EVENT,
        content=f"Turn order established: {' -> '.join(turn_order_names)}",
        summary="Game started - turn order established"
    )
    state.game_log.append(log_entry)
    
    # Set phase for what's coming next (argumentation)
    state.current_phase = GamePhase.ARGUMENTATION
    
    return state

def advance_to_next_player(state: GameState) -> GameState:
    """Node to advance to the next player's turn"""
    
    # Move to next player in turn order
    state.active_player_queue_index = (state.active_player_queue_index + 1) % len(state.turn_order)
    
    # If we've completed a full round, advance to next turn
    if state.active_player_queue_index == 0:
        state.current_turn += 1
        
        # Log turn advancement
        log_entry = LogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            turn=state.current_turn,
            phase=GamePhase.PLAYER_TURN_START,
            entry_type=LogEntryType.GAME_EVENT,
            content=f"Advanced to turn {state.current_turn}",
            summary=f"Turn {state.current_turn} started"
        )
        state.game_log.append(log_entry)
    
    # Set phase for what's coming next (argumentation by next player)
    state.current_phase = GamePhase.ARGUMENTATION
    
    return state

def check_game_over(state: GameState) -> GameState:
    """Node to check if game over conditions are met"""
    
    # Check if we're at the end of the final turn
    # (current turn equals max turns AND the last player just finished)
    is_last_player = state.active_player_queue_index == len(state.turn_order) - 1
    is_final_turn = state.current_turn == state.game_definition.game_length
    
    if is_final_turn and is_last_player:
        state.current_phase = GamePhase.FINAL_REPORTING
        
        # Log game over due to turn limit
        log_entry = LogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            turn=state.current_turn,
            phase=state.current_phase,
            entry_type=LogEntryType.UMPIRE_RULING,
            content=f"Game ended: Maximum turn limit of {state.game_definition.game_length} reached",
            summary="Game over - maximum turns reached"
        )
        state.game_log.append(log_entry)
        return state
    
    # Only do AI-based game over check at the end of complete turns
    # (when the last player has finished their turn)
    if not is_last_player:
        # Not at end of turn yet, continue
        return state
    
    # Second check: Use LLM to evaluate objective achievement and deadlock
    # This only runs at the end of complete turns (after last player)
    
    # Prepare context for game over check
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Turn Length: {state.game_definition.turn_length}
"""
    
    # Format actor objectives and status
    actor_objectives_status = []
    for i, actor_def in enumerate(state.game_definition.actors):
        actor_state = state.actor_states[i]
        objectives_str = "\n".join([f"  - {obj}" for obj in actor_def.objectives])
        effects_str = "\n".join([f"  - {effect}" for effect in actor_state.effects])
        
        actor_status = f"""
{actor_def.actor_name}:
Objectives:
{objectives_str}
Current Status/Effects:
{effects_str}
"""
        actor_objectives_status.append(actor_status)
    
    actor_objectives_status_str = "\n".join(actor_objectives_status)
    global_markers_str = "\n".join(state.global_narrative_markers) if state.global_narrative_markers else "None"
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    
    # Create game over check chain with structured output
    game_over_chain = GAME_OVER_CHECK_PROMPT | llm.with_structured_output(GameOverCheckResponse)
    
    try:
        response = game_over_chain.invoke({
            "game_context": game_context,
            "max_turns": state.game_definition.game_length,
            "current_turn": state.current_turn,
            "actor_objectives_status": actor_objectives_status_str,
            "game_state_summary": state.game_state_summary,
            "global_markers": global_markers_str
        })
        
        if response.should_end_game:
            state.current_phase = GamePhase.FINAL_REPORTING
            
            # Log game over decision
            log_entry = LogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                turn=state.current_turn,
                phase=state.current_phase,
                entry_type=LogEntryType.UMPIRE_RULING,
                content=f"Game over conditions met: {response.reasoning}. Objectives achieved by: {', '.join(response.objectives_achieved) if response.objectives_achieved else 'None'}",
                summary="Game over - objectives achieved or deadlock"
            )
            state.game_log.append(log_entry)
        
    except Exception as e:
        print(f"Error in game over check: {e}")
        # Fallback: continue game unless at turn limit
        # The turn limit was already checked above
    
    return state

def end_game_sequence(state: GameState) -> GameState:
    """Node to conduct final game assessment and reporting"""
    
    state.current_phase = GamePhase.FINAL_REPORTING
    
    # Prepare context for final assessment
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Game Length: {state.game_definition.game_length} turns
Actual Duration: {state.current_turn} turns
"""
    
    # Format actor objectives
    actor_objectives = []
    for actor_def in state.game_definition.actors:
        objectives_str = "\n".join([f"  - {obj}" for obj in actor_def.objectives])
        actor_objectives.append(f"{actor_def.actor_name}:\n{objectives_str}")
    
    actor_objectives_str = "\n\n".join(actor_objectives)
    
    # Create summary of game log
    recent_entries = state.game_log[-20:] if len(state.game_log) > 20 else state.game_log
    game_log_summary = "\n".join([entry.summary for entry in recent_entries if entry.summary])
    
    global_markers_str = "\n".join(state.global_narrative_markers) if state.global_narrative_markers else "None"
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
    
    # Create end game assessment chain with structured output
    assessment_chain = END_GAME_ASSESSMENT_PROMPT | llm.with_structured_output(EndGameAssessmentResponse)
    
    try:
        final_assessment = assessment_chain.invoke({
            "game_context": game_context,
            "actor_objectives": actor_objectives_str,
            "game_state_summary": state.game_state_summary,
            "global_markers": global_markers_str,
            "game_log_summary": game_log_summary
        })
        
        # Create comprehensive final log entry using structured assessment
        assessment_content = f"""FINAL GAME ASSESSMENT

{final_assessment.game_outcome_summary}

ACTOR PERFORMANCE:
""" + "\n".join([
    f"""
{assessment.actor_name}: {assessment.overall_performance}
  Objectives Achieved: {', '.join(assessment.objectives_achieved) if assessment.objectives_achieved else 'None'}
  Objectives Failed: {', '.join(assessment.objectives_failed) if assessment.objectives_failed else 'None'}
  Key Accomplishments: {', '.join(assessment.key_accomplishments) if assessment.key_accomplishments else 'None'}
  Major Setbacks: {', '.join(assessment.major_setbacks) if assessment.major_setbacks else 'None'}"""
    for assessment in final_assessment.actor_assessments
]) + f"""

KEY TURNING POINTS:
{chr(10).join([f"- {point}" for point in final_assessment.key_turning_points])}

STRATEGIC LESSONS:
{chr(10).join([f"- {lesson}" for lesson in final_assessment.strategic_lessons])}

WINNERS AND LOSERS:
{final_assessment.winners_and_losers}

NARRATIVE CONCLUSION:
{final_assessment.narrative_conclusion}"""
        
        final_log = LogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            turn=state.current_turn,
            phase=GamePhase.GAME_ENDED,
            entry_type=LogEntryType.UMPIRE_RULING,
            actor_name=None,
            content=assessment_content,
            summary="Final game assessment completed"
        )
        state.game_log.append(final_log)
        
    except Exception as e:
        print(f"Error in final assessment: {e}")
        # Create simple fallback assessment
        final_log = LogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            turn=state.current_turn,
            phase=GamePhase.GAME_ENDED,
            entry_type=LogEntryType.UMPIRE_RULING,
            content=f"Game concluded after {state.current_turn} turns. Final state: {state.game_state_summary}",
            summary="Game ended - basic assessment"
        )
        state.game_log.append(final_log)
    
    state.current_phase = GamePhase.GAME_ENDED
    return state

# --- CONDITIONAL EDGES ---

def is_game_over(state: GameState) -> str:
    """Conditional edge to determine if game should end"""
    if state.current_phase == GamePhase.FINAL_REPORTING:
        return "end_game"
    else:
        return "continue_game"

# --- GRAPH CONSTRUCTION ---

def create_main_game_graph() -> StateGraph:
    """Create the main game workflow graph that combines all modules"""
    
    # Create the main graph
    workflow = StateGraph(GameState)
    
    # Create subgraphs for the three main phases
    argumentation_graph = create_argumentation_graph()
    adjudication_graph = create_adjudication_graph()
    scenario_update_graph = create_scenario_update_graph()
    
    # Add main game flow nodes
    workflow.add_node("establish_turn_order", establish_turn_order)
    workflow.add_node("next_player_turn", advance_to_next_player)
    workflow.add_node("check_game_over", check_game_over)
    workflow.add_node("end_game_sequence", end_game_sequence)
    
    # Add subgraph nodes
    workflow.add_node("argumentation", argumentation_graph)
    workflow.add_node("adjudication", adjudication_graph)
    workflow.add_node("scenario_update", scenario_update_graph)
    
    # Set up main game flow according to updated flowchart
    workflow.add_edge(START, "establish_turn_order")
    workflow.add_edge("establish_turn_order", "argumentation")
    
    # Main game loop: argumentation -> adjudication -> scenario_update -> check_game_over
    workflow.add_edge("argumentation", "adjudication")
    workflow.add_edge("adjudication", "scenario_update")
    workflow.add_edge("scenario_update", "check_game_over")
    
    # Game over check branches
    workflow.add_conditional_edges(
        "check_game_over",
        is_game_over,
        {
            "end_game": "end_game_sequence",
            "continue_game": "next_player_turn"
        }
    )

    workflow.add_edge("next_player_turn", "argumentation")

    # End game
    workflow.add_edge("end_game_sequence", END)
    
    return workflow.compile()

# --- HELPER FUNCTIONS ---

def run_matrix_game(game_definition, max_turns=None, checkpointer=None):
    """
    Helper function to run a complete matrix game
    
    Args:
        game_definition: MatrixGame object defining the game setup
        max_turns: Optional override for maximum turns (uses game_definition.game_length if None)
        checkpointer: Optional checkpointer for persistence
    
    Returns:
        Final GameState after game completion
    """
    
    # Override game length if specified
    if max_turns is not None:
        game_definition.game_length = max_turns
    
    # Initialize game state
    initial_state = GameState.from_matrix_game_setup(game_definition)
    
    # Create and compile the graph
    graph = create_main_game_graph()
    
    # Run the game with increased recursion limit
    config = {
        "thread_id": str(uuid.uuid4()),
        "recursion_limit": 600
    }
    
    if checkpointer:
        final_state = graph.invoke(initial_state, config=config, checkpointer=checkpointer)
    else:
        final_state = graph.invoke(initial_state, config=config)
    
    return final_state

def stream_matrix_game(game_definition, max_turns=None, checkpointer=None, stream_mode="updates"):
    """
    Helper function to stream a matrix game execution
    
    Args:
        game_definition: MatrixGame object defining the game setup
        max_turns: Optional override for maximum turns
        checkpointer: Optional checkpointer for persistence
        stream_mode: Streaming mode - "updates", "values", "messages", "custom", or "debug"
    
    Yields:
        GameState updates as the game progresses
    """
    
    # Override game length if specified
    if max_turns is not None:
        game_definition.game_length = max_turns
    
    # Initialize game state
    initial_state = GameState.from_matrix_game_setup(game_definition)
    
    # Create and compile the graph
    graph = create_main_game_graph()
    
    # Stream the game with increased recursion limit
    config = {
        "thread_id": str(uuid.uuid4()),
        "recursion_limit": 600
    }
    
    if checkpointer:
        for state in graph.stream(initial_state, config=config, checkpointer=checkpointer, stream_mode=stream_mode):
            yield state
    else:
        for state in graph.stream(initial_state, config=config, stream_mode=stream_mode):
            yield state


graph = create_main_game_graph()