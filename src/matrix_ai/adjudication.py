from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import random
import statistics
import uuid
from datetime import datetime

from .schemas import (
    GameState, ArgumentStatus, AdjudicationMethod, LogEntry, LogEntryType, 
    CriticResponse, AdjudicationMethodResponse, EstProbabilityResponse,
    SecretArgumentTriggerResponse, GamePhase
)

# --- PROMPTS ---

SECRET_TRIGGER_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a secret argument evaluator for a matrix wargame. Your role is to determine if the current proposed action would trigger any pending secret arguments from any actor in the game.

Review all pending secret arguments from all actors and their trigger conditions. Determine which (if any) would be triggered by the proposed action.

Be precise - only trigger secret arguments when the proposed action clearly meets their trigger conditions. Consider:
- Does the proposed action directly relate to the trigger conditions?
- Would the attempt (regardless of success) trigger the secret?
- Are the circumstances described in the trigger conditions met?

Game Context:
{game_context}"""),
    ("human", """Proposed Action:
Actor: {actor_name}
Action: {action_description}
Supporting Reasons (Pros): {pros}

All Pending Secret Arguments (from all actors):
{pending_secrets}

Determine which secret arguments (if any) would be triggered by attempting this action.""")
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a critic AI in a matrix wargame. Your role is to provide counter-arguments (cons) for proposed actions by identifying obstacles that could prevent successful execution of the action.

Review the following argument and provide realistic reasons why the actor might FAIL TO SUCCESSFULLY EXECUTE this action. Focus on:
- Insufficient resources, authority, or capabilities of the actor
- Technical/practical barriers or limitations preventing execution  
- Opposition or interference from other actors that could block execution
- External obstacles, timing issues, or unfavorable conditions
- Missing prerequisites, access, or support needed for execution
- Logistical constraints or operational barriers
- Any revealed secret arguments that would interfere with execution

DO NOT focus on whether it's strategically wise or what the long-term consequences might be. Focus ONLY on obstacles to the actor successfully carrying out the specific action as proposed.

Be constructive but realistic. Identify legitimate execution obstacles rather than strategic opposition to outcomes.

Game Context:
{game_context}

Current Actor: {actor_name}
Actor Objectives: {actor_objectives}

Triggered Secret Arguments: {triggered_secrets}"""),
    ("human", """Argument to Review:
Action: {action_description}
Execution Reasons (Pros): {pros}

Please provide counter-arguments (cons) identifying obstacles that could prevent successful execution of this action.""")
])

ADJUDICATION_METHOD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an umpire AI determining how to adjudicate an argument in a matrix wargame.

Choose the adjudication method:
- AUTO_SUCCESS: Use when the actor is very likely to successfully execute the action with minimal execution obstacles
- ESTIMATIVE_PROBABILITY: Use when there are meaningful execution obstacles or uncertainties about successful completion

Consider the likelihood of successful execution:
- Does the actor have sufficient resources, authority, and capabilities?
- Are there significant execution obstacles identified in the cons?
- What external challenges or opposition might interfere with execution?
- Are there technical, practical, or logistical barriers?
- Impact of any triggered secret arguments on execution success

Focus on likelihood of successful completion of the action, not strategic wisdom or consequences.
     
Simple actions like "making a public statement" should succeed automatically unless there are specific execution barriers (e.g., lack of communication channels, censorship, technical failures, etc.).

Game Context:
{game_context}

Triggered Secret Arguments: {triggered_secrets}"""),
    ("human", """Argument to Adjudicate:
Action: {action_description}
Execution Reasons (Pros): {pros}
Execution Obstacles (Cons): {cons}

Choose the appropriate adjudication method based on the likelihood of successful execution of this action.""")
])

PROBABILITY_ESTIMATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert analyst estimating the probability of successful execution for a proposed action in a matrix wargame.

Provide a probability between 0.0 and 1.0 based on the likelihood of successful execution:
- Actor's resources, authority, and capabilities needed for the action
- Technical/practical feasibility of the proposed action
- External obstacles and challenges that could interfere with execution
- Opposition or interference from other actors
- Support, favorable conditions, or assets that aid execution
- Logistical and operational requirements vs. available resources
- Impact of any triggered secret arguments on execution success

Focus on the likelihood that the action will be successfully COMPLETED as proposed, not on strategic consequences or whether it's a good idea. Simple actions like "making a public statement" should have very high success probability unless there are specific execution barriers (e.g., lack of communication channels, censorship, technical failures, etc.).

Be realistic and well-reasoned in your assessment of execution likelihood.

Game Context:
{game_context}

Current Actor: {actor_name}
Actor Capabilities: {actor_forces}

Triggered Secret Arguments: {triggered_secrets}"""),
    ("human", """Argument to Assess:
Action: {action_description}
Execution Reasons (Pros): {pros}
Execution Obstacles (Cons): {cons}

Estimate the probability of successful execution (0.0 to 1.0) based on the likelihood of completing this action and provide your reasoning.""")
])

# --- NODE FUNCTIONS ---

def check_secret_triggers(state: GameState) -> GameState:
    """Node to check if the proposed action triggers any secret arguments from any actor"""
    
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for secret trigger check")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for secret trigger check")
        return state
    
    current_argument = current_actor_state.argument
    
    # Collect ALL pending secret arguments from ALL actors
    all_pending_secrets = []
    for actor_state in state.actor_states:
        for secret_arg in actor_state.pending_secret_arguments:
            if not secret_arg.is_triggered:
                all_pending_secrets.append({
                    "argument_id": secret_arg.argument_id,
                    "actor_name": secret_arg.proposing_actor_name,
                    "action": secret_arg.action_description,
                    "trigger_conditions": secret_arg.trigger_conditions
                })
    
    if not all_pending_secrets:
        # No pending secrets, continue without triggering
        return state
    
    # Prepare context
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Turn: {state.current_turn}
Current Game State: {state.game_state_summary}
"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    
    # Create secret trigger chain
    trigger_chain = SECRET_TRIGGER_CHECK_PROMPT | llm.with_structured_output(SecretArgumentTriggerResponse)
    
    try:
        trigger_response = trigger_chain.invoke({
            "game_context": game_context,
            "actor_name": current_actor.actor_name,
            "action_description": current_argument.action_description,
            "pros": current_argument.pros,
            "pending_secrets": all_pending_secrets
        })
        
        # Process triggered secret arguments
        for argument_id in trigger_response.triggered_arguments:
            # Search through ALL actors' pending secrets
            for actor_state in state.actor_states:
                for secret_arg in actor_state.pending_secret_arguments:
                    if secret_arg.argument_id == argument_id and not secret_arg.is_triggered:
                        secret_arg.is_triggered = True
                        secret_arg.is_revealed = True
                        
                        # Add to triggered secrets info
                        state.triggered_secrets_this_turn.append(f"{secret_arg.proposing_actor_name}: {secret_arg.action_description}")
                        
                        # Create log entry for triggered secret argument
                        trigger_log = LogEntry(
                            entry_id=str(uuid.uuid4()),
                            timestamp=datetime.now().isoformat(),
                            turn=state.current_turn,
                            phase=state.current_phase,
                            entry_type=LogEntryType.GAME_EVENT,
                            actor_name=secret_arg.proposing_actor_name,
                            content=f"Secret argument triggered by {current_actor.actor_name}'s proposed action: {secret_arg.action_description}",
                            summary=f"Secret argument revealed: {secret_arg.proposing_actor_name}"
                        )
                        state.game_log.append(trigger_log)
                        break
        
    except Exception as e:
        print(f"Error checking secret triggers: {e}")
        # Continue without triggering secrets
    
    return state

def gather_critic_feedback(state: GameState) -> GameState:
    """Node to gather critic feedback on the argument"""
    
    # Don't set phase here - phases only change between subgraphs
    
    # Get current actor and their argument
    current_actor_state = state.current_actor_state
    current_actor = state.current_actor_definition
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for critic feedback")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for critic feedback")
        return state
    
    current_argument = current_actor_state.argument
    
    # Prepare context for critic
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Turn: {state.current_turn}
Game State Summary: {state.game_state_summary}
"""
    
    # Get triggered secrets info
    triggered_secrets = state.triggered_secrets_this_turn
    triggered_secrets_str = "\n".join(triggered_secrets) if triggered_secrets else "None"
    
    # Initialize LLM for critic
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    
    # Create critic chain
    critic_chain = CRITIC_PROMPT | llm.with_structured_output(CriticResponse)
    
    try:
        # Get critic response
        critic_response = critic_chain.invoke({
            "game_context": game_context,
            "actor_name": current_actor.actor_name,
            "actor_objectives": current_actor.objectives,
            "action_description": current_argument.action_description,
            "pros": current_argument.pros,
            "triggered_secrets": triggered_secrets_str
        })
        
        # Update argument with cons
        current_argument.cons.extend(critic_response.cons)
        current_argument.status = ArgumentStatus.UNDER_REVIEW
        
    except Exception as e:
        print(f"Error in critic feedback: {e}")
        # Continue without critic feedback
    
    # Don't set phase here - only at subgraph boundaries
    
    return state

def determine_adjudication_method(state: GameState) -> GameState:
    """Node to determine the adjudication method"""
    
    # Don't set phase here - it was already set by the previous node
    
    current_actor_state = state.current_actor_state
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for adjudication method determination")
        return state
    
    current_argument = current_actor_state.argument
    
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Turn: {state.current_turn}
"""
    
    # Get triggered secrets info
    triggered_secrets = state.triggered_secrets_this_turn
    triggered_secrets_str = "\n".join(triggered_secrets) if triggered_secrets else "None"
    
    # Initialize LLM for umpire
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    
    # Create adjudication method chain
    method_chain = ADJUDICATION_METHOD_PROMPT | llm.with_structured_output(AdjudicationMethodResponse)
    
    try:
        method_response = method_chain.invoke({
            "game_context": game_context,
            "action_description": current_argument.action_description,
            "pros": current_argument.pros,
            "cons": current_argument.cons,
            "triggered_secrets": triggered_secrets_str
        })
        
        current_argument.adjudication_method = method_response.method
        current_argument.status = ArgumentStatus.AWAITING_ADJUDICATION
        
    except Exception as e:
        print(f"Error determining adjudication method: {e}")
        # Default to estimative probability
        current_argument.adjudication_method = AdjudicationMethod.ESTIMATIVE_PROBABILITY
        current_argument.status = ArgumentStatus.AWAITING_ADJUDICATION
    
    return state

def handle_auto_success(state: GameState) -> GameState:
    """Node to handle auto success adjudication"""
    
    current_actor_state = state.current_actor_state
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for auto success handling")
        return state
    
    current_argument = current_actor_state.argument
    
    # Update argument status
    current_argument.status = ArgumentStatus.ADJUDICATED_AUTO_SUCCESS
    current_argument.adjudication_method = AdjudicationMethod.AUTO_SUCCESS
    current_argument.is_successful = True
    
    # Set phase for what's coming next (state update subgraph)
    state.current_phase = GamePhase.STATE_UPDATE
    
    return state

def estimate_probability(state: GameState) -> GameState:
    """Node to gather probability estimates from AI panel"""
    
    current_actor = state.current_actor_definition
    current_actor_state = state.current_actor_state
    
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for probability estimation")
        return state
    
    if not current_actor:
        print("Warning: No current actor definition found for probability estimation")
        return state
    
    current_argument = current_actor_state.argument
    
    game_context = f"""
Game: {state.game_definition.name}
Background: {state.game_definition.background_briefing}
Current Turn: {state.current_turn}
Game State Summary: {state.game_state_summary}
"""
    
    # Get triggered secrets info
    triggered_secrets = state.triggered_secrets_this_turn
    triggered_secrets_str = "\n".join(triggered_secrets) if triggered_secrets else "None"
    
    # Initialize LLM for probability estimation
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
    
    # Create probability estimation chain
    prob_chain = PROBABILITY_ESTIMATION_PROMPT | llm.with_structured_output(EstProbabilityResponse)
    
    # Prepare batch inputs for multiple estimates (simulating AI panel)
    num_estimates = 3
    batch_inputs = []
    for i in range(num_estimates):
        batch_inputs.append({
            "game_context": game_context,
            "actor_name": current_actor.actor_name,
            "actor_forces": [f.unit_name for f in current_actor_state.current_forces],
            "action_description": current_argument.action_description,
            "pros": current_argument.pros,
            "cons": current_argument.cons,
            "triggered_secrets": triggered_secrets_str
        })
    
    try:
        # Use batch to get all estimates at once
        estimates = prob_chain.batch(batch_inputs)
        
        # Process the batch results
        for prob_response in estimates:
            current_argument.probability_estimates.append(prob_response.success_probability)
            
    except Exception as e:
        print(f"Error in batch probability estimation: {e}")
        # Add default estimates
        estimates = []
        for i in range(num_estimates):
            current_argument.probability_estimates.append(0.5)
            estimates.append(EstProbabilityResponse(
                success_probability=0.5,
                reasoning="Default estimate due to estimation error"
            ))
    
    # Calculate median probability
    probabilities = [est.success_probability for est in estimates]
    current_argument.final_probability = statistics.median(probabilities)
    
    return state

def evaluate_success(state: GameState) -> GameState:
    """Node to evaluate success based on estimated probability"""
    
    current_actor_state = state.current_actor_state
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for success evaluation")
        return state
    
    current_argument = current_actor_state.argument
    
    if current_argument.final_probability is None:
        current_argument.final_probability = 0.5
    
    # Use a threshold approach instead of dice rolling
    threshold = random.random()
    is_successful = threshold <= current_argument.final_probability
    
    # Update argument status
    if is_successful:
        current_argument.status = ArgumentStatus.ADJUDICATED_SUCCESS
    else:
        current_argument.status = ArgumentStatus.ADJUDICATED_FAILURE
    
    current_argument.adjudication_method = AdjudicationMethod.ESTIMATIVE_PROBABILITY
    current_argument.is_successful = is_successful
    
    # Set phase for what's coming next (state update subgraph)
    state.current_phase = GamePhase.STATE_UPDATE
    
    return state

# --- CONDITIONAL EDGES ---

def should_auto_succeed(state: GameState) -> str:
    """Conditional edge to determine if argument should auto-succeed"""
    current_actor_state = state.current_actor_state
    if not current_actor_state or not current_actor_state.argument:
        print("Warning: No current actor state or argument found for adjudication method routing")
        return "estimate_probability"
    
    current_argument = current_actor_state.argument
    if current_argument.adjudication_method == AdjudicationMethod.AUTO_SUCCESS:
        return "auto_success"
    else:
        return "estimate_probability"

# --- GRAPH CONSTRUCTION ---

def create_adjudication_graph() -> StateGraph:
    """Create the adjudication workflow graph"""
    
    # Create the graph
    workflow = StateGraph(GameState)
    
    # Add nodes
    workflow.add_node("check_secret_triggers", check_secret_triggers)
    workflow.add_node("gather_critics", gather_critic_feedback)
    workflow.add_node("determine_method", determine_adjudication_method)
    workflow.add_node("auto_success", handle_auto_success)
    workflow.add_node("estimate_probability", estimate_probability)
    workflow.add_node("evaluate_success", evaluate_success)
    
    # Add edges
    workflow.add_edge(START, "check_secret_triggers")
    workflow.add_edge("check_secret_triggers", "gather_critics")
    workflow.add_edge("gather_critics", "determine_method")
    workflow.add_conditional_edges(
        "determine_method",
        should_auto_succeed,
        {
            "auto_success": "auto_success",
            "estimate_probability": "estimate_probability"
        }
    )
    workflow.add_edge("auto_success", END)
    workflow.add_edge("estimate_probability", "evaluate_success")
    workflow.add_edge("evaluate_success", END)
    
    return workflow.compile()