from typing import List, Optional, Literal, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum

# --- ENUMS for Game Mechanics ---

class ArgumentStatus(str, Enum):
    PROPOSED = "Proposed"
    UNDER_REVIEW = "Under Review" # e.g., Critic AIs are offering cons
    AWAITING_ADJUDICATION = "Awaiting Adjudication"
    ADJUDICATED_SUCCESS = "Adjudicated - Success"
    ADJUDICATED_FAILURE = "Adjudicated - Failure"
    ADJUDICATED_AUTO_SUCCESS = "Adjudicated - Auto Success" # Compelling/Unchallenged
    REJECTED = "Rejected" # e.g., not a valid argument type

class LogEntryType(str, Enum):
    ARGUMENT = "Argument"
    GAME_EVENT = "Game Event"
    UMPIRE_RULING = "Umpire Ruling"
    PLAYER_MESSAGE = "Player Message" # General messages from actors not fitting other types

class GamePhase(str, Enum):
    SETUP = "Setup"
    PLAYER_TURN_START = "Player Turn Start" # Before deliberation
    ARGUMENTATION = "Argumentation"
    ASSESSMENT = "Assessment"
    ADJUDICATION = "Adjudication"
    STATE_UPDATE = "State Update"
    CONSEQUENCE_APPLICATION = "Consequence Application" # Update state post-adjudication
    END_OF_TURN_REVIEW = "End of Turn Review" # Umpire review, trigger secret args
    GAME_OVER_CHECK = "Game Over Check"
    FINAL_REPORTING = "Final Reporting" # Players state final summaries, umpire assesses objectives
    GAME_ENDED = "Game Ended"

class AdjudicationMethod(str, Enum):
    ESTIMATIVE_PROBABILITY = "Estimative Probability"
    AUTO_SUCCESS = "Auto Success"

# --- SETUP / DEFINITION MODELS ---

class ForceUnit(BaseModel):
    unit_name: str = Field(description="The name or type of the unit/asset (e.g., 'Motorised Infantry Regiment 3', 'Task Force 79.1', 'Submarine Force', 'Chemical Weapons', 'Public Support Unit').")
    starting_location: str = Field(description="The starting location of the unit on the game map (e.g., 'Stanley', 'Argentina', 'Eastern Mediterranean', 'Sarajevo', 'Kaliningrad').")
    details: Optional[str] = Field(None, description="Any specific notes about this unit.")


class Actor(BaseModel):
    actor_name: str = Field(description="The specific name of the player/faction/role (e.g., 'UK Government - Mrs. Thatcher', 'Russia', 'Tribal Elder').")
    actor_briefing: str = Field(description="A detailed description of the actor's perspective, motivations, capabilities, constraints, relationships with other actors, and current situation within the game's context. Provide enough information for a player (or AI) to effectively role-play this actor. This briefing is for the player and can contain secret information not known to other actors.")
    objectives: List[str] = Field(description="A list of 3-5 specific, measurable, achievable, relevant, and time-bound (SMART-like, though flexibility is key in matrix games) objectives for this actor to pursue during the game. Objectives should create potential conflict and interaction with other actors.")
    starting_forces: List[ForceUnit] = Field(description="A list of units, assets, resources, or capabilities controlled by this actor at the start of the game.")


class MatrixGame(BaseModel):
    name: str = Field(description="Title of the game scenario.")
    category: str = Field(description="Category/domain of the scenario (e.g., Political, Economic, Military, Environmental, etc.).")
    description: str = Field(description="A short sentence describing the game scenario.")
    introduction: str = Field(description="A brief, engaging paragraph setting the stage for the game.")
    background_briefing: str = Field(description="Detailed context and background information necessary to understand the scenario. Include historical context, key events leading up to the game's start, the current geopolitical situation, and any relevant factors influencing the actors and the conflict. Should provide enough detail for players to understand the world they are stepping into.")
    actors: List[Actor] = Field(description="An ordered list defining the key players or factions involved in the game. Each actor represents a distinct entity (nation, organization, group, individual) whose actions drive the narrative. The list should be ordered by the sequence of play.")
    victory_conditions: Optional[str] = Field(default=None, description="How the game ends and how winners are determined, if applicable.")
    turn_length: str = Field(description="The conceptual duration each game turn represents (e.g., '2-4 weeks', '1 month').")
    game_length: int = Field(description="The maximum number of turns the game can last")
    designer_notes: Optional[str] = Field(default=None, description="Optional insights about the game's design purpose, expected outcomes, or historical parallels.")

# --- DYNAMIC / IN-GAME STATE MODELS ---

class ForceUnitState(BaseModel):
    """
    Represents a force unit during active gameplay, inheriting its base definition
    from ForceUnit and adding fields for dynamic state.
    """
    unit_name: str = Field(description="The name or type of the unit/asset (e.g., 'Motorised Infantry Regiment 3', 'Task Force 79.1', 'Submarine Force', 'Chemical Weapons', 'Public Support Unit').")
    location: str = Field(description="The unit's current location on the game map, which may change from its starting location.")
    details: Optional[str] = Field(None, description="Any specific notes about this unit.")

# --- ARGUMENT MODELS ---

class BaseArgument(BaseModel):
    """Base model for all argument types."""
    argument_id: str = Field(description="Unique identifier for the argument.") # Consider UUID
    proposing_actor_name: str = Field(description="Name of the actor proposing the argument.")
    turn_proposed: int = Field(description="Game turn when the argument was proposed.")
    action_description: str = Field(description="The action the actor wants to take and its intended effect.")
    pros: List[str] = Field(description="Reasons explaining why the action is likely to be successfully executed (Pros) - focus on capabilities, resources, authority, favorable conditions, support, etc.")
    cons: List[str] = Field(default_factory=list, description="Obstacles that could prevent successful execution of the action (Cons) - focus on resource limitations, opposition, technical barriers, unfavorable conditions, etc.")
    status: ArgumentStatus = Field(default=ArgumentStatus.PROPOSED, description="Current status of the argument.")
    adjudication_method: Optional[AdjudicationMethod] = Field(None, description="Method used for adjudication (e.g., 'AutoSuccess', 'EstimativeProbability').")
    adjudication_narrative: Optional[str] = Field(None, description="Narrative description of the adjudication outcome.")
    is_successful: Optional[bool] = Field(None, description="True if the argument succeeded, False if it failed, None if not yet adjudicated.")
    probability_estimates: List[float] = Field(default_factory=list, description="List of probability estimates from AI panel (0.0 to 1.0).")
    final_probability: Optional[float] = Field(None, description="Final aggregated probability of success (median of estimates).")

class StandardArgument(BaseArgument):
    pass

class SecretArgument(BaseArgument):
    trigger_conditions: str = Field(description="Conditions under which this secret argument is revealed or activated.")
    is_triggered: bool = Field(default=False, description="Whether the trigger conditions have been met.")
    is_revealed: bool = Field(default=False, description="Whether the argument has been revealed to other players (even if not yet successful).")

ArgumentVariant = Union[StandardArgument, SecretArgument]


class ActorState(BaseModel):
    """
    Represents the dynamic state of an actor during the game.
    It contains the actor's static definition and tracks changing elements.
    """
    actor_name: str = Field(description="The name of the actor. Matches the name in the game_definition.")
    current_forces: List[ForceUnitState] = Field(default_factory=list, description="The actor's forces currently in play with their dynamic states.")
    effects: List[str] = Field(default_factory=list, description="List of conditions, or narrative markers resulting from arguments (e.g., 'Police Reform Successful', 'National infrastructure upgraded') or other game events.")
    argument: Optional[ArgumentVariant] = Field(default=None, description="Argument proposed by this actor in the current turn, awaiting adjudication.")
    pending_secret_arguments: List[SecretArgument] = Field(default_factory=list, description="Secret arguments made by this actor that are awaiting their trigger conditions.")
    internal_scratchpad: List[str] = Field(default_factory=list, description="AI's internal notes, multi-turn plans, or deliberations for this actor. Not typically shown to other players.")
    big_project_feedback: Optional[Dict[str, str]] = Field(default=None, description="Feedback from big project check containing 'original_action', 'reasoning', 'first_stage_action', and 'remaining_plan' for use in re-deliberation.")
    deliberation_attempts_this_turn: int = Field(default=0, description="Number of times this actor has restarted deliberation this turn to prevent infinite loops.")
    conversation_history: List[Tuple[str, str]] = Field(default_factory=list, description="Conversation history for this actor's deliberation prompts. Each tuple is (role, content) where role is 'human' or 'assistant'.")

    @classmethod
    def from_actor_setup(cls, actor_setup: Actor):
        """Helper method to initialize ActorState from an Actor definition."""
        active_forces = []
        for unit_def in actor_setup.starting_forces:
            active_forces.append(
                ForceUnitState(
                    unit_name=unit_def.unit_name,
                    location=unit_def.starting_location, # Use starting_location here
                    details=unit_def.details,
                )
            )
        return cls(actor_name=actor_setup.actor_name, current_forces=active_forces)

# --- LOGGING & EVENT MODELS ---

class LogEntry(BaseModel):
    entry_id: str # Consider UUID
    timestamp: str # ISO format timestamp
    turn: int
    phase: GamePhase
    entry_type: LogEntryType
    actor_name: Optional[str] = Field(None, description="Actor associated with this log entry, if applicable.")
    content: Union[ArgumentVariant, str] = Field(description="The actual data of the log entry (e.g., an argument, game event, or a narrative string).")
    summary: Optional[str]


# --- GAME STATE MODEL (for state graph) ---

class GameState(BaseModel):
    """
    Represents the overall state of the game at any point in time.
    """
    game_definition: MatrixGame # Reference to the static game setup
    current_turn: int = Field(default=1, description="The current turn number.")
    current_phase: GamePhase = Field(default=GamePhase.SETUP, description="The current phase of the turn or game.")
    actor_states: List[ActorState] = Field(default_factory=list, description="The dynamic states of all actors in the game.")
    active_player_queue_index: int = Field(default=0, description="Index of the actor IN THE TURN ORDER whose turn it is. Ranges from 0 to len(turn_order)-1.")
    game_log: List[LogEntry] = Field(default_factory=list, description="A chronological record of key arguments, decisions, and outcomes for after-action review.")
    game_state_summary: str = Field(default="", description="A brief narrative summary of the game state, including events that have occured in the game so far and their reprecussions.")
    global_narrative_markers: List[str] = Field(default_factory=list, description="Overall game state descriptors or ongoing world events not tied to a single actor, e.g., 'International sanctions regime in effect', 'Widespread humanitarian crisis'.")
    turn_order: List[int] = Field(default_factory=list, description="List of actor *indices from game_definition.actors* defining the current turn order. Can be modified or randomized.")
    triggered_secrets_this_turn: List[str] = Field(default_factory=list, description="List of secret arguments that were triggered during the current turn's adjudication, formatted as 'Actor: Action description'.")

    @classmethod
    def from_matrix_game_setup(cls, game_setup: MatrixGame):
        """Initializes the GameState from a MatrixGame setup."""
        actor_s = [ActorState.from_actor_setup(actor_def) for actor_def in game_setup.actors]
        initial_turn_order = list(range(len(game_setup.actors))) # Stores indices of actors from game_setup.actors
        
        initial_phase = GamePhase.SETUP
    
        return cls(
            game_definition=game_setup,
            actor_states=actor_s,
            turn_order=initial_turn_order,
            current_turn=1,
            current_phase=initial_phase,
            game_state_summary=game_setup.introduction,
        )

    @property
    def current_actor_state(self) -> Optional[ActorState]:
        """Convenience property to get the ActorState of the current actor based on turn_order and active_player_queue_index."""
        if not self.turn_order or not (0 <= self.active_player_queue_index < len(self.turn_order)):
            return None
        
        # turn_order contains the actual indices for actor_states
        actor_actual_index = self.turn_order[self.active_player_queue_index]
        
        if 0 <= actor_actual_index < len(self.actor_states):
            return self.actor_states[actor_actual_index]
        return None

    @property
    def current_actor_definition(self) -> Optional[Actor]:
        """Convenience property to get the Actor definition of the current actor."""
        current_s = self.current_actor_state
        if current_s:
            # Find the corresponding Actor definition from game_definition.actors
            for actor_def in self.game_definition.actors:
                if actor_def.actor_name == current_s.actor_name:
                    return actor_def
        return None
    
# --- STRUCTURED OUTPUT MODELS (for LLMs) ---

class ArgumentResponse(BaseModel): # This is the structured output for LLMs making arguments
    type: Literal["StandardArgument", "SecretArgument"] = Field(description="The type of argument being made.")
    action_description: str = Field(description="The action you want to take and its intended effect.")
    pros: List[str] = Field(description="Reasons explaining why the action is likely to be successfully executed (Pros) - focus on capabilities, resources, authority, favorable conditions, support, etc.")
    scratchpad_notes: str = Field(description="Any notes or thoughts you want to add to your scratchpad. Useful for multi-turn planning, as you will be able to refer to this later.")
    trigger_conditions: str = Field(description="If using a secret argument, conditions under which this secret argument is revealed or activated. Leave blank if not using a secret argument.")

class CriticResponse(BaseModel):
    cons: List[str] = Field(description="Obstacles that could prevent successful execution of the action (Cons) - focus on resource limitations, opposition, technical barriers, unfavorable conditions, etc.")

class AdjudicationMethodResponse(BaseModel):
    method: AdjudicationMethod = Field(description="The method of adjudication you want to use.")

class EstProbabilityResponse(BaseModel):
    reasoning: str = Field(description="A brief explanation of your reasoning for the estimated probability of success for the argument.")
    success_probability: float = Field(description="The estimated probability of success for the argument. This should be a floating point number between 0 and 1.")

class NarrativeResponse(BaseModel):
    adjudication_narrative: str = Field(description="A narrative account describing what happened - what the actor did (or failed to do) and the immediate consequences. If the action failed, explain that the actor tried to do this but failed because of specific reasons from the cons.")

class ForceUpdate(BaseModel):
    actor_name: str = Field(description="Name of the actor who owns this force unit")
    unit_name: str = Field(description="Name of the unit to update")
    location: Optional[str] = Field(None, description="New location for the unit")
    details: Optional[str] = Field(None, description="Updated details about the unit")

class WorldStateUpdateResponse(BaseModel):
    actor_effects: List[str] = Field(default_factory=list, description="New effects to add to the actor's effects list (e.g., 'Successfully negotiated trade deal', 'Lost credibility with allies').")
    force_updates: List[ForceUpdate] = Field(default_factory=list, description="Updates to force units with specific fields that can be updated.")
    global_narrative_markers: List[str] = Field(default_factory=list, description="New global narrative markers to add (e.g., 'Economic sanctions imposed', 'Humanitarian crisis escalating').")
    game_state_summary_update: str = Field(description="Updated summary of the current game state incorporating the results of this argument.")

class CombinedNarrativeAndWorldStateResponse(BaseModel):
    """Combined response that includes both narrative and world state updates for performance optimization"""
    adjudication_narrative: str = Field(description="A narrative account describing what happened - what the actor did (or failed to do) and the immediate consequences. If the action failed, explain that the actor tried to do this but failed because of specific reasons from the cons.")
    actor_effects: List[str] = Field(default_factory=list, description="New effects to add to the actor's effects list (e.g., 'Successfully negotiated trade deal', 'Lost credibility with allies').")
    force_updates: List[ForceUpdate] = Field(default_factory=list, description="Updates to force units with specific fields that can be updated.")
    global_narrative_markers: List[str] = Field(default_factory=list, description="New global narrative markers to add (e.g., 'Economic sanctions imposed', 'Humanitarian crisis escalating').")
    game_state_summary_update: str = Field(description="Updated summary of the current game state incorporating the results of this argument.")

class SecretArgumentValidationResponse(BaseModel):
    is_valid_secret: bool = Field(description="Whether this is truly a secret argument that should remain hidden until triggered.")
    reasoning: str = Field(description="Explanation of why this is or isn't a valid secret argument.")

class BigProjectCheckResponse(BaseModel):
    is_big_project: bool = Field(description="Whether this action constitutes a 'big project' that should be broken down into stages.")
    reasoning: str = Field(description="Explanation of why this is or isn't a big project.")
    first_stage_action: str = Field(description="If this is a big project, describe the first stage action that should be taken instead. Otherwise, leave blank.")
    remaining_plan: str = Field(description="If this is a big project, describe the remaining stages to be saved in the scratchpad. Otherwise, leave blank.")

class SecretArgumentTriggerResponse(BaseModel):
    triggered_arguments: List[str] = Field(default_factory=list, description="List of argument IDs for secret arguments that should be triggered by this outcome.")
    reasoning: str = Field(description="Explanation of why these secret arguments are being triggered.")

class GameOverCheckResponse(BaseModel):
    should_end_game: bool = Field(description="Whether the game should end based on current conditions.")
    reasoning: str = Field(description="Detailed explanation of why the game should or should not end.")
    objectives_achieved: List[str] = Field(default_factory=list, description="List of actor names who have achieved their primary objectives, if any.")
    deadlock_reason: Optional[str] = Field(None, description="If the game is deadlocked, explain why.")

class ObjectiveAssessment(BaseModel):
    actor_name: str = Field(description="Name of the actor being assessed.")
    objectives_achieved: List[str] = Field(default_factory=list, description="List of specific objectives that were achieved.")
    objectives_failed: List[str] = Field(default_factory=list, description="List of specific objectives that were not achieved.")
    overall_performance: str = Field(description="Overall assessment of the actor's performance (e.g., 'Highly Successful', 'Partially Successful', 'Failed').")
    key_accomplishments: List[str] = Field(default_factory=list, description="Major accomplishments or successes.")
    major_setbacks: List[str] = Field(default_factory=list, description="Significant failures or setbacks.")

class EndGameAssessmentResponse(BaseModel):
    game_outcome_summary: str = Field(description="High-level summary of how the game concluded.")
    actor_assessments: List[ObjectiveAssessment] = Field(description="Detailed assessment for each actor.")
    key_turning_points: List[str] = Field(default_factory=list, description="Major events or decisions that significantly influenced the game outcome.")
    strategic_lessons: List[str] = Field(default_factory=list, description="Key strategic insights or lessons from the game.")
    narrative_conclusion: str = Field(description="Engaging narrative summary of the game's conclusion and final state.")
    winners_and_losers: str = Field(description="Assessment of which actors succeeded or failed in achieving their goals.")