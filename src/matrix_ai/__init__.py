"""Matrix AI Prototype - A multiagent system using LangGraph to simulate matrix wargames."""

__version__ = "0.1.0"

from .schemas import (
    GameState,
    MatrixGame,
    Actor,
    ForceUnit,
    ArgumentVariant,
    StandardArgument,
    SecretArgument,
    GameOverCheckResponse,
    EndGameAssessmentResponse,
    ObjectiveAssessment,
    ForceUpdate,
)

from .adjudication import create_adjudication_graph
from .argumentation import create_argumentation_graph
from .scenario_update import create_scenario_update_graph
from .main_game_graph import create_main_game_graph, run_matrix_game, stream_matrix_game

__all__ = [
    "GameState",
    "MatrixGame", 
    "Actor",
    "ForceUnit",
    "ArgumentVariant",
    "StandardArgument",
    "SecretArgument",
    "GameOverCheckResponse",
    "EndGameAssessmentResponse", 
    "ObjectiveAssessment",
    "ForceUpdate",
    "create_adjudication_graph",
    "create_argumentation_graph",
    "create_scenario_update_graph",
    "create_main_game_graph",
    "run_matrix_game",
    "stream_matrix_game",
] 