# Matrix Wargame CLI

A command-line interface for running AI-powered matrix wargame simulations. MatrixAI uses structured simulations that use systematic frameworks to explore how different actors might behave in complex scenarios, providing insights into potential outcomes and unintended consequences of various actions and decisions.

**🌐 Want a graphical interface?** Check out the full web version at [MatrixAI Platform](https://use-matrixai.com) for an enhanced experience with visual interfaces and additional features.

## What are Matrix Wargames?

Matrix wargames are structured simulations that use systematic frameworks to explore how different actors might behave in complex scenarios. Unlike traditional board games or computer simulations, matrix games focus on understanding the decision-making processes, motivations, and interactions between multiple stakeholders.

These games are particularly valuable for strategic planning, policy analysis, training, and research, providing insights into potential outcomes and unintended consequences of various actions and decisions. Each simulation represents realistic actors with their own objectives, constraints, and information, making decisions that create cascading effects throughout the scenario.

## Features

- **🎯 Multiple Scenarios**: Pre-built scenarios covering diplomatic crises, military conflicts, economic disputes, and more
- **🤖 AI-Powered Simulation**: Automated actors with distinct personalities, objectives, and decision-making capabilities  
- **📊 Structured Argumentation**: Actions are evaluated through pros/cons analysis and probability estimation
- **⚡ Real-time Streaming**: Watch the simulation unfold step-by-step with live updates
- **🎮 Easy CLI Interface**: Simple commands to list and run any scenario
- **🧠 Advanced AI Reasoning**: Each actor uses sophisticated reasoning to make realistic decisions based on their role and objectives
- **🔄 Dynamic Interactions**: Complex multi-actor interactions where decisions create lasting effects on the scenario

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd matrix-wargame-cli
```

2. Create venv and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Usage

**List available scenarios:**
```bash
python run_scenario.py list
```

**Run a specific scenario:**
```bash
python run_scenario.py diplomatic-crisis
```

## Available Scenarios

- **diplomatic-crisis**: A tense 3-nation diplomatic scenario (2 turns)
- **climate-summit**: Emergency climate negotiations (4 turns)  
- **trade-dispute**: Global trade war escalation (5 turns)
- **race-to-agi**: Competition to achieve artificial general intelligence (6 turns)
- **attack-on-taiwan**: Military crisis in the Taiwan Strait (8 turns)
- **corporate-merger**: $127B hostile takeover battle (8 turns)
- **cyber-attack**: Critical infrastructure attack response (10 turns)
- **supply-chain-crisis**: Global supply chain collapse (12 turns)

## How It Works

### Matrix Game Concepts

**Arguments**: Each turn, AI actors propose actions with supporting reasons (pros) explaining why they should succeed.

**Adjudication**: A critic AI identifies potential obstacles (cons), then the action is either auto-approved or evaluated probabilistically.

**Narrative Evolution**: Successful and failed actions create lasting effects that shape the ongoing story and strategic situation.

**Turn Structure**: Each turn represents a realistic timeframe (days to weeks) where actors take meaningful actions.

**Multi-Actor Dynamics**: Each AI actor has distinct objectives, resources, and decision-making processes, creating realistic interactions and conflicts.

### Example Output

```
⏱️  Turn 1
   🎮 Nation Alpha's turn
   ❌ Launch an international diplomatic campaign to discredit satellite evidence
   🎮 Nation Beta's turn  
   ✅ Organize high-level coalition summit to maintain pressure and unity
   🎮 UN Mediator's turn
   ✅ Propose joint dialogue session focused on de-escalation measures

🏁 Simulation completed!

📊 Final Results:
   Duration: 2 turns
   Phase: Game Ended

📜 Final Situation:
   The UN Mediator achieved a significant breakthrough by facilitating 
   a trilateral agreement on joint border monitoring...
```

## Architecture

- **`src/matrix_ai/`**: Core simulation engine with LangGraph workflows
- **`scenarios/`**: JSON scenario definitions with actors, objectives, and setup
- **`run_scenario.py`**: Main CLI interface for running simulations
- **`pyproject.toml`**: Python package configuration and dependencies

## Creating Custom Scenarios

Scenarios are defined in JSON format with the following structure:

```json
{
  "name": "Your Scenario Name",
  "category": "Political/Military/Economic/Environmental", 
  "description": "Brief description of the scenario",
  "introduction": "Setting the stage...",
  "background_briefing": "Detailed context and background...",
  "actors": [
    {
      "actor_name": "Actor Name",
      "actor_briefing": "Detailed role description...",
      "objectives": ["Objective 1", "Objective 2"],
      "starting_forces": [
        {
          "unit_name": "Unit Name",
          "starting_location": "Location",
          "details": "Additional details"
        }
      ]
    }
  ],
  "victory_conditions": "How the game ends...",
  "turn_length": "2 weeks",
  "game_length": 6
}
```

## Technology Stack

- **LangGraph**: Workflow orchestration for multi-agent simulations
- **LangChain**: LLM integration and prompt management  
- **OpenAI GPT-4**: AI reasoning for actors, critics, and adjudication
- **Pydantic**: Data validation and structured outputs
- **Python 3.11+**: Core runtime environment

## Web Platform

Currently in preview mode. Will be ready soon!

For a more comprehensive experience with visual interfaces, scenario management, and additional features, check out the full MatrixAI web platform at [MatrixAI Platform](https://use-matrixai.com). The web version will include:

- AI-assisted scenario builder and editor  
- Advanced analytics and reporting
- Real-time collaboration features

## License

This project is licensed under the **AGPL v3.0 (Affero General Public License)**.

### What this means:

- **✅ You can freely use, modify, and distribute this code**
- **✅ You can use it for commercial purposes**
- **⚠️ If you modify and distribute or deploy the code, you must share your changes under the same AGPL v3.0 license**
- **⚠️ If you run this software on a server or provide it as a web service, you must make the complete source code (including any modifications) available to anyone who interacts with the service**
- **❌ You cannot combine this code with proprietary code without releasing your entire codebase under AGPL v3.0**

### Why AGPL v3.0?

The AGPL v3.0 license ensures that this open-source project remains open-source. Unlike permissive licenses (like MIT), the AGPL prevents companies from taking this code, making improvements, and keeping those improvements proprietary. This "copyleft" approach means:

1. **Network services must share source**: Even if you run this code as a web service without distributing it, you must still provide the source code to users
2. **Derivative works stay open**: Any software that incorporates this code must also be open-source
3. **Community benefits**: All improvements and modifications flow back to the community

This protects the project from proprietary appropriation while still allowing full commercial and personal use by those willing to keep their code open.

**Full license text**: See the LICENSE file for complete terms and conditions.