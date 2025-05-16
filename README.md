# Photo Culling Agent

An intelligent photo culling application that uses GPT-4o vision capabilities via LangGraph to analyze landscape photographs, grade them based on quality and metadata, and sort them into "keep" and "toss" categories.

## Project Overview

The Photo Culling Agent is an MVP that uses GPT-4o via LangGraph to analyze landscape photographs, grade them based on quality and metadata, and sort them into "keep" and "toss" categories. A Gradio interface provides a human-in-the-loop review process.

## Features

- Analyze landscape photographs using GPT-4o vision capabilities
- Grade photos based on composition, exposure, subject, and layering
- Sort photos into "keep" and "toss" categories with confidence levels
- Provide detailed decision rationale with weighted scoring
- Export detailed metadata in JSON format
- Human-in-the-loop review interface (coming soon)

## Development Status

- ✅ **Phase 1: Core Analysis Pipeline** - Complete
  - ✅ `ImageProcessor` for handling images
  - ✅ `GPTAnalyzer` with prompt engineering 
  - ✅ `MetadataManager` with JSON schema
  - ✅ Unit tests for all components

- 🔄 **Phase 2: LangGraph Workflow** - In Progress
  - ✅ LangGraph nodes and state management
  - ✅ Workflow edges and transitions
  - ✅ Keep/toss decision logic with weighted scoring and confidence levels
  - 🔄 End-to-end pipeline testing

- ⬜ **Phase 3: Gradio Interface** - Planned
  - ⬜ Image upload functionality
  - ⬜ Results and verdicts display
  - ⬜ User feedback collection
  - ⬜ Verdict override capability

- ⬜ **Phase 4: Integration and Testing** - Planned
  - ⬜ Full system integration
  - ⬜ Error handling and edge cases
  - ⬜ Logging implementation
  - ⬜ User flow testing

## Enhanced Decision Logic

The keep/toss decision logic uses a sophisticated system that:

1. **Applies weighted scoring** to different criteria (composition, exposure, subject, layering)
2. **Calculates confidence levels** ranging from DEFINITE_KEEP to DEFINITE_TOSS
3. **Provides detailed rationale** explaining the decision
4. **Identifies borderline cases** that would benefit from human review
5. **Allows customizable criteria weights** for different photography styles

This enhanced approach provides more nuanced decisions than simple binary keep/toss verdicts, making the system more useful for photographers with different preferences.

## Installation

### Prerequisites

- Python 3.11+
- Conda

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/photo-culling-agent.git
   cd photo-culling-agent
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate photo-culling-agent
   ```

3. Create a `.env` file with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

Run the application:
```bash
# Process a single image
python main.py --image path/to/image.jpg --output path/to/output/dir

# Process a directory of images
python main.py --dir path/to/images/dir --output path/to/output/dir

# Use custom weights for decision criteria
python main.py --image path/to/image.jpg --output path/to/output/dir --weights "composition=2.0,exposure=0.8,subject=1.0,layering=0.7,base_score=1.0"
```

## Project Structure

```
photo-culling-agent/
├── docs/                        # Documentation
│   ├── technical_design_document.md
│   └── photo_culler_agent_mvp.md
├── src/                         # Source code
│   └── photo_culling_agent/     # Main package
│       ├── image_processor/     # Image handling and preparation
│       ├── gpt_analyzer/        # GPT-4o API communication
│       ├── metadata_manager/    # Metadata and categorization
│       └── langgraph_pipeline/  # LangGraph workflow (Phase 2)
├── tests/                       # Unit tests
│   ├── test_image_processor.py
│   ├── test_gpt_analyzer.py
│   ├── test_metadata_manager.py
│   └── test_langgraph_pipeline.py
├── environment.yml              # Conda environment specification
├── main.py                      # Application entry point
└── README.md                    # This file
```

## Architecture

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│  Gradio UI      │<───>│  LangGraph    │<───>│  GPT-4o API    │
│  - Upload       │     │  Pipeline     │     │  (Vision)      │
│  - Review       │     │               │     │                │
└─────────────────┘     └───────────────┘     └────────────────┘
         ↑                      ↑                     ↑
         │                      │                     │
         v                      v                     v
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│  Metadata       │     │  Image        │     │  Feedback      │
│  Management     │     │  Processing   │     │  Collection    │
└─────────────────┘     └───────────────┘     └────────────────┘
```

## License

MIT License 