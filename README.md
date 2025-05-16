# Photo Culling Agent

An intelligent photo culling application that uses GPT-4o vision capabilities via LangGraph to analyze landscape photographs, grade them based on quality and metadata, and sort them into "keep" and "toss" categories.

## Project Overview

The Photo Culling Agent is an MVP that uses GPT-4o via LangGraph to analyze landscape photographs, grade them based on quality and metadata, and sort them into "keep" and "toss" categories. A Gradio interface provides a human-in-the-loop review process.

## Features

- Analyze landscape photographs using GPT-4o vision capabilities
- Grade photos based on composition, exposure, subject, and layering
- Sort photos into "keep" and "toss" categories
- Export detailed metadata in JSON format
- Human-in-the-loop review interface (coming soon)

## Development Status

- ✅ **Phase 1: Core Analysis Pipeline** - Complete
  - ✅ `ImageProcessor` for handling images
  - ✅ `GPTAnalyzer` with prompt engineering 
  - ✅ `MetadataManager` with JSON schema
  - ✅ Unit tests for all components

- 🔄 **Phase 2: LangGraph Workflow** - In Progress
  - ⬜ LangGraph nodes and state management
  - ⬜ Workflow edges and transitions
  - ⬜ Keep/toss decision logic
  - ⬜ End-to-end pipeline testing

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
python main.py
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
│   └── test_metadata_manager.py
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