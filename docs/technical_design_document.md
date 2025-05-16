# Photo Culling Agent - Technical Design Document

## Project Overview
The Photo Culling Agent is an MVP that uses GPT-4o via LangGraph to analyze landscape photographs, grade them based on quality and metadata, and sort them into "keep" and "toss" categories. A Gradio interface provides a human-in-the-loop review process.

## System Architecture

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

## Core Components

### 1. ImageProcessor
- Handles loading, validation, and preparation of images
- Extracts basic EXIF data if available
- Implements image format validation (.jpg/.jpeg only)

### 2. GPTAnalyzer
- Manages communication with OpenAI's GPT-4o API
- Formats prompts for consistent analysis
- Parses and validates API responses

### 3. LangGraphPipeline
- Defines the workflow graph with nodes for:
  - Image loading and preparation
  - GPT-4o analysis
  - Decision processing
  - Metadata generation
  - Human feedback integration
- Handles state management throughout the workflow

### 4. GradioInterface
- Provides image upload functionality
- Displays analysis results and verdicts
- Collects user feedback and verdict overrides
- Shows images grouped by verdict category

### 5. MetadataManager
- Implements the JSON metadata schema
- Manages in-memory categorization of images
- Provides export functionality for metadata

## Data Structures

### Image Metadata Schema
```json
{
  "filename": "IMG_1234.jpg",
  "verdict": "keep",
  "score": 87.4,
  "rating": "4.5 stars",
  "post_processed": false,
  "tags": ["strong composition", "layered depth", "vibrant sky"],
  "location": "Alabama Hills, California (approximate)",
  "analysis": {
    "composition": 90,
    "exposure": 85,
    "subject": 86,
    "layering": 89,
    "notes": "Natural framing from arch, distant snowy peaks, vibrant contrast"
  },
  "relative_rank": null,
  "user_verdict_override": null,
  "user_feedback": null,
  "learning_signal": null
}
```

## Development Phases

### Phase 1: Core Analysis Pipeline
- Implement `ImageProcessor` for handling images
- Build `GPTAnalyzer` with prompt engineering for consistent results
- Create `MetadataManager` with the defined schema
- Unit test each component

### Phase 2: LangGraph Workflow
- Implement the LangGraph nodes and state management
- Define the workflow edges and transitions
- Create decision logic for keep/toss verdicts
- Test the end-to-end pipeline with sample images

### Phase 3: Gradio Interface
- Build the upload interface with drag-and-drop functionality
- Create image gallery views for review
- Implement feedback collection UI
- Connect the interface to the LangGraph pipeline

### Phase 4: Integration and Testing
- Connect all components into a cohesive system
- Add error handling and edge cases
- Implement basic logging
- Test the full user flow with different image types

## Technical Decisions

### Libraries
- **LangGraph**: For workflow orchestration
- **OpenAI API**: For GPT-4o vision capabilities
- **Gradio**: For rapid UI development
- **Python**: Core programming language
- **Pillow**: For basic image processing if needed

### Testing Strategy
- Unit tests for core components using pytest
- Integration tests for workflow validation
- Manual UI testing for Gradio interface

### Future Considerations
- Modular design allows for easy extension for training from feedback
- Stateless design facilitates future scaling
- JSON metadata provides flexibility for different storage solutions 