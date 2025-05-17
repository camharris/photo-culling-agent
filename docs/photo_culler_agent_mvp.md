**Product Requirements Document (PRD)**

**Project Title:** Local Photo Grading MVP with LangGraph

**Objective:** Build a local-first MVP that uses GPT-4o (via LangGraph) to analyze landscape photographs from a specified folder, grade them based on visual quality and metadata, and sort them into "keep" and "toss" categories. A lightweight Gradio interface will be used to visually review and finalize photo decisions, replacing direct file system interaction.

---

**User Stories:**

1. As a photographer, I want to upload a batch of unedited JPEGs and have them automatically graded and sorted.
2. As a power user, I want to see scoring breakdowns for composition, exposure, subject, and layering.
3. As a developer, I want to override or update verdicts and scores in the metadata from a browser interface.
4. As a future builder, I want the system to support training on my feedback later.
5. As a user, I want to approve or override image decisions via a web interface before photos are finalized.

---

**Key Features:**

- **Image Upload via Gradio Interface**

  - Gradio web app allows drag-and-drop or multi-photo upload.
  - Only accepts `.jpg` or `.jpeg` files.

- **Grading Pipeline (via LangGraph):**

  - Load image using Vision-capable GPT-4o.
  - Ask GPT-4o to:
    - Assign a score (0-100).
    - Classify `verdict`: `keep` or `toss`.
    - Provide scores for: `composition`, `exposure`, `subject`, `layering`.
    - Optionally detect and include `location`.
    - Respect `post_processed` flag (default: false).
    - Generate human-readable tags + notes.

- **Human-in-the-Loop Review (Gradio UI):**

  - Displays images after initial grading for visual review.
  - Shows GPT-4o metadata and verdicts.
  - Allows user to:
    - Confirm or override the verdict
    - Submit feedback or tag favorites
  - Feedback is passed back into the LangGraph workflow.

- **Iterative Learning (MVP Approach via Prompt Contextualization):**
  - User feedback (agreement/disagreement, comments, overrides) from a processed batch is collected.
  - This feedback is summarized and dynamically added to the GPT-4o system prompt for analyzing subsequent batches within the same session.
  - The Gradio UI will include controls to clear the current batch and apply these session-based learnings before starting a new batch.

- **In-Memory or Logical Categorization**

  - During MVP, actual file movement is skipped.
  - Photos are grouped logically ("keep" vs. "toss") and displayed separately within the Gradio app.
  - Graded metadata is made available as downloadable `.json` files per image or in batch.

- **JSON Metadata Schema:**

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

---

**Tech Stack:**

- **LangGraph** for orchestration
- **Python** for core pipeline logic
- **OpenAI GPT-4o** for photo vision and scoring logic
- **Gradio** for human-in-the-loop review UI
- **Pillow** or similar lib for lightweight image inspection (optional)

---

**Out of Scope for MVP:**

- RAW image handling
- Advanced or persistent training from feedback (beyond session-based prompt contextualization)
- File system manipulation
- Real-time collaboration/multi-user
- Photo deduplication or clustering

---

**Deliverables:**

1. Gradio web app for image upload and HITL feedback
2. LangGraph pipeline for scoring and decisioning
3. JSON grading reports for `keep` images
4. Logic to organize photos in memory by verdict
5. Modular design for future feedback learning

---

**Next Steps:**

- Scaffold LangGraph graph structure
- Write grading node prompt for GPT-4o
- Build Gradio interface with image upload and metadata display
- Integrate LangGraph pipeline with Gradio feedback loop
- Export JSON grading data per photo
- Test full loop on local photo batch

---

**Future Considerations & Potential Enhancements:**

- **Persistent Learning:** Implement a mechanism (e.g., SQLite database or configuration files) to store aggregated feedback or learned model adjustments (like prompt modifications or weight changes) so that learnings persist across application sessions.
- **Advanced Example-Based Feedback:** Explore providing the model with more direct examples of (image + AI analysis + user feedback) to improve learning, potentially by selecting diverse and impactful feedback instances to include in prompts, mindful of context window limitations.
- **Refined UI for Overrides:** Enhance the UI to more clearly display when a user's verdict has overridden the AI's suggestion.
- **Detailed Logging and Error Handling:** Implement more robust logging throughout the application and improve error handling for edge cases.
- **Comparative Analysis Implementation:** Fully implement the comparative analysis node in the LangGraph pipeline to rank similar images and apply logic like "keep best N".

---
