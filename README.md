# Catalog Research Agent - Software Lifecycle Research Agent

An automated research agent built with LangGraph and LangChain that discovers software component lifecycle information including release dates and end-of-support dates.

## Features

- **Automated Research**: Uses Perplexity AI to search for software lifecycle information
- **Multi-stage Verification**: Implements confidence-based verification with iterative refinement
- **Source Attribution**: Tracks and validates information sources
- **Batch Processing**: Supports processing multiple components simultaneously
- **Configurable Prompts**: Centralized prompt management through YAML configuration

## Architecture

### Core Components

- **Nodes**: Research workflow stages (research, verification, followup_research, output_generation)
- **Tools**: Perplexity AI integration for initial and deep search capabilities
- **Models**: Pydantic models for type-safe data structures
- **Graph**: LangGraph workflow orchestration
- **Config**: Environment and LLM configuration management

<img width="1282" height="1378" alt="image" src="https://github.com/user-attachments/assets/4f232805-ed57-4b0c-94b4-a9a46d1b47a6" />


### Workflow

1. **Initial Research**: Generate search query and perform initial research
2. **Verification**: Analyze results, extract data, compute confidence scores
3. **Decision**: Determine next step based on confidence and iteration count
4. **Follow-up Research** (if needed): Perform deeper search with refined queries
5. **Output Generation**: Create final structured results

## Installation

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Azure OpenAI API access
- Perplexity API access

### Setup

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:
   ```bash
   cd catalog_research_agent
   poetry install
   ```

3. **Configure Environment**:
   
   **Option A - .env file (Recommended)**:
   ```bash
   # Create .env file in catalog_research_agent directory
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   ```
   
   **Option B - Environment Variables**:
   ```bash
   export AZURE_OPENAI_API_KEY="your_azure_openai_api_key_here"
   export PERPLEXITY_API_KEY="your_perplexity_api_key_here"
   ```

## Usage

### Batch Processing (Primary Usage)

1. **Create component list file**:
   ```json
   [
     "Microsoft .NET 7.0.x",
     "Oracle Java 17",
     "Ubuntu 22.04 LTS"
   ]
   ```

2. **Run batch processing**:
   ```bash
   cd catalog_research_agent
   poetry run python main.py components.json [max_concurrent]
   ```

   **Examples**:
   ```bash
   # Process components with default concurrency (3)
   poetry run python main.py components.txt
   
   # Process with custom concurrency
   poetry run python main.py components.json 5
   ```

### Example Output

```json
{
  "component": "Microsoft .NET 7.0.x",
  "description": ".NET 7.0 is a cross-platform framework...",
  "active_date": "2022-11-08",
  "eos_date": "2024-05-14",
  "confidence_score": 95.0,
  "sources": {
    "description_sources": [...],
    "active_date_sources": [...],
    "eos_date_sources": [...]
  },
  "verification_notes": "All dates verified through official Microsoft documentation",
  "iteration_count": 1
}
```

## Configuration

### Prompt Management

Prompts are centrally managed in `prompts.yaml`:

- `research.query_generation`: Initial search query generation
- `verification.analysis`: Result verification and extraction
- `tools.initial_search_system`: Initial search system prompt
- `tools.deep_search_system`: Deep search system prompt

### Lifecycle Definitions

The agent searches for specific lifecycle phases:

- **Active Date**: First day of general availability for commercial use
- **End of Support Date**: When vendor stops providing standard support

## Development

### Project Structure

```
catalog_research_agent/
├── main.py              # Primary entry point (batch processing)
├── graph.py             # LangGraph workflow definition
├── nodes.py             # Workflow node implementations
├── tools.py             # Perplexity AI search tools
├── models.py            # Pydantic data models
├── config.py            # Configuration management
├── prompt_loader.py     # YAML prompt loading utility
├── prompts.yaml         # Centralized prompt templates
├── pyproject.toml       # Poetry configuration & dependencies
└── poetry.lock          # Locked dependency versions
```

### Extending the Agent

1. **Add New Search Tools**: Implement tools in `tools.py`
2. **Modify Workflow**: Update nodes in `nodes.py`
3. **Change Prompts**: Edit `prompts.yaml`
4. **Add Data Fields**: Extend models in `models.py`

### Development Commands

```bash
# Install dependencies
poetry install

# Add new dependency
poetry add package-name

# Run in development mode
poetry run python main.py components.txt

# Run tests (if you add them)
poetry run pytest

# Update dependencies
poetry update
```

## API Keys

- **Azure OpenAI**: Required for LLM operations and structured output parsing
- **Perplexity AI**: Required for web search and research capabilities

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are properly set
2. **Rate Limiting**: Adjust `max_concurrent` parameter for batch processing
3. **Timeout Issues**: Check network connectivity and API service status

### Logging

Set logging level in `config.py`:
```python
logging.basicConfig(level=logging.INFO)  # Change to INFO, WARNING, or ERROR
```
