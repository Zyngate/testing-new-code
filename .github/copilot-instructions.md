# GitHub Copilot Instructions for Stelle Codebase

## Overview
This document provides guidance for AI coding agents working within the Stelle codebase. Understanding the architecture, workflows, and conventions is crucial for effective contributions.

## Architecture
The Stelle application is structured around a FastAPI framework, with distinct components organized into services and routes:
- **Services**: Handle business logic and data processing (e.g., `ai_service.py`, `recommendation_service.py`, `time_slot_service.py`).
- **Routes**: Define API endpoints and manage request handling (e.g., `goal_routes.py`, `plan_routes.py`, `chat_routes.py`).

### Key Components
- **AI Service**: Interacts with external APIs and manages AI-related tasks. See `services/ai_service.py` for implementation details.
- **Recommendation Engine**: Provides content recommendations based on user data and analytics. Refer to `services/recommendation_service.py`.
- **Time Slot Service**: Determines optimal posting times based on user behavior. Check `services/time_slot_service.py`.

## Developer Workflows
### Running the Application
To start the FastAPI application, use:
```bash
uvicorn main:app --reload
```
Ensure all dependencies are installed via:
```bash
pip install -r requirements.txt
```

### Testing
Run tests using pytest. Ensure to follow the naming conventions for test files (e.g., `test_*.py`).

### Debugging
Utilize logging extensively throughout the codebase. Check `config.py` for logger configurations.

## Project-Specific Conventions
- **Naming**: Use snake_case for variable and function names. Class names should be in CamelCase.
- **Error Handling**: Implement try-except blocks to manage exceptions gracefully, especially in asynchronous functions.
- **Documentation**: Use docstrings to document functions and classes clearly.

## Integration Points
- **External APIs**: The application integrates with various external services (e.g., Groq API). API keys are managed via environment variables.
- **Database**: MongoDB is used for data storage. Refer to `database.py` for connection details and collections.

## Communication Patterns
- **WebSocket**: For real-time interactions, such as goal setting, use the WebSocket endpoints defined in `goal_routes.py`.
- **REST API**: Standard HTTP methods (GET, POST) are used for CRUD operations across various resources.

## Conclusion
This document serves as a foundational guide for AI agents to navigate the Stelle codebase effectively. For further assistance, refer to specific service files and route definitions as needed.