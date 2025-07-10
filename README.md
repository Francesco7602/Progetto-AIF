# AIF-Project

**Artificial Intelligence Fundamentals – University of Pisa**

## Overview

This project explores intelligent agent design for the MiniHack environment, a research platform based on NetHack.
## Project Structure

- `AgentMinihack.py`: Main agent logic and MiniHack interface
- `kb.pl`: Prolog knowledge base for the FOL agent
- `main.py`: Entry point and experiment runner
- `level/`: Custom levels for testing
- `memory`, `utility.py`: Support modules
- `requirements.txt`: Python dependencies

## Setup

1. **Create a virtual environment** (Python 3.9+ recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the agent:**

   ```bash
   python main.py
   ```

## Notes

- SWI-Prolog is required for the logic-based agent.
- Custom levels are in the `level/` directory.
- Main dependencies: MiniHack, NLE, numpy, pandas, pyswip.

## Contact

For questions or suggestions:
- Francesco Romeo - franciromeo76@gmail.com
- Raffaele Cadau - r.cadau@studenti.unipi.it
