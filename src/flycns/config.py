"""Environment configuration for the NeuPrint client used by the project."""

import os

from dotenv import load_dotenv

# Load environment variables from a local .env file when present.
load_dotenv()

NEUPRINT_SERVER = os.getenv("NEUPRINT_SERVER", "https://neuprint.janelia.org")
NEUPRINT_DATASET = os.getenv("NEUPRINT_DATASET", "male-cns:v1.0")
NEUPRINT_TOKEN = os.getenv("NEUPRINT_TOKEN")

if not NEUPRINT_TOKEN:
    raise ValueError("NEUPRINT_TOKEN is not set. Please set it in the .env file.")