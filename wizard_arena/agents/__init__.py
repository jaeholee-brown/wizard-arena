# wizard_arena/agents/__init__.py
from .base import WizardAgent
from .random_agent import RandomWizardAgent
from .llm_agents import LLMCallFailed, LLMWizardAgent

__all__ = [
    "WizardAgent",
    "RandomWizardAgent",
    "LLMWizardAgent",
    "LLMCallFailed",
]
