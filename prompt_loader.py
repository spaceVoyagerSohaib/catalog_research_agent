import yaml
from pathlib import Path
from typing import Dict, Any

class PromptLoader:
    _instance = None
    _prompts = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._prompts is None:
            self._load_prompts()
    
    def _load_prompts(self):
        prompts_file = Path(__file__).parent / "prompts.yaml"
        with open(prompts_file, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)
    
    def get_prompt(self, category: str, prompt_name: str, **kwargs) -> str:
        prompt_template = self._prompts[category][prompt_name]
        return prompt_template.format(**kwargs)
    
    def get_all_prompts(self) -> Dict[str, Any]:
        return self._prompts

prompt_loader = PromptLoader() 