from llama3 import Llama


class LLMAssembler:
    def __init__(self, device='cuda:4', model_type='llama3'):
        """
        Args:
            device: CUDA device (e.g., 'cuda:4')
            model_type: One of ['llama3', 'qwen3', 'llama3-8b', 'qwen3-14b']
        """
        # Map model_type to Llama class model_type parameter
        model_type_mapping = {
            'llama3': 'llama3-8b',
            'qwen3': 'qwen3-14b',
            'llama3-8b': 'llama3-8b',
            'qwen3-14b': 'qwen3-14b'
        }

        if model_type in model_type_mapping:
            llama_model_type = model_type_mapping[model_type]
            self.model = Llama(device=device, model_type=llama_model_type)
        else:
            # Fallback to llama3-8b
            print(f"⚠️  Unknown model_type '{model_type}', defaulting to llama3-8b")
            self.model = Llama(device=device, model_type='llama3-8b')

    
    def assemble(self, prompt: str, temperature: float = 0.3) -> str:
        return self.model.assemble(prompt, temperature=temperature)
    
    def question_answer(self, prompt: str) -> str:
        return self.model.question_answer(prompt)
    
    def name_entity_extraction(self,prompt: str) -> str:
        return self.model.name_entity_extraction(prompt)
        
