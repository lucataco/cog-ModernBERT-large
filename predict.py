# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_ID = "answerdotai/ModernBERT-large"
MODEL_CACHE = "checkpoints"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download and cache the model during setup
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)
        self.model = AutoModelForMaskedLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Enter a sentence with a [MASK] token, and the model will predict the missing word",
            default="The capital of France is [MASK]."
        ),
    ) -> str:
        """Run a single prediction on the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move inputs to the same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        masked_index = inputs["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
        predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)

        return predicted_token
