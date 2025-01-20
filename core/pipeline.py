# core/pipeline.py
import transformers
from langchain_huggingface import HuggingFacePipeline
from utils.stopping_criteria import StopOnString, StoppingCriteriaList
from utils.logger import logger
import torch

class PipelineBuilder:
    def __init__(self, model, tokenizer, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings

    def build(self):
        try:
            logger.info("Creating HuggingFace pipeline...")
            
            generation_pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                max_new_tokens=self.settings.MAX_NEW_TOKENS,
                do_sample=True,
                top_p=self.settings.TOP_P,
                temperature=self.settings.TEMPERATURE,
                repetition_penalty=self.settings.REPETITION_PENALTY,
                no_repeat_ngram_size=self.settings.NO_REPEAT_NGRAM_SIZE
            )

            stop_criteria = StoppingCriteriaList([
                StopOnString("User:", self.tokenizer)
            ])

            llm = HuggingFacePipeline(
                pipeline=generation_pipeline,
                pipeline_kwargs={"stopping_criteria": stop_criteria}
            )

            logger.info("Pipeline created successfully")
            return llm

        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise e
