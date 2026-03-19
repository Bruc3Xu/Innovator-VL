"""
vLLM-deployed Judge Model with OpenAI-compatible interface
Supports async calls and batch processing
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger("VLLMJudge")


@dataclass
class VLLMJudgeRequest:
    """Request to vLLM-deployed judge model"""
    prompt: str
    completion: str
    answer: Optional[str] = None
    answer_type: Optional[str] = None
    question_type: Optional[str] = None  # e.g., 'math', 'reasoning', 'coding'
    metadata: Optional[Dict] = None


@dataclass
class VLLMJudgeResponse:
    """Response from vLLM judge model"""
    score: float
    explanation: Optional[str] = None
    detailed_scores: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


class AsyncVLLMJudgeClient:
    """Async vLLM judge model client"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get("judge_url", "http://localhost:8000/v1")
        self.model_name = config.get("judge_model", "judge-model")
        self.api_key = config.get("api_key", "dummy-key")  # vLLM may not need a real key
        self.timeout = config.get("timeout", 30.0)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        # Create async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def create_judge_prompt(self, request: VLLMJudgeRequest) -> str:
        """Create system prompt for judge model"""

        # Base system prompt
        base_system = """You are an expert evaluator specializing in {question_type} problems.
Your task is to evaluate the quality of a model's completion based on the provided ground truth answer.

You MUST output a JSON object with the exact following structure:
{
  "score": <float between 0.0 and 1.0>,
  "explanation": "brief explanation of your evaluation",
  "detailed_scores": {
    "format": <0.0-1.0, how well the answer follows expected format>,
    "accuracy": <0.0-1.0, how accurate the answer is>,
    "reasoning": <0.0-1.0, quality of reasoning if applicable>
  },
  "confidence": <0.0-1.0, your confidence in this evaluation>
}

Evaluation criteria:
- Score 1.0: Perfect answer, fully correct and well-formatted
- Score 0.5-0.9: Partially correct or minor formatting issues
- Score 0.0-0.4: Incorrect answer or major errors"""

        # Adjust system prompt based on question type
        if request.question_type == "math":
            system = base_system.format(question_type="mathematical") + """

Additional math evaluation rules:
1. Consider both final answer AND intermediate steps/calculations
2. Mathematical expressions can be equivalent in different forms (e.g., 1/2 = 0.5)
3. Check if reasoning steps logically lead to the conclusion
4. For computational problems, verify the calculation process"""

        elif request.question_type == "reasoning":
            system = base_system.format(question_type="logical reasoning") + """

Additional reasoning evaluation rules:
1. Evaluate the clarity and logic of the reasoning process
2. Check if conclusions logically follow from premises
3. Consider evidence and justifications provided
4. Assess the coherence and consistency of arguments"""

        elif request.question_type == "coding":
            system = base_system.format(question_type="programming") + """

Additional coding evaluation rules:
1. Check if the code solves the specified problem correctly
2. Evaluate code quality, readability, and efficiency
3. Proper syntax and error handling
4. Correct use of programming concepts"""

        else:
            system = base_system.format(question_type="general")

        return system

    def create_user_prompt(self, request: VLLMJudgeRequest) -> str:
        """Create user prompt for judge model"""

        prompt_parts = []

        # Question description
        prompt_parts.append(f"Question: {request.prompt}")

        # Ground truth answer (if available)
        if request.answer:
            prompt_parts.append(f"Ground Truth Answer: {request.answer}")

        # Answer type
        if request.answer_type:
            prompt_parts.append(f"Expected Answer Type: {request.answer_type}")

        # Model completion
        prompt_parts.append(f"Model Completion:\n{request.completion}")

        # Evaluation requirements
        prompt_parts.append("\nEvaluate this completion based on the criteria provided in the system message.")

        return "\n\n".join(prompt_parts)

    async def judge_single(self, request: VLLMJudgeRequest) -> VLLMJudgeResponse:
        """Evaluate single sample"""

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self.create_judge_prompt(request)
            },
            {
                "role": "user",
                "content": self.create_user_prompt(request)
            }
        ]

        # Add retry mechanism
        for attempt in range(self.max_retries):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=500,   # Sufficient space for JSON output
                    response_format={"type": "json_object"}  # Force JSON format
                )

                # Parse response
                content = completion.choices[0].message.content
                response_dict = json.loads(content)  # Parse JSON safely

                return VLLMJudgeResponse(
                    score=float(response_dict["score"]),
                    explanation=response_dict.get("explanation", ""),
                    detailed_scores=response_dict.get("detailed_scores"),
                    confidence=float(response_dict.get("confidence", 0.8))
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for judge request")
                    # Return default values
                    return VLLMJudgeResponse(
                        score=0.0,
                        explanation=f"Judge model failed after {self.max_retries} attempts",
                        detailed_scores={"format": 0.0, "accuracy": 0.0, "reasoning": 0.0},
                        confidence=0.0
                    )

    async def judge_batch(self, requests: List[VLLMJudgeRequest]) -> List[VLLMJudgeResponse]:
        """Evaluate batch"""
        # Process all requests concurrently
        tasks = [self.judge_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                final_results.append(VLLMJudgeResponse(
                    score=0.0,
                    explanation=f"Error in judge: {str(result)}",
                    detailed_scores={"format": 0.0, "accuracy": 0.0, "reasoning": 0.0},
                    confidence=0.0
                ))
            else:
                final_results.append(result)

        return final_results


# Adapter for integration with existing systems
class VLLMJudgeModelAdapter:
    """Adapter that integrates vLLM Judge into the existing scoring system"""

    def __init__(self, config: Dict):
        self.config = config
        self.vllm_client = AsyncVLLMJudgeClient(config)

        # Configure scoring weights
        self.thinking_weight = config.get("thinking_weight", 0.3)
        self.answer_weight = config.get("answer_weight", 0.6)
        self.format_weight = config.get("format_weight", 0.1)

    async def evaluate(self, prompt: str, completion: str, answer: str,
                      prompt_type: str = "normal", answer_type: str = "ANY") -> Dict[str, float]:
        """Evaluate a single response"""

        # Determine question type
        question_type = self._determine_question_type(prompt, answer_type)

        # Create request
        request = VLLMJudgeRequest(
            prompt=prompt,
            completion=completion,
            answer=answer,
            answer_type=answer_type,
            question_type=question_type,
            metadata={"prompt_type": prompt_type}
        )

        # Call vLLM judge
        response = await self.vllm_client.judge_single(request)

        # Transform result format
        return {
            "score": response.score,
            "thinking_score": response.detailed_scores.get("reasoning", 0.0) if response.detailed_scores else 0.0,
            "answer_score": response.detailed_scores.get("accuracy", response.score) if response.detailed_scores else response.score,
            "format_score": response.detailed_scores.get("format", 0.0) if response.detailed_scores else 0.0,
            "explanation": response.explanation,
            "confidence": response.confidence,
            "source": "vllm_judge",
            "validation_method": "vllm_model",
            "prompt_type": prompt_type
        }

    def _determine_question_type(self, prompt: str, answer_type: str) -> str:
        """Determine question type based on prompt and answer type"""
        if answer_type in ["NUMBER", "MATH_EXPRESSIONS"]:
            return "math"
        elif answer_type in ["HTML_CODE", "SVG_CODE", "GENERAL_CODE"]:
            return "coding"
        elif "reason" in prompt.lower() or "explain" in prompt.lower():
            return "reasoning"
        else:
            return "general"


# Usage example
async def main():
    """Usage example"""

    # Configuration
    config = {
        "judge_url": "http://localhost:8000/v1",  # Your vLLM service address
        "judge_model": "your-judge-model-name",  # Your judge model name
        "api_key": "dummy-key",  # Can be dummy key in vLLM compatible mode
        "thinking_weight": 0.3,
        "answer_weight": 0.6,
        "format_weight": 0.1,
        "timeout": 30.0,
        "max_retries": 3
    }

    # Create adapter
    adapter = VLLMJudgeModelAdapter(config)

    # Test cases
    test_inputs = [
        {
            "prompt": "Calculate: 156 + 234 = ?",
            "completion": """
<think>
Let me calculate 156 + 234:
156
+234
----
390
</think>
<answer>390</answer>
            """,
            "answer": "390",
            "prompt_type": "thinking",
            "answer_type": "NUMBER"
        },
        {
            "prompt": "Rectangle area calculation: length=10cm, width=5cm",
            "completion": "<answer>50 square cm</answer>",
            "answer": "50",
            "prompt_type": "normal",
            "answer_type": "NUMBER"
        }
    ]

    # Execute evaluation
    for i, test_input in enumerate(test_inputs):
        result = await adapter.evaluate(**test_input)

        print(f"\nTest {i+1} ({test_input['prompt_type']}):")
        print(f"Score: {result['score']:.3f}")
        print(f"Thinking: {result['thinking_score']:.3f}")
        print(f"Answer: {result['answer_score']:.3f}")
        print(f"Format: {result['format_score']:.3f}")
        print(f"Explanation: {result['explanation']}")
        print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())