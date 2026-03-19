"""
Judge Factory - Unified entry for creating different types of Judge models
Supports: local rules, enhanced Judge, vLLM-deployed Judge
"""

import os
from typing import Dict, Optional
from .reward_system import RewardSystem
from .enhanced_judge_adapter import create_enhanced_judge_reward
from .vllm_judge_adapter import VLLMJudgeModelAdapter


def create_judge(config: Dict, judge_type: str = "auto") -> object:
    """
    Unified factory function for creating Judge models

    Args:
        config: Configuration dictionary
        judge_type:
            - "auto": Auto-detect
            - "rule": Rule-based reward
            - "enhanced": Enhanced local Judge
            - "vllm": vLLM-deployed Judge Model

    Returns:
        Judge model instance
    """
    if judge_type == "auto":
        # Auto-detect best configuration
        if config.get("judge", {}).get("judge_url"):
            judge_type = "vllm"
        elif config.get("reward", {}).get("type") == "enhanced_judge":
            judge_type = "enhanced"
        else:
            judge_type = "rule"

    if judge_type == "vllm":
        # vLLM-deployed Judge Model
        vllm_config = {
            "judge_url": config.get("judge", {}).get("judge_url", "http://localhost:8000/v1"),
            "judge_model": config.get("judge", {}).get("model_name", "judge-model"),
            "api_key": config.get("judge", {}).get("api_key", os.getenv("JUDGE_API_KEY", "dummy")),
            "timeout": config.get("judge", {}).get("timeout", 30.0),
            "max_retries": config.get("judge", {}).get("max_retries", 3),
            "thinking_weight": 0.3,
            "answer_weight": 0.6,
            "format_weight": 0.1,
        }
        # This is for compatibility only; full integration should be used in practice
        return VLLMJudgeModelAdapter(vllm_config)

    elif judge_type == "enhanced":
        # Enhanced local Judge
        enhanced_config = {
            "async_pool_size": config.get("reward", {}).get("config", {}).get("async_pool_size", 4),
            "timeout": config.get("reward", {}).get("config", {}).get("timeout", 30.0),
            "scoring_weights": config.get("reward", {}).get("config", {}).get("scoring_weights", {
                "thinking_prompt": {"format": 0.10, "thinking": 0.30, "answer": 0.60},
                "normal_prompt": {"format": 0.05, "thinking": 0.15, "answer": 0.80}
            }),
            "validation_layers": config.get("reward", {}).get("config", {}).get("validation_layers", [
                "exact_match", "math_verify", "choice_normalize"
            ])
        }
        return create_enhanced_judge_reward(
            config=enhanced_config,
            use_enhanced=True
        )

    elif judge_type == "rule":
        # Original rule-based reward system
        return RewardSystem()

    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


# Configuration examples
JUDGE_CONFIG_EXAMPLES = {
    # 1. vLLM-deployed Judge Model
    "vllm": {
        "type": "vllm",
        "config": {
            "judge_url": "http://your-vllm-server:8000/v1",
            "model_name": "your-judge-model",  # Your judge model name
            "api_key": "dummy-key",  # vLLM compatible mode
            "timeout": 30.0,
            "max_retries": 3,
            "system_prompt_version": "v1.0"
        }
    },

    # 2. Enhanced local Judge
    "enhanced": {
        "type": "enhanced",
        "config": {
            "async_pool_size": 4,
            "scoring_weights": {
                "thinking_prompt": {"format": 0.10, "thinking": 0.30, "answer": 0.60},
                "normal_prompt": {"format": 0.05, "thinking": 0.15, "answer": 0.80}
            },
            "validation_layers": [
                "exact_match",
                "math_verify",
                "choice_normalize"
            ],
            "thinking_evaluation": {
                "min_length": 15,
                "logic_threshold": 0.3,
                "relevance_threshold": 0.25
            }
        }
    },

    # 3. Traditional rule-based
    "rule": {
        "type": "rule",
        "config": {}  # Use default configuration
    }
}


# Integration into training configuration
TRAINING_CONFIG_TEMPLATE = """
# training_config.yaml

# Judge model configuration
judge:
  type: vllm  # or "enhanced", "rule"
  config:
    # vLLM configuration
    judge_url: ${JUDGE_MODEL_URL:http://localhost:8000/v1}
    model_name: ${JUDGE_MODEL_NAME:judge-model}
    api_key: ${JUDGE_API_KEY:dummy-key}
    timeout: 30.0
    max_retries: 3
    # Scoring weights
    thinking_weight: 0.30
    answer_weight: 0.60
    format_weight: 0.10

# Training configuration
reward:
  type: ${JUDGE_TYPE:enhanced_judge}  # Backward compatible
  config:
    # Original configuration...

# Environment variable settings:
# export JUDGE_MODEL_URL=http://your-vllm-server:8000/v1
# export JUDGE_MODEL_NAME=your-judge-model
# export JUDGE_TYPE=vllm
"""


if __name__ == "__main__":
    # Test factory function
    import asyncio

    # 1. Test vLLM mode
    vllm_config = JUDGE_CONFIG_EXAMPLES["vllm"]["config"]
    vllm_judge = create_judge({"judge": vllm_config}, "vllm")
    print("✓ vLLM judge created")

    # 2. Test enhanced mode
    enhanced_config = JUDGE_CONFIG_EXAMPLES["enhanced"]["config"]
    enhanced_judge = create_judge({"reward": {"config": enhanced_config}}, "enhanced")
    print("✓ Enhanced judge created")

    # 3. Test rule mode
    rule_judge = create_judge({}, "rule")
    print("✓ Rule judge created")

    print("\n🎯 Judge Factory is ready!")
    print("Options: vllm / enhanced / rule / auto")