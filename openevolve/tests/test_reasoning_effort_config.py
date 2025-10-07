"""
Tests for reasoning_effort configuration parameter
"""

import unittest
import yaml
import asyncio
from unittest.mock import Mock
import tempfile
import os

from openevolve.config import Config, LLMConfig, LLMModelConfig
from openevolve.llm.openai import OpenAILLM


class TestReasoningEffortConfig(unittest.TestCase):
    """Tests for reasoning_effort parameter handling in configuration"""

    def test_reasoning_effort_in_llm_config(self):
        """Test that reasoning_effort can be loaded from YAML config at LLM level"""
        yaml_config = {
            "log_level": "INFO",
            "llm": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "test-key",
                "temperature": 0.7,
                "max_tokens": 100000,
                "timeout": 5000,
                "retries": 1000000,
                "reasoning_effort": "high",
                "models": [
                    {
                        "name": "gpt-oss-120b",
                        "weight": 1.0
                    }
                ]
            }
        }
        
        # This should not raise a TypeError
        config = Config.from_dict(yaml_config)
        
        self.assertEqual(config.llm.reasoning_effort, "high")
        self.assertEqual(config.llm.models[0].reasoning_effort, "high")

    def test_reasoning_effort_in_model_config(self):
        """Test that reasoning_effort can be specified per model"""
        yaml_config = {
            "log_level": "INFO", 
            "llm": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "test-key",
                "models": [
                    {
                        "name": "gpt-oss-120b",
                        "weight": 1.0,
                        "reasoning_effort": "medium"
                    },
                    {
                        "name": "gpt-4",
                        "weight": 0.5,
                        "reasoning_effort": "high"
                    }
                ]
            }
        }
        
        config = Config.from_dict(yaml_config)
        
        self.assertEqual(config.llm.models[0].reasoning_effort, "medium")
        self.assertEqual(config.llm.models[1].reasoning_effort, "high")

    def test_reasoning_effort_inheritance(self):
        """Test that model configs inherit reasoning_effort from parent LLM config"""
        yaml_config = {
            "log_level": "INFO",
            "llm": {
                "api_base": "https://api.openai.com/v1", 
                "api_key": "test-key",
                "reasoning_effort": "low",
                "models": [
                    {
                        "name": "gpt-oss-120b",
                        "weight": 1.0
                        # No reasoning_effort specified - should inherit
                    }
                ]
            }
        }
        
        config = Config.from_dict(yaml_config)
        
        self.assertEqual(config.llm.reasoning_effort, "low")
        self.assertEqual(config.llm.models[0].reasoning_effort, "low")

    def test_reasoning_effort_model_override(self):
        """Test that model-level reasoning_effort overrides LLM-level"""
        yaml_config = {
            "log_level": "INFO",
            "llm": {
                "api_base": "https://api.openai.com/v1",
                "api_key": "test-key", 
                "reasoning_effort": "low",
                "models": [
                    {
                        "name": "gpt-oss-120b",
                        "weight": 1.0,
                        "reasoning_effort": "high"  # Override parent
                    }
                ]
            }
        }
        
        config = Config.from_dict(yaml_config)
        
        self.assertEqual(config.llm.reasoning_effort, "low")
        self.assertEqual(config.llm.models[0].reasoning_effort, "high")

    def test_openai_llm_uses_reasoning_effort(self):
        """Test that OpenAILLM stores and uses reasoning_effort from config"""
        # Create a mock model config with reasoning_effort
        model_cfg = Mock()
        model_cfg.name = "gpt-oss-120b"
        model_cfg.system_message = "system"
        model_cfg.temperature = 0.7
        model_cfg.top_p = 0.95
        model_cfg.max_tokens = 4096
        model_cfg.timeout = 60
        model_cfg.retries = 3
        model_cfg.retry_delay = 5
        model_cfg.api_base = "https://api.openai.com/v1"
        model_cfg.api_key = "test-key"
        model_cfg.random_seed = None
        model_cfg.reasoning_effort = "high"
        
        # Mock OpenAI client to avoid actual API calls
        with unittest.mock.patch('openai.OpenAI'):
            llm = OpenAILLM(model_cfg)
            
        # Verify the reasoning_effort is stored
        self.assertEqual(llm.reasoning_effort, "high")

    def test_reasoning_effort_passed_to_api_params(self):
        """Test that reasoning_effort is included in API call parameters"""
        model_cfg = Mock()
        model_cfg.name = "gpt-oss-120b" 
        model_cfg.system_message = "system"
        model_cfg.temperature = 0.7
        model_cfg.top_p = 0.95
        model_cfg.max_tokens = 4096
        model_cfg.timeout = 60
        model_cfg.retries = 3
        model_cfg.retry_delay = 5
        model_cfg.api_base = "https://api.openai.com/v1"
        model_cfg.api_key = "test-key"
        model_cfg.random_seed = None
        model_cfg.reasoning_effort = "medium"
        
        with unittest.mock.patch('openai.OpenAI'):
            llm = OpenAILLM(model_cfg)
            
            # Test the _call_api method directly with mocked client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            llm.client.chat.completions.create.return_value = mock_response
            
            # Test OpenAI reasoning model (gpt-oss-120b at openai.com should use reasoning logic)
            test_params = {
                "model": "gpt-oss-120b",
                "messages": [{"role": "system", "content": "Test"}, {"role": "user", "content": "Test"}],
                "max_completion_tokens": 4096,
                "reasoning_effort": "medium"
            }
            
            result = asyncio.run(llm._call_api(test_params))
            
            # Verify the API was called with reasoning_effort
            llm.client.chat.completions.create.assert_called_once_with(**test_params)

    def test_yaml_file_loading_with_reasoning_effort(self):
        """Test loading reasoning_effort from actual YAML file"""
        yaml_content = """
log_level: INFO
llm:
  api_base: https://api.openai.com/v1
  api_key: test-key
  temperature: 0.7
  max_tokens: 100000
  timeout: 5000
  retries: 1000000
  reasoning_effort: high
  models:
  - name: gpt-oss-120b
    weight: 1.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                config = Config.from_yaml(f.name)
                self.assertEqual(config.llm.reasoning_effort, "high")
                self.assertEqual(config.llm.models[0].reasoning_effort, "high")
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()