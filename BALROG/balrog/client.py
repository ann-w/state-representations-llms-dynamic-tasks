import base64
import datetime
import logging
import time
import json
import csv
import os
import tempfile
from collections import namedtuple
from io import BytesIO

import google.generativeai as genai
from anthropic import Anthropic
from google.generativeai import caching
from openai import OpenAI

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
except ImportError:  # pragma: no cover - optional dependency
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForVision2Seq = None
    AutoProcessor = None

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional dependency
    InferenceClient = None

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "raw_completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
        "prompt",
    ],
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """Base class for LLM client wrappers.

    Provides common functionality for interacting with different LLM APIs, including
    handling retries and common configuration settings. Subclasses should implement
    the `generate` method specific to their LLM API.
    """

    def __init__(self, client_config):
        """Initialize the LLM client wrapper with configuration settings.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.load_kwargs = getattr(client_config, "load_kwargs", {}) or {}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.alternate_roles = client_config.alternate_roles

    def serialize_prompt(self, messages):
        """Serialize prompt messages for logging and debugging purposes."""

        serialized_messages = []
        for msg in messages:
            serialized_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "has_attachment": msg.attachment is not None,
                }
            )
        return serialized_messages

    def generate(self, messages):
        """Generate a response from the LLM given a list of messages.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)
        raise Exception(f"Failed to execute {func.__name__} after {self.max_retries} retries.")


def process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for OpenAI.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    """Process an image for Anthropic's Claude API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for Claude.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for Anthropic
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
    }


def ensure_pil_image(attachment):
    """Convert various attachment types to RGB PIL images for local models."""

    if Image is None:
        raise ImportError("Pillow is required for image attachments. Install it via 'pip install pillow'.")

    if isinstance(attachment, Image.Image):
        return attachment.convert("RGB")

    if isinstance(attachment, bytes):
        return Image.open(BytesIO(attachment)).convert("RGB")

    if isinstance(attachment, str) and os.path.exists(attachment):
        return Image.open(attachment).convert("RGB")

    if hasattr(attachment, "shape"):
        return Image.fromarray(attachment).convert("RGB")

    raise ValueError("Unsupported attachment type for Qwen local client. Provide a PIL image or path/bytes array.")


class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "nvidia" or self.client_name.lower() == "xai":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # For OpenAI, always use the standard API regardless of base_url
                self.client = OpenAI()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(process_image_openai(msg.attachment))
            if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == msg.role:
                converted_messages[-1]["content"].extend(new_content)
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        raw_text = response.choices[0].message.content.strip()

        return LLMResponse(
            model_id=self.model_id,
            completion=raw_text,
            raw_completion=raw_text,
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
            prompt=self.serialize_prompt(messages),
        )


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with Google's Generative AI API."""

    def __init__(self, client_config):
        """Initialize the GoogleGenerativeAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Generative AI client if not already initialized."""
        if not self._initialized:
            self.model = genai.GenerativeModel(self.model_id)

            # Create kwargs dictionary for GenerationConfig
            client_kwargs = {
                "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                client_kwargs["temperature"] = temperature

            self.generation_config = genai.types.GenerationConfig(**client_kwargs)
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Generative AI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Generative AI API.
        """
        # Convert standard Message objects to Gemini's format
        converted_messages = []
        for msg in messages:
            parts = []
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            if msg.content:
                parts.append(msg.content)
            if msg.attachment is not None:
                parts.append(msg.attachment)
            converted_messages.append(
                {
                    "role": role,
                    "parts": parts,
                }
            )
        return converted_messages

    def get_completion(self, converted_messages, max_retries=5, delay=5):
        """Get the completion from the model with retries upon failure.

        Args:
            converted_messages (list): Messages formatted for the Generative AI API.
            max_retries (int, optional): Maximum number of retries. Defaults to 5.
            delay (int, optional): Delay between retries in seconds. Defaults to 5.

        Returns:
            Response object from the API.

        Raises:
            Exception: If the API call fails after the maximum number of retries.
        """
        retries = 0
        while retries < max_retries:
            try:
                response = self.model.generate_content(
                    converted_messages,
                    generation_config=self.generation_config,
                )
                return response
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during generate_content: {e}. Retry {retries}/{max_retries}")
                sleep_time = delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)

        # If maximum retries are reached and still no valid response
        raise Exception(f"Failed to get a valid completion after {max_retries} retries.")

    def extract_completion(self, response):
        """Extract the completion text from the API response.

        Args:
            response: The response object from the API.

        Returns:
            str: The extracted completion text.
            
        Raises:
            Exception: If response is None or missing expected fields.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")

        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")
            
        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        text = getattr(content_parts[0], "text", None)
        if text is None:
            raise Exception("No text found in the content parts.")
            
        return text.strip()

    def generate(self, messages):
        """Generate a response from the Generative AI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Generative AI API.
        """
        self._initialize_client()

        converted_messages = self.convert_messages(messages)

        def api_call():
            response = self.model.generate_content(
                converted_messages,
                generation_config=self.generation_config,
            )
            # Attempt to extract completion immediately after API call
            completion = self.extract_completion(response)
            # Return both response and completion if successful
            return response, completion

        try:
            # Execute the API call and extraction together with retries
            response, completion = self.execute_with_retries(api_call)

            # Check if the successful response contains an empty completion
            if not completion or completion.strip() == "":
                logger.warning(f"Gemini returned an empty completion for model {self.model_id}. Returning default empty response.")
                return LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    raw_completion="",
                    stop_reason="empty_response",
                    input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    reasoning=None,
                    prompt=self.serialize_prompt(messages),
                )
            else:
                # If completion is not empty, return the normal response
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    raw_completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", "unknown")
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=(
                        getattr(response.usage_metadata, "prompt_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    output_tokens=(
                        getattr(response.usage_metadata, "candidates_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    reasoning=None,
                    prompt=self.serialize_prompt(messages),
                )
        except Exception as e:
            logger.error(f"API call failed after {self.max_retries} retries: {e}. Returning empty completion.")
            # Return a default response indicating failure
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                raw_completion="",
                stop_reason="error_max_retries",
                input_tokens=0, # Assuming 0 tokens consumed if call failed
                output_tokens=0,
                reasoning=None,
                prompt=self.serialize_prompt(messages),
            )


class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API."""

    def __init__(self, client_config):
        """Initialize the ClaudeWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Claude client if not already initialized."""
        if not self._initialized:
            self.client = Anthropic()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Claude API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Claude API.
        """
        converted_messages = []
        for msg in messages:
            converted_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
            if converted_messages[-1]["role"] == "system":
                # Claude doesn't support system prompt and requires alternating roles
                converted_messages[-1]["role"] = "user"
                converted_messages.append({"role": "assistant", "content": "I'm ready!"})
            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_claude(msg.attachment))

        return converted_messages

    def generate(self, messages):
        """Generate a response from the Claude API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Claude API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        completion = response.content[0].text.strip()

        return LLMResponse(
            model_id=self.model_id,
            completion=completion,
            raw_completion=completion,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
            prompt=self.serialize_prompt(messages),
        )


class HuggingFaceWrapper(LLMClientWrapper):
    """Wrapper for interacting with Hugging Face Inference Endpoints."""

    def __init__(self, client_config):
        super().__init__(client_config)
        if InferenceClient is None:
            raise ImportError(
                "huggingface_hub is required for HuggingFace clients. Install it via 'pip install huggingface_hub'."
            )
        self._initialized = False
        self.provider = getattr(client_config, "provider", None)
        if not self.provider:
            # Default to the plain HTTPS inference backend to avoid optional dependencies
            self.provider = "hf-inference"

    def _initialize_client(self):
        if self._initialized:
            return

        token = (
            os.environ.get("HF_API_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        endpoint = self.base_url.strip() if self.base_url else None

        client_kwargs = {"timeout": self.timeout}
        if endpoint:
            self.client = InferenceClient(
                endpoint_url=endpoint,
                token=token,
                provider=self.provider,
                **client_kwargs,
            )
        else:
            self.client = InferenceClient(
                model=self.model_id,
                token=token,
                provider=self.provider,
                **client_kwargs,
            )

        self._initialized = True

    def _ensure_text_only(self, messages):
        for msg in messages:
            if msg.attachment is not None:
                raise ValueError("HuggingFace client does not currently support image attachments in BALROG.")

    def convert_messages(self, messages):
        parts = []
        for msg in messages:
            role = msg.role.capitalize()
            parts.append(f"{role}: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def generate(self, messages):
        self._initialize_client()
        self._ensure_text_only(messages)
        prompt = self.convert_messages(messages)

        def api_call():
            generation_kwargs = {
                "max_new_tokens": self.client_kwargs.get("max_tokens", 1024),
                "return_full_text": False,
            }
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                generation_kwargs["temperature"] = temperature
            return self.client.text_generation(prompt, **generation_kwargs)

        completion = self.execute_with_retries(api_call)
        completion = completion.strip()

        return LLMResponse(
            model_id=self.model_id,
            completion=completion,
            raw_completion=completion,
            stop_reason="stop",
            input_tokens=0,
            output_tokens=0,
            reasoning=None,
            prompt=self.serialize_prompt(messages),
        )


class QwenTransformersWrapper(LLMClientWrapper):
    """Wrapper for running Qwen VL models locally via Hugging Face Transformers."""

    def __init__(self, client_config):
        super().__init__(client_config)
        missing = []
        if torch is None:
            missing.append("torch")
        if (AutoModelForCausalLM is None and AutoModelForVision2Seq is None) or AutoProcessor is None:
            missing.append("transformers")
        if Image is None:
            missing.append("pillow")
        if missing:
            raise ImportError(
                "The local Qwen client requires the following packages: "
                + ", ".join(missing)
                + ". Install them via 'pip install torch transformers pillow'."
            )
        self._initialized = False
        self._target_device = None

    def _default_max_memory(self):
        max_memory = {"cpu": "120GiB"}
        if torch is None or not torch.cuda.is_available():
            return max_memory

        reserve_ratio = 0.92  # keep headroom to avoid fragmentation spikes
        fallback_total = 40 * 1024**3
        for idx in range(torch.cuda.device_count()):
            try:
                total_mem = torch.cuda.get_device_properties(idx).total_memory
            except Exception:  # pragma: no cover - CUDA device query failures
                total_mem = fallback_total
            budget_gib = max(1, int((total_mem / (1024 ** 3)) * reserve_ratio))
            max_memory[idx] = f"{budget_gib}GiB"
        return max_memory

    def _build_default_load_kwargs(self):
        if torch is None:
            return {}

        load_kwargs = {}
        if torch.cuda.is_available():
            alloc_conf = "expandable_segments:True"
            if not os.environ.get("PYTORCH_ALLOC_CONF"):
                os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
            if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

            load_kwargs.update(
                {
                    "torch_dtype": torch_dtype,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "max_memory": self._default_max_memory(),
                    "offload_folder": os.environ.get(
                        "BALROG_QWEN_OFFLOAD",
                        os.path.join(tempfile.gettempdir(), "qwen_offload"),
                    ),
                }
            )
        else:
            load_kwargs["torch_dtype"] = torch.float32

        return load_kwargs

    def _should_use_vision_loader(self, config):
        if config is None:
            hint = self.model_id.lower()
            return "vl" in hint or "vision" in hint

        model_type = (getattr(config, "model_type", "") or "").lower()
        if "vl" in model_type or "vision" in model_type:
            return True

        architectures = getattr(config, "architectures", None) or []
        for arch in architectures:
            arch_name = arch.lower()
            if "vl" in arch_name or "vision" in arch_name:
                return True

        auto_map = getattr(config, "auto_map", None) or {}
        if isinstance(auto_map, dict) and "AutoModelForVision2Seq" in auto_map:
            return True

        return False

    def _resolve_target_device(self):
        if torch is None:
            return "cpu"

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            for device in device_map.values():
                if device and device != "cpu":
                    return torch.device(device)
            first_device = next(iter(device_map.values()))
            return torch.device(first_device)

        if hasattr(self.model, "device"):
            return self.model.device

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_model(self):
        if self._initialized:
            return

        config = None
        if AutoConfig is not None:
            try:
                config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as exc:  # pragma: no cover - config loading
                logger.warning("Could not load config for %s: %s", self.model_id, exc)

        prefer_vision_loader = self._should_use_vision_loader(config)

        model_kwargs = {"trust_remote_code": True}
        model_kwargs.update(self._build_default_load_kwargs())

        # allow extra load_kwargs from config, if any
        if self.load_kwargs:
            model_kwargs.update(self.load_kwargs)

        offload_folder = model_kwargs.get("offload_folder")
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)

        load_errors = []
        self.model = None

        loader_sequence = []
        if prefer_vision_loader and AutoModelForVision2Seq is not None:
            loader_sequence.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
        if AutoModelForCausalLM is not None:
            loader_sequence.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        if not prefer_vision_loader and AutoModelForVision2Seq is not None:
            loader_sequence.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))

        if not loader_sequence:
            raise RuntimeError("transformers is missing required auto-model classes for Qwen")

        for label, loader_cls in loader_sequence:
            try:
                self.model = loader_cls.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception as exc:  # pragma: no cover - model loading
                load_errors.append(f"{label}: {exc}")
                self.model = None

        if self.model is None:
            combined = "; ".join(load_errors) if load_errors else "unknown error"
            raise RuntimeError(f"Failed to load Qwen model '{self.model_id}': {combined}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - processor loading
            raise RuntimeError(f"Failed to load Qwen processor '{self.model_id}': {exc}") from exc

        if self.model.config.pad_token_id is None and hasattr(self.processor, "tokenizer"):
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        if not torch.cuda.is_available():
            self.model.to("cpu")

        self.model.eval()
        self._target_device = self._resolve_target_device()
        self._initialized = True

    def _prepare_messages(self, messages):
        converted = []
        images = []
        for msg in messages:
            content_blocks = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            if msg.attachment is not None:
                image = ensure_pil_image(msg.attachment)
                content_blocks.append({"type": "image", "image": image})
                images.append(image)
            if content_blocks:
                converted.append({"role": msg.role, "content": content_blocks})
        return converted, images

    def _build_generation_kwargs(self):
        generation_kwargs = {
            key: value
            for key, value in self.client_kwargs.items()
            if key not in {"max_tokens", "temperature"}
        }

        max_new_tokens = self.client_kwargs.get("max_tokens")
        if max_new_tokens is None:
            max_new_tokens = 1024
        generation_kwargs["max_new_tokens"] = max_new_tokens

        temperature = self.client_kwargs.get("temperature")
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
            generation_kwargs.setdefault("do_sample", temperature > 0)
        else:
            generation_kwargs.setdefault("do_sample", False)

        return generation_kwargs

    def generate(self, messages):
        self._initialize_model()
        converted_messages, images = self._prepare_messages(messages)
        if not converted_messages:
            raise ValueError("No valid messages to send to the Qwen client.")

        prompt_text = self.processor.apply_chat_template(
            converted_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        processor_kwargs = {
            "text": [prompt_text],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images

        model_inputs = self.processor(**processor_kwargs)
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self._target_device)
        else:
            model_inputs = {
                k: v.to(self._target_device) if isinstance(v, torch.Tensor) else v
                for k, v in model_inputs.items()
            }

        input_length = model_inputs["input_ids"].shape[-1]
        generation_kwargs = self._build_generation_kwargs()

        def model_call():
            with torch.no_grad():
                return self.model.generate(**model_inputs, **generation_kwargs)

        generated_ids = self.execute_with_retries(model_call)
        completion_token_ids = generated_ids[:, input_length:]
        completion = self.processor.batch_decode(completion_token_ids, skip_special_tokens=True)[0].strip()

        input_tokens = int(input_length)
        output_tokens = int(completion_token_ids.shape[-1])

        return LLMResponse(
            model_id=self.model_id,
            completion=completion,
            raw_completion=completion,
            stop_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=None,
            prompt=self.serialize_prompt(messages),
        )

def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the client name.

    Args:
        client_config: Configuration object containing client-specific settings.

    Returns:
        callable: A factory function that returns an instance of the appropriate LLM client.
    """

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower or "xai" in client_name_lower:
            # NVIDIA and XAI use OpenAI-compatible API, so we use the OpenAI wrapper
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_name_lower:
            return ClaudeWrapper(client_config)
        elif "qwen" in client_name_lower:
            return QwenTransformersWrapper(client_config)
        elif "huggingface" in client_name_lower or "hf" == client_name_lower:
            return HuggingFaceWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
