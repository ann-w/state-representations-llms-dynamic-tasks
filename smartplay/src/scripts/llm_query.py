import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import random

import dotenv
import httpx
import ollama
import requests
import torch
import yaml
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

if TYPE_CHECKING:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # pragma: no cover



def load_config(model_name: str) -> Dict:
    """
    Load YAML configuration for a given model.
    If the file is missing, fall back to a sane set of defaults
    (temperature / max_tokens / top-p / do_sample) for every model type.
    """
    model_name = model_name.split("/")[-1].lower()
    config_path = f"src/config/model_parameters/{model_name}.yaml"

    # If a YAML config exists – use it
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Otherwise return unified defaults
    logging.info(
        f"No configuration file found for model '{model_name}'. "
        "Using default parameters."
    )
    return {
        "temperature": 0.2,
        "max_tokens": 2056,
        "top_p": 0.95,
        "do_sample": True,
    }


def count_tokens(text: str, tokenizer) -> int:
    """
    Counts the number of tokens in a given text using the specified tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer: The HuggingFace tokenizer to use.

    Returns:
        int: The number of tokens in the input text.
    """
    tokens = tokenizer.encode(text, return_tensors="pt")
    return len(tokens[0])


def _is_qwen_vl_model(model_name: str) -> bool:
    normalized = (model_name or "").lower()
    return "qwen" in normalized and "vl" in normalized


def _should_mock_qwen_vl(config: Dict) -> bool:
    env_flag = os.getenv("QWEN_VL_MOCK", "").strip().lower()
    env_enabled = env_flag in {"1", "true", "yes", "on"}
    return bool(config.get("mock", False) or env_enabled)


def _extract_plain_text(messages: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        content = messages.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                chunk.get("text", "")
                for chunk in content
                if isinstance(chunk, dict) and chunk.get("type") == "text"
            )
        return str(content)
    if isinstance(messages, list):
        collected: List[str] = []
        for item in messages:
            if not isinstance(item, dict):
                collected.append(str(item))
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                collected.append(content)
            elif isinstance(content, list):
                collected.extend(
                    chunk.get("text", "")
                    for chunk in content
                    if isinstance(chunk, dict) and chunk.get("type") == "text"
                )
        return "\n".join(filter(None, collected))
    return str(messages)


def _shorten_text(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _load_vision_images(image_paths: Optional[List[str]]) -> List[Any]:
    if not image_paths:
        return []
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        logging.warning("Pillow is not installed; skipping image inputs for Qwen3-VL.")
        return []

    loaded_images: List[Any] = []
    for path in image_paths:
        if not path:
            continue
        try:
            with Image.open(path) as img:
                loaded_images.append(img.convert("RGB"))
        except Exception as exc:  # pragma: no cover
            logging.warning(f"Failed to load image '{path}' for Qwen3-VL: {exc}")
    return loaded_images


def _prepare_qwen_vl_messages(
    messages: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    image_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
        chat_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                qwen_ready = []
                for chunk in content:
                    if not isinstance(chunk, dict):
                        qwen_ready.append({"type": "text", "text": str(chunk)})
                        continue
                    chunk_type = chunk.get("type", "text")
                    if chunk_type == "text":
                        qwen_ready.append({"type": "text", "text": chunk.get("text", "")})
                    elif chunk_type == "image":
                        qwen_ready.append(chunk)
                    else:
                        qwen_ready.append({"type": "text", "text": str(chunk)})
            else:
                qwen_ready = [{"type": "text", "text": str(content)}]
            chat_messages.append({"role": role, "content": qwen_ready})
    else:
        text_payload = messages if isinstance(messages, str) else str(messages)
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_payload,
                    }
                ],
            }
        ]

    if image_paths:
        pil_images = _load_vision_images(image_paths)
        if pil_images:
            # attach images to the most recent user message to keep alignment with prompts
            for msg in reversed(chat_messages):
                if msg.get("role") == "user":
                    user_msg = msg
                    break
            else:
                user_msg = {"role": "user", "content": []}
                chat_messages.append(user_msg)
            for pil_image in reversed(pil_images):
                user_msg.setdefault("content", []).insert(0, {"type": "image", "image": pil_image})

    return chat_messages


def _build_mock_qwen_vl_query(
    model_name: str,
    count_token_usage: bool,
) -> Callable:
    def query_model(
        messages: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        count_token_usage: bool = count_token_usage,
        temperature: Optional[float] = None,
        images: Optional[List[str]] = None,
        **_: Any,
    ) -> Tuple[str, Optional[int], Optional[int]]:
        plain_text = _extract_plain_text(messages)
        vision_count = len(images) if images else 0
        response = (
            f"[Mock {model_name}] Vision frames: {vision_count}. "
            f"Prompt excerpt: {_shorten_text(plain_text or '[empty]')}"
        )
        if count_token_usage:
            approx_input = max(1, len(response.split()))
            approx_output = max(1, len(response.split()) // 4)
            return response, approx_input, approx_output
        return response, None, None

    return query_model


def get_deepseek_query(
    model_name: str,
    deepseek_api_key: str,
    config: Dict,
    count_token_usage: bool = True,
    temperature=None,
) -> Callable:
    """
    Creates a query function for DeepSeek's API, which is compatible with the OpenAI client.

    Args:
        model_name (str): The DeepSeek model to use (e.g., "deepseek-chat").
        deepseek_api_key (str): The API key for DeepSeek.
        config (Dict): Configuration parameters such as max_tokens, temperature, etc.
        count_token_usage (bool): Whether to include token usage in the return values.
        temperature: Temperature override; if None, will be read from config.

    Returns:
        Callable: A function that queries the DeepSeek model.
    """
    # Initialize the DeepSeek (OpenAI-compatible) client
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage: bool = count_token_usage,
        temperature=temperature,
    ) -> Tuple[str, int, int]:


        try:
            # Create a completion request to DeepSeek (OpenAI-compatible) chat API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=config.get("max_tokens", 2056),
                temperature=temperature if temperature is not None else config.get("temperature", 0.2),
                n=1,  # Generate one completion per request
                stream=False,
            )

            # Extract the assistant's response content
            output_text = response.choices[0].message.content

            # Attempt to extract token usage (if provided by DeepSeek)
            try:
                num_input_tokens = response.usage.prompt_tokens
                num_output_tokens = response.usage.completion_tokens
            except AttributeError:
                # If no usage info is in the response
                num_input_tokens = None
                num_output_tokens = None

            if count_token_usage:
                return output_text, num_input_tokens, num_output_tokens
            else:
                return output_text, None, None

        except Exception as e:
            logging.error(f"Error calling DeepSeek API: {e}")
            return "", None, None

    return query_model


def get_openai_query(
    model_name: str,
    openai_api_key: str,
    config: Dict,
    count_token_usage: bool = True,
    temperature=None,
) -> Callable:
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=openai_api_key)

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage=count_token_usage,
        temperature=temperature,
    ) -> Tuple[str, int, int]:

        try:
            # Create 
            response = client.responses.create(
                model=model_name,
                input=messages,
                max_output_tokens=config.get("max_tokens", 4096),
            )

            # Extract the assistant's response content
            output_text = response.output_text


            # Extract token counts from the response if available
            try:
                usage = response.usage
                num_input_tokens = usage.input_tokens
                num_output_tokens = usage.output_tokens
                total_tokens = usage.total_tokens
            except AttributeError:
                num_input_tokens = num_output_tokens = total_tokens = None

            # Return the assistant's response and token counts
            if count_token_usage:
                if num_input_tokens is not None and num_output_tokens is not None:
                    logging.info(
                        f"Input tokens: {num_input_tokens}, Output tokens: {num_output_tokens}, Total tokens: {total_tokens}"
                    )
                else:
                    logging.warning(
                        "Token usage information is not available in the response."
                    )
                return output_text, num_input_tokens, num_output_tokens
            else:
                return output_text, None, None

        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return "", None, None

    return query_model


def get_huggingface_query(
    model_id: str, config: Dict, count_token_usage: bool = False
) -> Callable:
    """
    Creates a query function for a HuggingFace model using the API.

    Args:
        model_id (str): The ID of the HuggingFace model to use.
        config (Dict): A dictionary of configuration parameters.

    Returns:
        Callable: A function that queries the model with a given message.
    """
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    ENDPOINT = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def query(
        payload,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        rate_limit_delay: float = 2.0,
    ):
        retries = 0
        response = None

        while retries < max_retries:
            try:
                response = requests.post(ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response and response.status_code == 429:  # too many requests
                    logging.warning(
                        f"Rate limit hit. Sleeping for {rate_limit_delay} seconds."
                    )
                    time.sleep(rate_limit_delay)
                    rate_limit_delay *= 2  # Exponential backoff for rate limits
                else:
                    logging.warning(
                        f"HTTPError: {e}. Retrying in {backoff_factor * (2 ** retries)} seconds..."
                    )
                    time.sleep(
                        backoff_factor * (2**retries)
                    )  # Exponential backoff for other errors
                retries += 1
            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"RequestException: {e}. Retrying in {backoff_factor * (2 ** retries)} seconds..."
                )
                time.sleep(backoff_factor * (2**retries))
                retries += 1

        logging.warning("Failed to get a valid response after retries.")
        return None

    def query_model(
        messages: List[Dict[str, str]],
        count_token_usage: bool = count_token_usage,
        temperature=None,
    ) -> Tuple[str, int, int]:

        payload = {
            "inputs": messages,
            "parameters": {
                "do_sample": config.get("do_sample", False),
                "max_new_tokens": config.get("max_tokens", 2056),
                "temperature": temperature if temperature is not None else config.get("temperature", 0.2),
                "return_full_text": config.get("return_full_text", False),
            },
        }

        # Count input tokens if required
        num_input_tokens = (
            count_tokens(messages, tokenizer) if count_token_usage else None
        )

        response = query(payload)

        # Raise an exception if the query failed (i.e., response is None)
        if response is None:
            raise RuntimeError("Failed to get a valid response after retries.")

        output_text = response[0]["generated_text"]

        # Count output tokens if required
        num_output_tokens = (
            count_tokens(output_text, tokenizer) if count_token_usage else None
        )

        # Return based on token usage flag
        if count_token_usage:
            return output_text, num_input_tokens, num_output_tokens
        else:
            return output_text, None, None

    return query_model


def get_qwen3_vl_local_query(
    model_name: str,
    config: Dict,
    count_token_usage: bool = False,
) -> Callable:
    if _should_mock_qwen_vl(config):
        logging.warning(
            "Qwen3-VL mock mode is enabled. Set QWEN_VL_MOCK=0 or mock=false in the model config to run the real model."
        )
        return _build_mock_qwen_vl_query(model_name, count_token_usage)

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Qwen3-VL requires a newer transformers build. Please install transformers>=4.43.0 with Qwen3 VL support."
        ) from exc

    torch.cuda.empty_cache()
    device_map = config.get("device_map", "auto")
    dtype_config = config.get("dtype", config.get("torch_dtype", "auto"))
    torch_dtype: Union[str, torch.dtype]
    if isinstance(dtype_config, str):
        dtype_key = dtype_config.lower()
        if dtype_key == "auto":
            torch_dtype = "auto"
        elif dtype_key in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif dtype_key in ("float16", "fp16", "half"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = dtype_config

    attn_impl = "flash_attention_2" if config.get("use_flash_attention", False) else None

    if attn_impl and (torch_dtype == "auto" or torch_dtype == torch.float32):
        logging.warning(
            "FlashAttention 2 benefits from bfloat16/fp16. Defaulting dtype to bfloat16 for Qwen3-VL."
        )
        torch_dtype = torch.bfloat16

    loading_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }

    quantization_mode = config.get("use_quantization")
    if quantization_mode:
        try:
            from transformers import BitsAndBytesConfig

            if quantization_mode == "8bit" or quantization_mode is True:
                loading_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization_mode == "4bit":
                loading_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:  # pragma: no cover
            logging.warning("bitsandbytes is not available; proceeding without quantization.")

    model: Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **{k: v for k, v in loading_kwargs.items() if v is not None},
    )
    model.eval()

    processor: AutoProcessor = AutoProcessor.from_pretrained(model_name)

    max_new_tokens = config.get("max_tokens", 512)
    do_sample = config.get("do_sample", True)
    top_p = config.get("top_p", 0.9)
    default_temperature = config.get("temperature", 0.2)

    def query_model(
        messages: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        count_token_usage: bool = count_token_usage,
        temperature: Optional[float] = None,
        images: Optional[List[str]] = None,
        **_: Any,
    ) -> Tuple[str, Optional[int], Optional[int]]:
        chat_messages = _prepare_qwen_vl_messages(messages, image_paths=images)

        try:
            inputs = processor.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            logging.error(f"Failed to tokenize Qwen3-VL prompt: {exc}")
            return "", None, None

        try:
            inputs = inputs.to(model.device)  # type: ignore[attr-defined]
        except Exception:
            # Some accelerate configs do not expose model.device; fall back to cuda/cpu heuristics
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        temperature_value = temperature if temperature is not None else default_temperature
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature_value,
            "do_sample": do_sample,
            "top_p": top_p,
        }

        input_token_tensor = inputs.get("input_ids")
        num_input_tokens: Optional[int] = None
        if count_token_usage and input_token_tensor is not None:
            num_input_tokens = int(input_token_tensor.shape[-1])

        try:
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            logging.error(f"Qwen3-VL generation failed: {exc}")
            return "", num_input_tokens, None

        trimmed_sequences = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_text = decoded[0] if isinstance(decoded, list) else decoded

        num_output_tokens: Optional[int] = None
        if count_token_usage:
            num_output_tokens = int(sum(t.numel() for t in trimmed_sequences))

        return output_text, num_input_tokens, num_output_tokens

    return query_model


def get_local_huggingface_query(
    model_name: str, config: Dict, count_token_usage: bool = False
) -> Callable:
    """
    Creates a local query function for a HuggingFace model.

    Args:
        model_name (str): The name of the HuggingFace model to use, loaded locally.
        config (Dict): A dictionary of configuration parameters for local execution.

    Returns:
        Callable: A function that locally queries the model with a given message.
    """
    if _is_qwen_vl_model(model_name):
        return get_qwen3_vl_local_query(model_name, config, count_token_usage)

    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Set dtype for mixed precision
    dtype = torch.float16 if config.get("use_mixed_precision", False) else torch.float32

    # Default parameters for loading the model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": None,  # Ensure no offloading
        "trust_remote_code": True,
    }

    # Dynamically update model_kwargs with relevant config parameters
    valid_model_kwargs_keys = {"use_quantization", "use_flash_attention"}

    for key in valid_model_kwargs_keys:
        if key in config:
            if key == "use_quantization" and config[key]:
                try:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                except ImportError:
                    logging.info(
                        "Quantization is not supported. Please install the necessary libraries or disable this feature."
                    )
            elif key == "use_flash_attention" and config[key]:
                model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)  # Move the model to the GPU after loading

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set up the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    def query_model(
        messages, count_token_usage: bool = count_token_usage, temperature=0.2
    ) -> Tuple[str, int, int]:
        generation_args = {
            "max_new_tokens": config.get("max_tokens", 2056),
            "return_full_text": config.get("return_full_text", False),
            "temperature": temperature,
            "do_sample": config.get("do_sample", True),
        }

        # Concatenate messages into a single input string
        if isinstance(messages, str):
            input_text = messages
        elif isinstance(messages, list) and all(
            isinstance(msg, dict) and "text" in msg for msg in messages
        ):
            input_text = " ".join([msg["text"] for msg in messages])
        else:
            logging.error(
                "Invalid input type for 'messages'. Expected a string or a list of dictionaries with 'text' keys."
            )
            return "", None, None

        # Count input tokens if required
        num_input_tokens = (
            count_tokens(input_text, tokenizer) if count_token_usage else None
        )

        try:
            output = pipe(input_text, **generation_args)
            output_text = output[0]["generated_text"]
        except Exception as e:
            logging.error(f"Error during text generation: {e}")
            return "", None, None

        # Count output tokens if required
        num_output_tokens = (
            count_tokens(output_text, tokenizer) if count_token_usage else None
        )

        # Return based on token usage flag
        if count_token_usage:
            return output_text, num_input_tokens, num_output_tokens
        else:
            return output_text, None, None

    return query_model


# Create a persistent client for connection pooling
_ollama_client = None


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = httpx.Client(
            base_url="http://127.0.0.1:11434",
            timeout=30.0,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _ollama_client


def get_ollama_query(
    model_name: str,
    config: dict = None,
    count_token_usage: bool = False,
    temperature: float = None,
    vision: bool = False,
) -> Callable:
    """
    Creates a query function for an Ollama model.

    Args:
        model_name (str): The name of the Ollama model to use.
        config (Dict): A dictionary of configuration parameters.
        count_token_usage (bool): Whether to count the token usage.

    Returns:
        Callable: A function that queries the model with a given message.
    """

    def query_model(
        prompt: str,
        count_token_usage: bool = count_token_usage,
        temperature: float = temperature,
        images: List[str] | None = None,
    ) -> Tuple[str, int, int]:
        # Speed optimizations
        options = {
            "num_batch": 1024,  # Larger batch for faster text-only generation
            "num_gpu": -1,      # Use all available GPU layers
            "num_thread": -1,   # Use all CPU threads
            # NOTE: num_keep deliberately omitted for now in base dict; we'll add conditionally
            "keep_alive": "10m",  # Keep model loaded for 10 minutes
        }

        # Handle temperature parameter
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = 0.2  # Default fast temperature

        try:
            if vision and images:
                # Vision path uses chat API. Some Ollama vision models (e.g. qwen2.5vl) currently
                # error with: "Failed to create new sequence: SameBatch may not be specified within numKeep"
                # when a large num_keep is set or when num_keep overlaps image embedding token spans.
                # We therefore (a) remove num_keep entirely and (b) reduce num_batch to a modest value.
                vision_options = options.copy()
                # Lower batch size for multimodal inputs; 1024 can trigger internal indexing issues.
                vision_options["num_batch"] = 64
                # Ensure we do NOT set num_keep for vision to avoid SameBatch conflict.
                if "num_keep" in vision_options:
                    vision_options.pop("num_keep")
                # (Optional) allow future config override
                if config and (override := config.get("vision_num_batch")):
                    vision_options["num_batch"] = override
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images,
                    }
                ]
                response = ollama.chat(model=model_name, messages=messages, options=vision_options)
                result = response["message"]["content"]
            else:
                response = ollama.generate(model=model_name, prompt=prompt, options=options)
                result = response["response"]
        except Exception as e:
            logging.error(f"Ollama query failed (vision={vision}): {e}. Falling back to text-only generate.")
            try:
                response = ollama.generate(model=model_name, prompt=prompt, options=options)
                result = response.get("response", "")
            except Exception as e2:
                logging.error(f"Fallback Ollama generate failed: {e2}")
                return "", None, None

        return result, None, None

    return query_model


def get_query(model_name: str, model_type: str, vision: bool = False) -> Callable:
    """Get the correct model query function based on the model name and type.
    Params:
        model_name (str): The name of the model to use.
        model_type (str): The type of query to use. Either "openai", "huggingface",
                          "local_huggingface", "ollama", or "deepseek".
    """
    if model_type == "ollama":
        return get_ollama_query(model_name, {}, vision=vision)

    config = load_config(model_name)

    dotenv.load_dotenv()

    if model_type == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("No OpenAI API key found. Please set it in the .env file.")
        return get_openai_query(model_name, openai_api_key, config)
    elif model_type == "local_huggingface":
        return get_local_huggingface_query(model_name, config)
    elif model_type == "huggingface":
        return get_huggingface_query(model_name, config)
    elif model_type == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError(
                "No DeepSeek API key found. Please set it in the .env file."
            )
        return get_deepseek_query(model_name, deepseek_api_key, config)
    else:
        raise ValueError(
            f"Unknown query type: {model_type}. Please use 'openai', 'huggingface', 'local_huggingface', 'ollama', or 'deepseek'."
        )
