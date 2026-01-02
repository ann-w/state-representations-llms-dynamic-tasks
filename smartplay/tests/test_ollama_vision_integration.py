import types
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts.llm_query import get_ollama_query

class MockOllamaModule:
    def __init__(self):
        self.last_call = None
    def chat(self, model, messages, options=None):
        self.last_call = ("chat", model, messages, options)
        return {"message": {"content": "Action: 1"}}
    def generate(self, model, prompt, options=None):
        self.last_call = ("generate", model, prompt, options)
        return {"response": "Action: 1"}

mock = MockOllamaModule()
# Inject mock module
sys.modules.setdefault('ollama', mock)


def test_ollama_vision_chat_path(tmp_path, monkeypatch):
    # Replace imported name inside module scope if already imported
    import scripts.llm_query as lq
    lq.ollama = mock

    q = get_ollama_query("llava", vision=True)

    # Create dummy image file
    img_path = tmp_path / "frame.png"
    from PIL import Image
    Image.fromarray(np.zeros((4,4,3), dtype=np.uint8)).save(img_path)

    out, _, _ = q("Describe", images=[str(img_path)])
    assert "Action" in out
    assert mock.last_call[0] == "chat"


def test_ollama_text_fallback(monkeypatch):
    import scripts.llm_query as lq
    lq.ollama = mock
    q = get_ollama_query("llama3", vision=False)
    out, _, _ = q("Choose action 1")
    assert "Action" in out
    assert mock.last_call[0] == "generate"
