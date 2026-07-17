"""Video captioning with a Gemma-style multimodal chat model.

Videos are passed to the processor as evenly sampled frames converted to
image content items, mirroring how the dataset's fine-tuning pipeline feeds
Gemma-4 (see gemma4-FT's ds_wrapper).
"""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class Captioner:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        token: str | None = None,
    ):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            token=token,
            dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        self.model.eval()

    def caption(
        self,
        frames: list[Image.Image],
        user_text: str,
        system_text: str | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )
        return " ".join(text.split())
