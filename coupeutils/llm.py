import os
import logging
import vertexai
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel
from typing import Dict, List, Any, Optional, Union
import anthropic
from openai import OpenAI
import json5
import re


class LlmUtils:
    def __init__(self, vertex_project_id: Optional[str] = None):
        self.vertex_project_id = vertex_project_id or os.environ.get("PROJECT_ID")

    def send_to_vertex_ai(
        self,
        prompt: str,
        model_name: str = "gemini-1.5-flash-001",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        top_k: int = 40,
        top_p: float = 0.95,
        output_format: str = "json",
    ) -> Union[str, Dict]:
        """Sends a prompt to Vertex AI's Gemini and returns the generated text."""
        vertexai.init(project=self.vertex_project_id, location="us-central1")
        model =     model = GenerativeModel("gemini-1.5-flash-001")
        response = model.generate_content(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_k=top_k,
            top_p=top_p,
            response_mime_type=(
                "application/json" if output_format == "json" else "text/plain"
            ),
        )
        if output_format == "json":
            return self.clean_up_json_text(response.text)
        else:
            return response.text

    def send_to_vertex_ai_multimodal(
        self,
        prompt: str,
        file_uris: List[str],
        model_name: str = "gemini-1.5-flash-001",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        output_format: str = "json",
    ) -> Union[str, Dict]:
        """Sends a multimodal prompt (text and files - PDFs or images) to Vertex AI's Gemini."""
        vertexai.init(project=self.vertex_project_id, location="us-central1")
        model = generative_models.GenerativeModel(model_name=model_name)

        parts = []
        for uri in file_uris:
            if uri.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
                mime_type = (
                    "application/pdf" if uri.lower().endswith(".pdf") else "image/*"
                )
                file_part = generative_models.Part.from_uri(uri, mime_type=mime_type)
                parts.append(file_part)
            else:
                logging.warning(f"Unsupported file type: {uri}. Skipping.")
        parts.append(prompt)

        response = model.generate_content(
            parts,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            generation_config={
                "response_mime_type": (
                    "application/json" if output_format == "json" else "text/plain"
                )
            },
        )
        if output_format == "json":
            return self.clean_up_json_text(response.text)
        else:
            return response.text

    @staticmethod
    def send_to_anthropic(
        prompt: str,
        model_name: str = "claude-3-5-sonnet-20240620",
        max_tokens_to_sample: int = 4000,
        temperature: float = 0.0,
        output_format: str = "json",
    ) -> Union[str, Dict]:
        """Sends a prompt to Anthropic's Claude and returns the generated text."""
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.completions.create(
            model=model_name,
            max_tokens_to_sample=max_tokens_to_sample,
            temperature=temperature,
            prompt=prompt,
        )
        if output_format == "json":
            return {"text": response.completion}
        else:
            return response.completion

    @staticmethod
    def send_to_openai(
        prompt: str,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        output_format: str = "json",
    ) -> Union[str, Dict]:
        """Sends a prompt to OpenAI and returns the generated text."""
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=(
                {"type": "json_object"} if output_format == "json" else {"type": "text"}
            ),
        )
        if output_format == "json":
            return response.choices[0].message.content
        else:
            return response.choices[0].message.content.text

    @staticmethod
    def clean_up_json_text(text: str) -> Dict[str, Any]:
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        # Use regex to extract the content between the first and last curly brackets
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group()

        try:
            return json5.loads(text)
        except ValueError as e:
            print(f"Failed to parse JSON5. Input string: {text}")
            print(f"JSON5 parsing error: {str(e)}")
            raise  # Re-raise the ValueError to be handled up the stack


class PromptUtils:
    def __init__(self, prompt_file_path: str):
        self.prompt_template = self.load_prompt_from_file(prompt_file_path)
        self.formatted_prompt = None

    @staticmethod
    def load_prompt_from_file(file_path: str) -> str:
        """Loads a prompt from a text file."""
        with open(file_path, "r") as f:
            return f.read()

    def format_prompt(self, **kwargs) -> str:
        """Formats the loaded prompt template with data and additional keyword arguments."""
        self.formatted_prompt = self.prompt_template
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                value = json5.dumps(value)
            self.formatted_prompt = self.formatted_prompt.replace(f"${key}$", str(value))
        return self.formatted_prompt
