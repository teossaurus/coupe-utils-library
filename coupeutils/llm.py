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
import json


class LlmUtils:
    def __init__(self, vertex_project_id: Optional[str] = None):
        self.vertex_project_id = vertex_project_id or os.environ.get("PROJECT_ID")

    def send_to_vertex_ai(
        self,
        prompt: str,
        system_instruction=None,
        model_name: str = "gemini-1.5-flash-001",
        temperature: float = 0.0,
        max_output_tokens: int = 8000,
        output_format: str = "json",
        use_json_assist: bool = False,
    ) -> Union[str, Dict]:
        """Sends a prompt to Vertex AI's Gemini and returns the generated text."""
        vertexai.init(project=self.vertex_project_id, location="us-central1")

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": (
                "application/json" if output_format == "json" else "text/plain"
            ),
        }

        safety_config = [
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]

        if system_instruction:
            model = GenerativeModel(
                system_instruction=system_instruction,
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_config,
            )

        else:
            model = GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_config,
            )

        response = model.generate_content(prompt)

        if output_format == "json":
            return self.clean_up_json_text(response.text, use_json_assist)
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
        use_json_assist: bool = False,
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
            return self.clean_up_json_text(response.text, use_json_assist)
        else:
            return response.text

    @staticmethod
    def send_to_anthropic(
        prompt: str,
        model_name: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        output_format: str = "json",
        use_json_assist: bool = False,
    ) -> Union[str, Dict]:
        
        """Sends a prompt to Anthropic's Claude and returns the generated text."""
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        output = response.content
        response_text = output[0].text
        if output_format == "json":
            return LlmUtils.clean_up_json_text(response_text, use_json_assist)
        else:
            return response_text

    @staticmethod
    def send_to_openai(
        prompt: str,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        output_format: str = "json",
        use_json_assist: bool = False,
    ) -> Union[str, Dict]:
        """Sends a prompt to OpenAI and returns the generated text."""
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        client = OpenAI(api_key=openai_api_key)
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
            return LlmUtils.clean_up_json_text(response.choices[0].message.content, use_json_assist)
        else:
            return response.choices[0].message.content.text

    @staticmethod
    def clean_up_json_text(text: Union[str, Dict[str, Any]], use_json_assist: bool = False) -> Dict[str, Any]:
        if isinstance(text, dict):
            return text

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
            if use_json_assist:
                return LlmUtils._json_assist(text)
            else:
                {"error": f"Failed to parse JSON: {str(e)}", "original_response": text}
                raise  # Re-raise the ValueError to be handled up the stack

    @staticmethod
    def _json_assist(invalid_json: str) -> Dict[str, Any]:
        print("Attempting JSON assist...")
        prompt_tempate = "The json above is not a valid json. Format it as valid json. Respond with the complete valid json."
        formatted_prompt = "\n\n".join([invalid_json, prompt_tempate])
        
        response = LlmUtils.send_to_openai(
            prompt=formatted_prompt,
            model_name="gpt-4o-mini",
            temperature=0.0,
            max_tokens=16383,
            output_format="json"
        )
        
        return response


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
            self.formatted_prompt = self.formatted_prompt.replace(
                f"${key}$", str(value)
            )
        return self.formatted_prompt
