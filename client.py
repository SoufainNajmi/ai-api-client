import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class AIClient:
    """Petit client HTTP pour une API d'IA compatible OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("AI_API_BASE_URL")
            or os.getenv("AI_API_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self.model = model or os.getenv("AI_MODEL") or "openai/gpt-oss-20b"
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
            })

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}{path}"

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if choices:
            first_choice = choices[0]
            message = first_choice.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                texts.append(str(text))
                    if texts:
                        return "\n".join(texts)
                if content:
                    return str(content)

            text = first_choice.get("text")
            if text:
                return str(text)

        if "output" in data:
            output = data["output"]
            if isinstance(output, list) and output:
                first_output = output[0]
                if isinstance(first_output, dict):
                    if "content" in first_output:
                        return str(first_output["content"])
                    if "text" in first_output:
                        return str(first_output["text"])

        raise ValueError(f"Réponse inattendue de l'API : {data}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "Aucune clé API trouvée. Ajoute AI_API_KEY ou OPENAI_API_KEY dans ton .env ou passe --api-key."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [],
            "temperature": temperature,
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})

        payload["messages"].append({"role": "user", "content": prompt})

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = self.session.post(
            self._url("/chat/completions"),
            json=payload,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 401:
                raise ValueError(
                    "401 Unauthorized: la clé API est invalide, expirée ou la base URL est incorrecte. "
                    "Vérifie AI_API_KEY / OPENAI_API_KEY et AI_API_BASE_URL / AI_API_URL dans ton .env."
                ) from exc
            raise ValueError(f"Erreur HTTP {response.status_code}: {response.text}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("La réponse de l'API n'est pas un JSON valide.") from exc

        return self._extract_text(data)
