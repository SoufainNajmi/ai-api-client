import argparse
import sys

from client import AIClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Client CLI pour envoyer une requête à une API d'IA.",
    )
    parser.add_argument("prompt", nargs="*", help="Le texte ou la question à envoyer à l'IA.")
    parser.add_argument("--api-key", dest="api_key", help="Clé API (ou utilise AI_API_KEY / OPENAI_API_KEY).")
    parser.add_argument("--base-url", dest="base_url", help="URL de base de l'API, ex. https://api.groq.com/openai/v1")
    parser.add_argument("--model", dest="model", help="Nom du modèle à utiliser, ex. gpt-4o-mini")
    parser.add_argument("--system", dest="system_prompt", help="Instruction système pour guider la réponse.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Température de génération (0.0 à 1.0).")
    parser.add_argument("--max-tokens", type=int, help="Nombre maximum de tokens à générer.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    text = " ".join(args.prompt).strip()
    if not text:
        parser.print_help()
        return 1

    client = AIClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    try:
        response = client.generate(
            prompt=text,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:  # pragma: no cover - sortie CLI
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
