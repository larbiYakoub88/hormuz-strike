#!/usr/bin/env python3
"""
Téléchargeur de paires de devises depuis une API REST.

Usage:
    python download_forex_pairs.py EURUSD GBPUSD USDJPY
    python download_forex_pairs.py EURUSD GBPUSD --timeframe 60 --count 50000 --output-dir ./data
"""

import argparse
import json
import os
import sys
from typing import List

import requests


def download_pair(
    pair: str,
    timeframe: int = 30,
    count: int = 99999,
    output_dir: str = ".",
) -> bool:
    """
    Télécharge les données d'une paire depuis l'API et les sauvegarde dans un fichier JSON.

    Args:
        pair: Nom de la paire (ex: EURUSD)
        timeframe: Période (valeur dans l'URL, exemple 30)
        count: Nombre de points de données (paramètre count)
        output_dir: Dossier de destination (créé s'il n'existe pas)

    Returns:
        True si succès, False sinon
    """
    url = f"http://localhost:8080/api/rates/{pair}/{timeframe}?count={count}"
    output_path = os.path.join(output_dir, f"{pair}.json")

    try:
        print(f"Téléchargement de {pair} depuis {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Lève une exception pour les codes 4xx/5xx

        # Sauvegarde du contenu JSON (formaté pour lisibilité)
        data = response.json()
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Sauvegardé dans {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Erreur réseau pour {pair} : {e}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"  ✗ Réponse invalide (non-JSON) pour {pair} : {e}", file=sys.stderr)
    except OSError as e:
        print(f"  ✗ Erreur fichier pour {pair} : {e}", file=sys.stderr)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge les données de paires de devises depuis une API REST."
    )
    parser.add_argument(
        "pairs",
        nargs="+",
        help="Liste des paires à télécharger (ex: EURUSD GBPUSD)",
    )
    parser.add_argument(
        "--timeframe",
        type=int,
        default=30,
        help="Période dans l'URL (défaut: 30)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=99999,
        help="Nombre de points de données (défaut: 99999)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Dossier de destination (défaut: répertoire courant)",
    )

    args = parser.parse_args()

    success_count = 0
    for pair in args.pairs:
        if download_pair(
            pair=pair,
            timeframe=args.timeframe,
            count=args.count,
            output_dir=args.output_dir,
        ):
            success_count += 1

    print(f"\nRésumé : {success_count}/{len(args.pairs)} paires téléchargées avec succès.")


if __name__ == "__main__":
    main()