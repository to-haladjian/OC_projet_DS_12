"""Deploie l'API sur un Space Hugging Face, depuis le pipeline.

Le Space utilise le SDK Docker et reconstruit l'image a partir des fichiers pousses ici.
C'est le meme `Dockerfile` que celui du depot : il a ete ecrit pour cet hebergeur, avec un
utilisateur d'uid 1000 et un port pilote par la variable `PORT`.

Ce que recoit le Space est volontairement reduit : le `Dockerfile`, `api/`, `src/`,
`models/`, et la carte du Space. Ni donnees, ni notebooks, ni tests.

Deux variables d'environnement sont attendues :

* `HF_TOKEN` — jeton d'ecriture Hugging Face, fourni par les secrets du depot ;
* `HF_SPACE` — identifiant du Space, sous la forme `utilisateur/nom-du-space`.

Usage : `python deploy/pousse_espace.py`
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RACINE = Path(__file__).resolve().parents[1]

#: Ce qui part vers le Space. Tout le reste est volontairement laisse de cote.
A_POUSSER = ("Dockerfile", "api", "src", "models")

#: Carte du Space : son en-tete YAML declare le SDK Docker et le port a exposer.
CARTE = RACINE / "deploy" / "hf_space" / "README.md"


def assemble(destination: Path) -> None:
    """Copie dans `destination` les seuls fichiers dont le Space a besoin."""
    for nom in A_POUSSER:
        source = RACINE / nom
        if not source.exists():
            raise FileNotFoundError(f"{nom} est absent du depot")
        if source.is_dir():
            # __pycache__ et les .pyc n'ont rien a faire dans l'image.
            shutil.copytree(
                source, destination / nom,
                ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
            )
        else:
            shutil.copy2(source, destination / nom)
    shutil.copy2(CARTE, destination / "README.md")


def main() -> int:
    jeton = os.environ.get("HF_TOKEN")
    espace = os.environ.get("HF_SPACE")
    if not jeton or not espace:
        print("HF_TOKEN ou HF_SPACE absent : deploiement ignore.", file=sys.stderr)
        return 1

    api = HfApi(token=jeton)
    # `exist_ok` : le premier run cree le Space, les suivants le mettent a jour.
    api.create_repo(repo_id=espace, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as travail:
        destination = Path(travail)
        assemble(destination)
        poids = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
        print(f"{espace} : envoi de {poids / 1024**2:.1f} Mo")

        api.upload_folder(
            repo_id=espace,
            repo_type="space",
            folder_path=str(destination),
            # Sans cela, un fichier retire du depot resterait indefiniment sur le Space.
            delete_patterns="*",
            commit_message=f"Deploiement depuis {os.environ.get('GITHUB_SHA', 'local')[:7]}",
        )

    print(f"deploye : https://huggingface.co/spaces/{espace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
