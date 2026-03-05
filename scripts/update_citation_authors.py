#!/usr/bin/env python3
"""Update the authors list in CITATION.cff from GitHub contributor data.

Fetches contributors from all public repos in the teamtomo org, enriches each
entry with name, affiliation, and ORCID (from GitHub social accounts), sorts
alphabetically by family name (alias-only contributors go to the bottom),
then rewrites the authors section of CITATION.cff.

Usage:
    python scripts/update_citation_authors.py

Set the GITHUB_TOKEN environment variable to avoid rate limiting.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_API = "https://api.github.com"
ORG = "teamtomo"
REPO_ROOT = Path(__file__).parent.parent
CITATION_FILE = REPO_ROOT / "CITATION.cff"

BOT_LOGINS = {"actions-user", "dependabot", "dependabot[bot]", "github-actions[bot]"}


def github_get(path: str, token: str | None = None) -> list | dict:
    url = f"{GITHUB_API}{path}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "teamtomo-citation-updater")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode()
        return json.loads(body) if body.strip() else []


def get_contributors(token: str | None = None) -> set[str]:
    """Return unique contributor logins across all public repos in the org."""
    repos = github_get(f"/orgs/{ORG}/repos?per_page=100&type=public", token)
    logins: set[str] = set()
    for repo in repos:
        try:
            contribs = github_get(
                f"/repos/{ORG}/{repo['name']}/contributors?per_page=100&anon=false",
                token,
            )
            if isinstance(contribs, list):
                for c in contribs:
                    if c.get("type") == "User" and c["login"] not in BOT_LOGINS:
                        logins.add(c["login"])
        except (urllib.error.HTTPError, ValueError) as e:
            print(f"  Warning: could not fetch contributors for {repo['name']}: {e}")
    return logins


def get_orcid(login: str, token: str | None = None) -> str | None:
    """Return the ORCID URL from a user's GitHub social accounts, if present."""
    accounts = github_get(f"/users/{login}/social_accounts", token)
    for account in accounts:
        url = account.get("url", "")
        if "orcid.org" in url:
            return url
    return None


def split_name(full_name: str) -> tuple[str | None, str | None]:
    """Split a full name into (given_names, family_name) on the last space."""
    parts = full_name.strip().split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, parts[0]
    return " ".join(parts[:-1]), parts[-1]


def format_author_entry(info: dict, orcid: str | None) -> str:
    """Return a CITATION.cff author YAML block."""
    login = info["login"]
    full_name = (info.get("name") or "").strip()
    company = (info.get("company") or "").strip().lstrip("@").strip() or None

    given, family = split_name(full_name)

    fields: list[tuple[str, str]] = []
    if given:
        fields.append(("given-names", given))
    if family:
        fields.append(("family-names", family))
    fields.append(("alias", login))
    if company:
        fields.append(("affiliation", company))
    if orcid:
        fields.append(("orcid", f'"{orcid}"'))

    lines = [f"  - {fields[0][0]}: {fields[0][1]}"]
    for key, value in fields[1:]:
        lines.append(f"    {key}: {value}")
    return "\n".join(lines)


def update_citation_authors(author_blocks: list[str]) -> None:
    """Replace the authors section in CITATION.cff, preserving everything else."""
    content = CITATION_FILE.read_text()
    match = re.search(r"^authors:.*$", content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find 'authors:' section in CITATION.cff")
    preamble = content[: match.start()]
    CITATION_FILE.write_text(
        preamble + "authors:\n" + "\n".join(author_blocks) + "\n"
    )


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")

    print(f"Fetching contributors across all {ORG} repos...")
    logins = get_contributors(token)
    print(f"Found {len(logins)} contributors\n")

    authors: list[tuple[dict, str | None]] = []
    for login in sorted(logins):
        print(f"  Fetching {login}...")
        info = github_get(f"/users/{login}", token)
        orcid = get_orcid(login, token)
        authors.append((info, orcid))

    # Sort alphabetically by family name; alias-only entries go to the bottom
    def sort_key(entry: tuple[dict, str | None]) -> tuple[int, str]:
        info, _ = entry
        _, family = split_name((info.get("name") or "").strip())
        if family:
            return (0, family.lower())
        return (1, info["login"].lower())

    authors.sort(key=sort_key)
    author_blocks = [format_author_entry(info, orcid) for info, orcid in authors]

    update_citation_authors(author_blocks)
    print(f"\nUpdated {CITATION_FILE.name} with {len(author_blocks)} authors.")


if __name__ == "__main__":
    main()
