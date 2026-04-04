"""Update the authors list in CITATION.cff from GitHub contributor data.

Note for Developers
-------------------
Set the GITHUB_TOKEN environment variable to your GitHub personal access token via
`export GITHUB_TOKEN=your_token_here` to increase API rate limits.
"""

# NOTE: Unhandled edge cases (not pressing, but likely to encounter):
# - Existing contributor updates name on GitHub profile (will duplicate person)
# - Existing contributor adds ORCID to GitHub profile (will duplicate person)
# - New contributor has same name as existing (will merge with existing, but may be
#   different person)
# - Contributor previously only had GitHub username, but later adds name fields (again
#   will duplicate entries, one with GitHub username as given name, one with real name)

import click
import warnings

from pathlib import Path
from dataclasses import dataclass
import json
import re
import urllib.error
import urllib.request
import yaml

GITHUB_API = "https://api.github.com"
ORG = "teamtomo"
REPO_ROOT = Path(__file__).parent.parent
CITATION_FILE = REPO_ROOT / "CITATION.cff"

# Exclude bot accounts from contributor list
BOT_LOGINS = {"actions-user", "dependabot", "dependabot[bot]", "github-actions[bot]"}

# Limit number of entries to fetch per API call
PER_PAGE_MAX = 100

# Defaults for CITATION.cff metadata fields
CFF_VERSION = "1.2.0"
TITLE = "teamtomo: modular Python packages for cryo-EM and cryo-ET"
VERSION = "0.5.0"
ABSTRACT = ""
MESSAGE = "If you use this software, please cite it as below."
LICENSE = "BSD-3-Clause"
TYPE = "software"
# REPOSITORY_CODE = "https://github.com/teamtomo/teamtomo"
# URL = "https://teamtomo.org"
# DOI = "10.5281/zenodo.18405652"
KEYWORDS = ["cryo-EM", "cryo-ET", "python"]


@dataclass(frozen=True)
class Author:
    """A contributor entry with optional metadata to place into CITATION.cff.

    Note
    ----
    Zenodo currently only supports CITATION.cff author fields for
    [given-names, family-names, affiliation, orcid] where one of 'given-names' or
    'family-names' is required. We only include these fields for contributors.
    """

    given_names: str | None = None
    family_names: str | None = None
    affiliation: str | None = None
    orcid: str | None = None

    def __post_init__(self):
        """Validate that at least one of given_names or family_names is provided."""
        if not self.given_names and not self.family_names:
            raise ValueError(
                "At least one of 'given_names' or 'family_names' must be provided.\n"
                f"Author: {self}"
            )


@dataclass
class CitationCFF:
    """Contents of the CITATION.cff file, including metadata and a list of authors."""

    cff_version: str
    title: str
    version: str
    license: str
    type: str
    abstract: str
    message: str
    authors: list[Author]
    keywords: list[str]

    def to_dict(self) -> dict:
        """Convert to ordered dictionary for YAML serialization."""
        return {
            "cff-version": self.cff_version,
            "title": self.title,
            "license": self.license,
            "version": self.version,
            "message": self.message,
            "type": self.type,
            "abstract": self.abstract,
            "authors": [
                {
                    k: v
                    for k, v in {
                        "given-names": author.given_names,
                        "family-names": author.family_names,
                        "affiliation": author.affiliation,
                        "orcid": author.orcid,
                    }.items()
                    if v is not None
                }
                for author in self.authors
            ],
            "keywords": self.keywords,
        }

    def render_yaml(self) -> str:
        """Render the CitationCFF data as a YAML string."""
        return yaml.safe_dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def github_get(path: str, token: str | None = None) -> list | dict:
    """Make an authenticated request to the GitHub API.

    Parameters
    ----------
    path : str
        The API endpoint path, e.g. '/orgs/{ORG}/repos'.
    token : str, optional
        A GitHub personal access token to increase rate limits. If not provided,
        the request will be unauthenticated and subject to lower rate limits.

    Returns
    -------
    list or dict
        The parsed JSON response from the GitHub API.
    """
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


def parse_existing_citation() -> CitationCFF | None:
    """Parse the existing CITATION.cff file at the repo root, if it exists."""
    if not CITATION_FILE.exists():
        return None

    content = CITATION_FILE.read_text()
    data = yaml.safe_load(content)

    authors = []
    for author_data in data.get("authors", []):
        authors.append(
            Author(
                given_names=author_data.get("given-names"),
                family_names=author_data.get("family-names"),
                affiliation=author_data.get("affiliation"),
                orcid=author_data.get("orcid"),
            )
        )

    return CitationCFF(
        cff_version=data.get("cff-version", ""),
        title=data.get("title", ""),
        version=data.get("version", ""),
        license=data.get("license", ""),
        type=data.get("type", ""),
        abstract=data.get("abstract", ""),
        message=data.get("message", ""),
        authors=authors,
        keywords=data.get("keywords", []),
    )


def get_all_authors(token: str | None = None) -> list[Author]:
    """Fetch the list of contributors from GitHub API and convert to Author objects."""
    repos = github_get(f"/orgs/{ORG}/repos?per_page={PER_PAGE_MAX}&type=public", token)
    logins: set[str] = set()

    for repo in repos:
        repo_logins = get_repo_logins(repo["name"], token)
        logins.update(repo_logins)

    return [parse_login_to_author(login, token) for login in logins]


def get_repo_logins(repo_name: str, token: str | None = None) -> list[str]:
    """Fetch list of contributors for a specific repo as list of logins"""
    contributors = github_get(
        f"/repos/{ORG}/{repo_name}/contributors?per_page={PER_PAGE_MAX}", token
    )
    logins = []

    for contributor in contributors:
        login = contributor["login"]
        type_ = contributor.get("type", "")
        if login in BOT_LOGINS or type_ == "Bot":
            continue

        logins.append(login)

    return logins


def parse_login_to_author(login: str, token: str | None = None) -> Author:
    """Convert a GitHub login to an Author object with placeholder metadata.

    Note
    ----
    This is a simplistic approach which depends on a GitHub profile having a maintained
    name field and/or company field.
    """
    login_info = github_get(f"/users/{login}", token)
    name = login_info.get("name", "")
    company = login_info.get("company", None)
    company = company.strip() if company else None

    # Parsing for ORCID with separate API call
    orcid = parse_login_to_orcid(login, token)

    # Split name into given and family names
    # NOTE: This is a simplistic approach which does not broadly cover all conventions.
    given_names = None
    family_names = None
    name = name.strip() if name else None
    if not name:
        warnings.warn(f"User '{login}' GitHub profile name, using login instead.")
        given_names = login
    else:
        parts = name.split()
        if len(parts) == 1:
            given_names = parts[0]
        elif len(parts) > 1:
            given_names = " ".join(parts[:-1])
            family_names = parts[-1]

    return Author(
        given_names=given_names,
        family_names=family_names,
        affiliation=company,
        orcid=orcid,
    )


def parse_login_to_orcid(login: str, token: str | None = None) -> str | None:
    """Attempt an ORCID resolution from GitHub profile.

    Parameters
    ----------
    login : str
        The GitHub username.
    token : str, optional
        A GitHub personal access token to increase rate limits.

    Returns
    -------
    str | None
        The ORCID if found, otherwise None.
    """
    # 1. Check social accounts for ORCID URL
    accounts_info = github_get(f"/users/{login}/social_accounts", token)
    for account in accounts_info:
        url = account.get("url", "")
        if "orcid.org" in url:
            return url

    # 2. Try scanning profile page for ORCID URL as fallback
    try:
        req = urllib.request.Request(f"https://github.com/{login}")
        req.add_header("User-Agent", "teamtomo-citation-updater")
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode()
        match = re.search(r'href="(https://orcid\.org/[\w-]+)"', html)
        if match:
            return match.group(1)
    except urllib.error.URLError:
        pass
    return None


def construct_citation(token: str | None = None) -> CitationCFF:
    """Construct a CitationCFF object with the latest contributors from GitHub."""
    authors = get_all_authors(token)
    authors = sorted(authors, key=lambda a: (a.family_names or "", a.given_names or ""))

    return CitationCFF(
        cff_version=CFF_VERSION,
        title=TITLE,
        license=LICENSE,
        version=VERSION,
        type=TYPE,
        abstract=ABSTRACT,
        message=MESSAGE,
        authors=authors,
        keywords=KEYWORDS,
    )


def update_current_citation_authors(
    old_citation: CitationCFF, new_citation: CitationCFF
) -> CitationCFF:
    """Merge new contributors with existing authors, preserving old metadata.

    Matches authors by ORCID first, then by name. If a match is found in the
    old citation, uses the old entry (preserves any manual edits). Otherwise,
    adds the new author.
    """
    old_authors_by_id = {
        author.orcid: author for author in old_citation.authors if author.orcid
    }
    old_authors_by_name = {
        (author.given_names, author.family_names): author
        for author in old_citation.authors
    }
    new_authors_set = set(new_citation.authors)

    merged_authors = []
    for new_author in new_authors_set:
        if new_author.orcid and new_author.orcid in old_authors_by_id:
            merged_authors.append(old_authors_by_id[new_author.orcid])
        elif (new_author.given_names, new_author.family_names) in old_authors_by_name:
            merged_authors.append(
                old_authors_by_name[(new_author.given_names, new_author.family_names)]
            )
        else:
            # New author not in old list
            merged_authors.append(new_author)

    # Sort alphabetically by family name, then given name
    merged_authors.sort(key=lambda a: (a.family_names or "", a.given_names or ""))

    return CitationCFF(
        cff_version=old_citation.cff_version,
        title=old_citation.title,
        license=old_citation.license,
        version=old_citation.version,
        type=old_citation.type,
        abstract=old_citation.abstract,
        message=old_citation.message,
        authors=merged_authors,
        keywords=old_citation.keywords,
    )


@click.command()
@click.option("--token", envvar="GITHUB_TOKEN", help="Optional GitHub token.")
@click.option("--rewrite", is_flag=True, help="Construct entirely new citation.")
def cli(token: str | None, rewrite: bool):
    """Command-line interface to update CITATION.cff with latest contributors."""
    old_citation = parse_existing_citation()
    new_citation = construct_citation(token)

    if rewrite or not CITATION_FILE.exists():
        final_citation = new_citation
    else:
        final_citation = update_current_citation_authors(old_citation, new_citation)

    CITATION_FILE.write_text(final_citation.render_yaml())
    print(f"Updated {CITATION_FILE} with {len(final_citation.authors)} authors.")


if __name__ == "__main__":
    cli()
