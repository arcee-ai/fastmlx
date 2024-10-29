"""
Script to automatically update the CHANGELOG.md file based on GitHub releases.

This script fetches release information from the GitHub API and updates
the CHANGELOG.md file with the latest release notes.

Usage:
    python update_changelog.py

Requirements:
    - requests
    - python-dotenv

Make sure to set up a .env file with your GitHub token:
    GITHUB_TOKEN=your_token_here
"""

import os
import re
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub repository details
REPO_OWNER = "Blaizzy"
REPO_NAME = "fastmlx"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# File paths
CHANGELOG_PATH = "docs/changelog.md"


def get_releases():
    """Fetch all releases information from GitHub API."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def parse_version(version_string):
    """Parse version string to tuple, handling 'v' prefix."""
    return tuple(map(int, version_string.lstrip("v").split(".")))


def compare_versions(v1, v2):
    """Compare two version tuples."""
    return (v1 > v2) - (v1 < v2)


def create_issue_link(issue_number):
    """Create a clickable link for an issue."""
    return f"[#{issue_number}](https://github.com/{REPO_OWNER}/{REPO_NAME}/issues/{issue_number})"


def create_contributor_link(username):
    """Create a clickable link for a contributor."""
    return f"[@{username}](https://github.com/{username})"


def format_release_notes(body):
    """Format the release notes in a cleaner structure."""
    formatted_notes = []
    current_section = None

    for line in body.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith("##"):
            current_section = line.lstrip("#").strip()
            formatted_notes.append(f"\n**{current_section}**\n")
        elif line.startswith("**Full Changelog**"):
            # Skip the full changelog link
            continue
        elif line.startswith("*"):
            if line.startswith("* @"):
                # Handle new contributors
                match = re.match(
                    r"\* @(\w+) made their first contribution in (https://.*)", line
                )
                if match:
                    username, url = match.groups()
                    formatted_notes.append(
                        f"- {create_contributor_link(username)} made their first contribution in [{url}]({url})"
                    )
            else:
                # Clean up bullet points
                cleaned_line = re.sub(
                    r"by @(\w+) in (https://.*)",
                    lambda m: f"by {create_contributor_link(m.group(1))}",
                    line,
                )
                cleaned_line = cleaned_line.replace("* ", "- ")
                formatted_notes.append(cleaned_line)
        else:
            formatted_notes.append(line)

    # Replace issue numbers with clickable links
    formatted_notes = [
        re.sub(r"#(\d+)", lambda m: create_issue_link(m.group(1)), item)
        for item in formatted_notes
    ]

    return "\n".join(formatted_notes)


def update_changelog(releases):
    """Update the CHANGELOG.md file with the release information."""
    with open(CHANGELOG_PATH, "r") as file:
        content = file.read()

    # Extract existing versions from changelog
    existing_versions = re.findall(r"##\s*\[?v?([\d.]+)", content)
    existing_versions = [parse_version(v) for v in existing_versions]

    # Sort releases by version (newest first)
    releases.sort(key=lambda r: parse_version(r["tag_name"]), reverse=True)

    new_content = ""
    added_versions = []

    for release in releases:
        version = parse_version(release["tag_name"])

        # Skip if this version is already in the changelog
        if version in existing_versions:
            continue

        print(f"Adding new version: {'.'.join(map(str, version))}")  # Debug print

        release_date = datetime.strptime(
            release["published_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).strftime("%d %B %Y")
        new_content += f"## [{release['tag_name']}] - {release_date}\n\n"
        new_content += format_release_notes(release["body"])
        new_content += "\n\n"
        added_versions.append(version)

    if new_content:
        # Find the position to insert new content (after the header and introduction)
        header_end = content.find("\n\n", content.find("# Changelog"))
        if header_end == -1:
            header_end = content.find("\n", content.find("# Changelog"))

        if header_end != -1:
            updated_content = (
                content[: header_end + 2] + new_content + content[header_end + 2 :]
            )
        else:
            # If we can't find the proper position, just prepend the new content
            updated_content = "# Changelog\n\n" + new_content + content

        # Write updated changelog
        with open(CHANGELOG_PATH, "w") as file:
            file.write(updated_content)
    else:
        print("No new versions to add")


def main():
    try:
        releases = get_releases()
        update_changelog(releases)
        print("Changelog updated and formatted successfully")
    except requests.RequestException as e:
        print(f"Error fetching release information: {e}")
    except IOError as e:
        print(f"Error updating changelog file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
