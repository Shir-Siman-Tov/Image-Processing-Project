"""Get figures/results out of a throwaway Colab clone and onto GitHub.

Each notebook's setup cell does a fresh `git clone`/`git pull` into Colab's
ephemeral /content filesystem (see notebooks/01_dataset_visualization.ipynb
cell 1). Anything save_figure()/save_results_csv() write there is lost when
the runtime disconnects unless it's copied out before that happens. Both
functions here are no-ops outside Colab.

sync_outputs_to_drive() is a plain backup - no auth needed, since Drive is
already mounted by each notebook's setup cell.

git_commit_and_push() closes the loop by pushing straight back to GitHub, so
the README's figures/results actually update without a manual download/
git-add step. It needs a GitHub personal access token (repo write scope)
stored once in Colab's Secrets manager (key icon, left sidebar) under the
name GITHUB_TOKEN, with notebook access enabled - see the README's "Running
on Colab" section. The token is read via google.colab.userdata (never
printed, never written to a file) and passed to `git push` only as a one-off
target URL, so it never ends up stored in the clone's `origin` remote.
"""
import shutil
import subprocess

from ipproj import config

DRIVE_OUTPUTS_ROOT = "/content/drive/MyDrive/ipproj_outputs"
REPO_URL = "github.com/Shir-Siman-Tov/Image-Processing-Project.git"


def sync_outputs_to_drive() -> None:
    """Copy figures/ and results/ into Drive as a durable backup (Colab only)."""
    if not config.IS_COLAB:
        return
    for src in (config.FIGURES_ROOT, config.REPORTS_ROOT):
        if not src.exists():
            continue
        dest = f"{DRIVE_OUTPUTS_ROOT}/{src.name}"
        shutil.copytree(src, dest, dirs_exist_ok=True)


def git_commit_and_push(message: str, token_secret_name: str = "GITHUB_TOKEN") -> None:
    """Commit figures/ and results/ and push straight to GitHub (Colab only)."""
    if not config.IS_COLAB:
        return
    from google.colab import userdata

    token = userdata.get(token_secret_name)
    repo_dir = str(config.REPO_ROOT)

    subprocess.run(
        ["git", "-C", repo_dir, "config", "user.email", "colab-sync@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "-C", repo_dir, "config", "user.name", "Colab Auto-sync"], check=True)
    subprocess.run(["git", "-C", repo_dir, "add", "figures", "results"], check=True)

    commit = subprocess.run(
        ["git", "-C", repo_dir, "commit", "-m", message], capture_output=True, text=True
    )
    if commit.returncode != 0:
        if "nothing to commit" in commit.stdout:
            return
        raise RuntimeError(f"git commit failed:\n{commit.stdout}\n{commit.stderr}")

    remote_url = f"https://{token}@{REPO_URL}"
    push = subprocess.run(
        ["git", "-C", repo_dir, "push", remote_url, "HEAD:main"], capture_output=True, text=True
    )
    if push.returncode != 0:
        raise RuntimeError(
            "git push failed:\n" + push.stderr.replace(token, "***").replace(remote_url, "***")
        )
