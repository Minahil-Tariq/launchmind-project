import os
import json
import requests
import base64
from datetime import datetime
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from message_bus import receive_messages, send_message, create_message
from utils.logger import log_step, log_error, log_message_flow, execute_with_retry

# ─── GitHub Config ───────────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("GITHUB_REPO")

def _gh_headers():
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }


class EngineerAgent:
    """
    Engineer Agent — Assignment Role:
      1. Receives product spec from Product agent
      2. Uses LLM to generate a complete HTML landing page
      3. Creates GitHub branch, commits HTML, opens a GitHub Issue, opens a PR
      4. Sends PR URL + Issue URL back to CEO as a structured result message
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "fallback_key")

        self.agent = Agent(
            role="Software Engineer",
            goal="Build a complete, responsive HTML landing page for a startup",
            backstory=(
                "You are a senior front-end engineer who specialises in building "
                "beautiful, conversion-optimised landing pages from product specifications. "
                "You write clean, semantic HTML with embedded CSS."
            ),
            verbose=os.getenv("DEBUG", "False") == "True",
            llm="groq/llama-3.1-8b-instant"
        )

        self.state = {
            "last_idea": None,
            "last_spec": None,
            "last_focus": None
        }

    # ─── Message Loop ────────────────────────────────────────────────────────

    def process_messages(self):
        """Poll the 'engineer' queue and handle incoming messages."""
        messages = receive_messages("engineer")
        if not messages:
            return

        for msg in messages:
            try:
                msg_type = msg.get("message_type")
                log_step("EngineerAgent", "Process Message",
                         f"Received '{msg_type}' from '{msg['from_agent']}'")

                if msg_type == "task":
                    self.state["last_idea"] = msg["payload"].get("idea", "Unknown Startup")
                    self.state["last_spec"] = msg["payload"].get("spec", {})
                    self.state["last_focus"] = msg["payload"].get("focus", "Build the landing page")
                    self._run(parent_msg_id=msg["message_id"])

                elif msg_type == "revision_request":
                    feedback = msg["payload"].get("feedback", "Please improve the landing page.")
                    log_step("EngineerAgent", "Handling Revision", f"Feedback: {feedback}")
                    self.state["last_focus"] = (
                        f"REVISION FEEDBACK TO APPLY: {feedback} | "
                        f"Original Task: Build landing page for {self.state['last_idea']}"
                    )
                    self._run(parent_msg_id=msg["message_id"])

            except Exception as e:
                log_error("EngineerAgent", "Message Processing Failure", e,
                          details=f"Message ID: {msg.get('message_id')}")

    # ─── Core Pipeline ───────────────────────────────────────────────────────

    def _run(self, parent_msg_id):
        """Full pipeline: LLM → HTML → GitHub branch/commit/issue/PR → notify CEO."""
        if not GITHUB_TOKEN or not REPO_NAME:
            log_error("EngineerAgent", "Missing Credentials",
                      Exception("GITHUB_TOKEN or GITHUB_REPO not set in .env"))
            self._send_failure(parent_msg_id, "GitHub credentials not configured.")
            return

        # Step 1: LLM generates the HTML
        html_content = self._generate_html_with_llm()
        if not html_content:
            log_error("EngineerAgent", "Pipeline Halted", Exception("LLM failed to output parseable HTML."))
            self._send_failure(parent_msg_id, "HTML Generation Failed. LLM output unparseable or totally missing requested markup. Resubmit task.")
            return

        # Step 2: Save locally
        try:
            with open("landing_page.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            log_step("EngineerAgent", "HTML Saved Locally", "landing_page.html written.")
        except Exception as e:
            log_error("EngineerAgent", "Local Save Failed", e)

        # Step 3: GitHub flow
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"agent-landing-page-{timestamp}"

            self._create_branch(branch_name)
            self._commit_file(branch_name, "index.html", html_content,
                              "Add landing page generated by Engineer Agent <agent@launchmind.ai>")
            issue_url = self._create_issue(
                title="Initial landing page",
                body=(
                    f"The Engineer Agent has generated the first version of the landing page "
                    f"for **{self.state['last_idea']}**.\n\n"
                    "This issue tracks the initial build and review of the landing page."
                )
            )
            pr_url = self._open_pr(
                branch_name=branch_name,
                title="Initial landing page",
                body=(
                    f"## 🎓 {self.state['last_idea']} — Landing Page\n\n"
                    "This PR introduces the landing page generated by the Engineer Agent "
                    "based on the Product specification.\n\n"
                    "### Features included\n"
                    "- Modern, responsive design\n"
                    "- Hero section with value proposition\n"
                    "- Features grid\n"
                    "- How-it-works section\n"
                    "- Email sign-up form\n\n"
                    "---\n*Automatically generated by LaunchMind Engineer Agent 🤖*"
                )
            )
        except Exception as e:
            log_error("EngineerAgent", "GitHub Flow Failure", e)
            self._send_failure(parent_msg_id, str(e))
            return

        # Step 4: Send result back to CEO
        result_payload = {
            "status": "success",
            "pr_url": pr_url,
            "issue_url": issue_url,
            "branch": branch_name,
            "html_path": "index.html",
            "html_preview": html_content[:300] + "..."
        }

        reply = create_message(
            from_agent="engineer",
            to_agent="ceo",
            message_type="result",
            payload=result_payload,
            parent_message_id=parent_msg_id
        )
        send_message(reply)
        log_message_flow(reply["message_id"], "engineer", "ceo", "result")
        log_step("EngineerAgent", "Result Sent to CEO",
                 payload={"pr_url": pr_url, "issue_url": issue_url})

        print(f"\n{'='*50}")
        print(f"✅ Engineer Agent Complete!")
        print(f"   📎 PR: {pr_url}")
        print(f"   🐛 Issue: {issue_url}")
        print(f"   🌿 Branch: {branch_name}")
        print(f"{'='*50}\n")

    # ─── LLM: Generate HTML ──────────────────────────────────────────────────

    def _generate_html_with_llm(self):
        """Use the LLM to generate a complete HTML landing page from the product spec."""
        spec = self.state.get("last_spec", {})
        idea = self.state.get("last_idea", "Smart Startup")

        value_prop = spec.get("value_proposition", idea)
        features = spec.get("features", [])
        features_text = "\n".join(
            [f"  - {f.get('name','')}: {f.get('description','')}" for f in features[:5]]
        )
        personas = spec.get("personas", [])
        personas_text = "\n".join(
            [f"  - {p.get('name','')}: {p.get('pain_point','')}" for p in personas]
        )

        prompt = f"""
You are an elite, Silicon-Valley UI/UX engineer. Generate a COMPLETE, ultra-premium production-ready HTML landing page for the following startup. Return ONLY raw HTML — no markdown, no code fences, no explanation.

Startup Idea: {idea}
Value Proposition: {value_prop}

Key Features:
{features_text}

Target Users / Pain Points:
{personas_text}

The HTML must include:
1. <head> with SEO meta tags, charset, viewport, and all CSS embedded in a <style> block.
2. Import modern Google Fonts (e.g. 'Inter' or 'Outfit').
3. A stunning Hero section with: product name as <h1>, value proposition as subheadline, and an interactive CTA button.
4. A Features section using a clean CSS Grid layout for cards.
5. A "How It Works" section with 3 visual steps.
6. An email sign-up form that uses Vanilla JavaScript to simulate a "Loading -> Success" state when clicked, proving dynamic implementation logic.
7. A modern footer.

Design & Aesthetics (MANDATORY):
- Achieve an ultra-premium look using Glassmorphism (backdrop-filter: blur) on cards and floating nav elements.
- Use a highly curated dark-mode color palette (e.g. deeply saturated dark indigos/blacks with vibrant neon accents or smooth multi-stop gradients).
- Embed smooth micro-animations: cards must lift and glow on hover (box-shadow transitions), buttons must pulse or shift beautifully.
- Fully responsive (mobile-friendly with @media queries).

CRITICAL INSTRUCTION: Return nothing else except pure HTML. Do not say "Here is your code". Output MUST start exactly with "<!DOCTYPE html>" and end exactly with "</html>".
"""

        task = Task(
            description=prompt,
            agent=self.agent,
            expected_output="A complete HTML document starting with <!DOCTYPE html>"
        )
        crew = Crew(agents=[self.agent], tasks=[task])

        log_step("EngineerAgent", "LLM Generate HTML", f"Calling LLM to generate landing page for: {idea}")
        try:
            result = execute_with_retry(crew, "EngineerAgent", "HTML Generation")
            raw = result.raw.strip() if hasattr(result, "raw") else str(result).strip()

            import re
            match = re.search(r'<!DOCTYPE html>[\s\S]*</html>', raw, re.IGNORECASE)
            if match:
                raw = match.group(0).strip()
            
            if "<!DOCTYPE" in raw and "<html" in raw:
                log_step("EngineerAgent", "HTML Generation Success",
                         f"Generated {len(raw)} characters of HTML.")
                return raw
            else:
                raise ValueError("LLM did not return complete valid HTML document enclosing <!DOCTYPE html> and </html> tags.")

        except Exception as e:
            raw_output = getattr(result, "raw", str(result)) if 'result' in locals() else "No Output"
            log_error("EngineerAgent", "LLM HTML Generation Failed", e, details=f"Raw text snippets: {raw_output[:250]}...")
            return None

    # ─── GitHub API Helpers ──────────────────────────────────────────────────

    def _get_default_branch_sha(self):
        url = f"https://api.github.com/repos/{REPO_NAME}/git/refs/heads/main"
        log_step("EngineerAgent", "Fetch Base SHA", f"GET {url}")
        resp = requests.get(url, headers=_gh_headers())
        if resp.status_code != 200:
            raise Exception(f"Failed to get base SHA: {resp.text}")
        return resp.json()["object"]["sha"]

    def _create_branch(self, branch_name):
        base_sha = self._get_default_branch_sha()
        url = f"https://api.github.com/repos/{REPO_NAME}/git/refs"
        payload = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}
        log_step("EngineerAgent", "Create Branch", f"Creating '{branch_name}'")
        resp = requests.post(url, headers=_gh_headers(), json=payload)
        if resp.status_code == 201:
            log_step("EngineerAgent", "Branch Created", branch_name)
        elif "already exists" in resp.text:
            log_step("EngineerAgent", "Branch Exists (OK)", branch_name)
        else:
            raise Exception(f"Failed to create branch: {resp.text}")

    def _commit_file(self, branch_name, file_path, content, commit_message):
        url = f"https://api.github.com/repos/{REPO_NAME}/contents/{file_path}"
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": commit_message,
            "content": encoded,
            "branch": branch_name,
            "author": {"name": "EngineerAgent", "email": "agent@launchmind.ai"}
        }
        # If file exists on this branch, include its SHA to update it
        check = requests.get(url, headers=_gh_headers(), params={"ref": branch_name})
        if check.status_code == 200:
            payload["sha"] = check.json().get("sha")

        log_step("EngineerAgent", "Commit File", f"Committing {file_path} to {branch_name}")
        resp = requests.put(url, headers=_gh_headers(), json=payload)
        if resp.status_code not in (200, 201):
            raise Exception(f"Failed to commit file: {resp.text}")
        log_step("EngineerAgent", "File Committed", f"{file_path} committed successfully.")

    def _create_issue(self, title, body):
        url = f"https://api.github.com/repos/{REPO_NAME}/issues"
        payload = {"title": title, "body": body}
        log_step("EngineerAgent", "Create Issue", title)
        resp = requests.post(url, headers=_gh_headers(), json=payload)
        if resp.status_code != 201:
            raise Exception(f"Failed to create issue: {resp.text}")
        issue_url = resp.json()["html_url"]
        log_step("EngineerAgent", "Issue Created", issue_url)
        print(f"   🐛 GitHub Issue: {issue_url}")
        return issue_url

    def _open_pr(self, branch_name, title, body):
        url = f"https://api.github.com/repos/{REPO_NAME}/pulls"
        payload = {"title": title, "body": body, "head": branch_name, "base": "main"}
        log_step("EngineerAgent", "Open PR", f"PR from {branch_name}")
        resp = requests.post(url, headers=_gh_headers(), json=payload)
        if resp.status_code == 201:
            pr_url = resp.json()["html_url"]
            log_step("EngineerAgent", "PR Opened", pr_url)
            print(f"   🔀 GitHub PR: {pr_url}")
            return pr_url
        elif resp.status_code == 422:
            # PR may already exist — try to find it
            list_resp = requests.get(url, headers=_gh_headers(),
                                     params={"head": f"{REPO_NAME.split('/')[0]}:{branch_name}"})
            if list_resp.status_code == 200 and list_resp.json():
                pr_url = list_resp.json()[0]["html_url"]
                log_step("EngineerAgent", "Existing PR Found", pr_url)
                return pr_url
        raise Exception(f"Failed to open PR: {resp.text}")

    def _send_failure(self, parent_msg_id, reason):
        """Notify CEO of failure so it can handle it."""
        reply = create_message(
            from_agent="engineer",
            to_agent="ceo",
            message_type="result",
            payload={"status": "failed", "error": reason},
            parent_message_id=parent_msg_id
        )
        send_message(reply)
        log_message_flow(reply["message_id"], "engineer", "ceo", "result (failure)")
        log_error("EngineerAgent", "Pipeline Failure Reported to CEO", Exception(reason))
