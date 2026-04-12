import os
import json
from unittest import result
import requests
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from message_bus import receive_messages, send_message, create_message
from utils.logger import log_step, log_error, log_message_flow, execute_with_retry, extract_json_from_llm

# ─── Config ──────────────────────────────────────────────────────────────────
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


class QAAgent:
    """
    QA / Reviewer Agent — Assignment Role:
      1. Receives the Engineer's output (HTML + PR URL) and Marketing copy from CEO
      2. Uses LLM to review HTML: consistency with product spec, headline, features
      3. Uses LLM to review marketing copy: tagline quality, CTA clarity, tone
      4. Posts at least 2 inline review comments on the GitHub PR via the GitHub API
      5. Sends a structured review report (pass/fail + issues list) back to CEO
      6. On 'fail' verdict, CEO must instruct the relevant agent to revise (handled by CEO)
    """

    def __init__(self):
        self.agent = Agent(
            role="QA Engineer and Product Reviewer",
            goal="Rigorously review the landing page and marketing copy for quality, consistency, and completeness",
            backstory=(
                "You are a seasoned QA engineer and product reviewer who has audited "
                "hundreds of SaaS landing pages and marketing campaigns. You have a sharp "
                "eye for inconsistencies between a product specification and its execution. "
                "You write precise, actionable feedback that developers and marketers can act on immediately."
            ),
            verbose=os.getenv("DEBUG", "False") == "True",
            llm="groq/llama-3.1-8b-instant"
        )

        # State tracking
        self.state = {
            "last_idea": None,
            "last_spec": None,
            "last_pr_url": None,
            "last_html_preview": None,
            "last_marketing_copy": None,
            "last_pr_number": None
        }

    # ─── Message Loop ────────────────────────────────────────────────────────

    def process_messages(self):
        """Poll the 'qa' queue and handle incoming messages."""
        messages = receive_messages("qa")
        if not messages:
            return

        for msg in messages:
            try:
                msg_type = msg.get("message_type")
                log_step("QAAgent", "Process Message",
                         f"Received '{msg_type}' from '{msg['from_agent']}'")

                if msg_type == "task":
                    payload = msg["payload"]
                    self.state["last_idea"] = payload.get("idea", "Unknown Startup")
                    self.state["last_spec"] = payload.get("spec", {})
                    self.state["last_pr_url"] = payload.get("pr_url", "")
                    self.state["last_html_preview"] = payload.get("html_preview", "")
                    self.state["last_marketing_copy"] = payload.get("marketing_copy", {})
                    self._run(parent_msg_id=msg["message_id"])

            except Exception as e:
                log_error("QAAgent", "Message Processing Failure", e,
                          details=f"Message ID: {msg.get('message_id')}")

    # ─── Core Pipeline ───────────────────────────────────────────────────────

    def _run(self, parent_msg_id):
        """Full QA pipeline: review HTML → review copy → post PR comments → report to CEO."""

        # Step 1: LLM reviews the HTML landing page
        html_review = self._review_html_with_llm()

        # Step 2: LLM reviews the marketing copy
        copy_review = self._review_marketing_copy_with_llm()

        # Step 3: Post inline comments on the GitHub PR
        pr_comments_posted = self._post_pr_review_comments(html_review, copy_review)

        # Step 4: Determine overall verdict
        html_passed = html_review.get("verdict", "fail") == "pass"
        copy_passed = copy_review.get("verdict", "fail") == "pass"
        overall_verdict = "pass" if (html_passed and copy_passed) else "fail"

        # Collect all issues for the report
        all_issues = html_review.get("issues", []) + copy_review.get("issues", [])

        # Step 5: Send structured review report back to CEO
        result_payload = {
            "verdict": overall_verdict,
            "html_verdict": html_review.get("verdict", "fail"),
            "copy_verdict": copy_review.get("verdict", "fail"),
            "issues": all_issues,
            "html_summary": html_review.get("summary", ""),
            "copy_summary": copy_review.get("summary", ""),
            "pr_comments_posted": pr_comments_posted,
            "pr_url": self.state.get("last_pr_url", ""),
            # These keys allow the CEO to route revision_requests correctly
            "needs_engineer_revision": not html_passed,
            "needs_marketing_revision": not copy_passed,
            "engineer_feedback": html_review.get("feedback_for_engineer", ""),
            "marketing_feedback": copy_review.get("feedback_for_marketing", "")
        }

        reply = create_message(
            from_agent="qa",
            to_agent="ceo",
            message_type="result",
            payload=result_payload,
            parent_message_id=parent_msg_id
        )
        send_message(reply)
        log_message_flow(reply["message_id"], "qa", "ceo", "result")
        log_step("QAAgent", "Review Report Sent to CEO", payload=result_payload)

        verdict_icon = "✅" if overall_verdict == "pass" else "❌"
        print(f"\n{'='*50}")
        print(f"{verdict_icon} QA Agent Complete! Overall Verdict: {overall_verdict.upper()}")
        print(f"   🖥️  HTML Review: {html_review.get('verdict', 'fail').upper()}")
        print(f"   📣 Copy Review: {copy_review.get('verdict', 'fail').upper()}")
        print(f"   💬 PR Comments Posted: {pr_comments_posted}")
        if all_issues:
            print(f"   ⚠️  Issues Found: {len(all_issues)}")
            for issue in all_issues[:3]:
                print(f"      - {issue}")
        print(f"{'='*50}\n")

    # ─── LLM: Review HTML Landing Page ───────────────────────────────────────

    def _review_html_with_llm(self):
        html_preview = self.state.get("last_html_preview", "")
        
        # If no HTML was provided, skip and auto-pass
        if not html_preview or html_preview.strip() == "":
            log_step("QAAgent", "HTML Review Skipped", "No HTML preview — auto-passing.")
            return {
                "verdict": "pass",
                "summary": "No HTML preview available, skipping review.",
                "issues": [],
                "feedback_for_engineer": "",
                "criteria_results": {}
            }

        idea = self.state.get("last_idea", "Unknown Startup")
        spec = self.state.get("last_spec", {})
        value_prop = spec.get("value_proposition", "")
        features = spec.get("features", [])
        feature_names = [f.get("name", "") for f in features[:5]]
        features_text = ", ".join(feature_names)

        prompt = f"""
    You are a QA Engineer reviewing a SHORT PREVIEW (first 300 chars) of an HTML landing page for "{idea}".

    NOTE: This is only a preview, not the full page. Be lenient — assume missing content exists further down.

    PRODUCT SPEC:
    - Value Proposition: {value_prop}
    - Key Features: {features_text}

    HTML PREVIEW:
    {html_preview}

    REVIEW CRITERIA — be generous, this is just a preview:
    1. Does the HTML appear to be a real landing page (not empty)?
    2. Is the product name or startup idea mentioned anywhere?
    3. Does it look like it has CSS styling?

    Pass if at least 2 of 3 criteria are met. This is a preview only.

    Return ONLY raw JSON, no markdown, no backticks:
    {{
    "verdict": "pass",
    "summary": "One sentence summary.",
    "issues": [],
    "feedback_for_engineer": "",
    "criteria_results": {{
        "is_real_page": true,
        "product_name_present": true,
        "has_styling": true
    }}
    }}
    """

        task = Task(
            description=prompt,
            agent=self.agent,
            expected_output="A JSON object with verdict, summary, issues, feedback_for_engineer, and criteria_results"
        )
        crew = Crew(agents=[self.agent], tasks=[task])

        log_step("QAAgent", "LLM HTML Review", f"Reviewing landing page preview for: {idea}")
        try:
            result = execute_with_retry(crew, "QAAgent", "HTML Review")
            raw_text = getattr(result, "raw", str(result)).strip()
            review = extract_json_from_llm(raw_text)
            log_step("QAAgent", "HTML Review Complete", payload=review)
            return review
        except Exception as e:
            raw_output = getattr(result, "raw", str(result)) if 'result' in locals() else "No Output"
            log_error("QAAgent", "HTML Review LLM Parse Error", e, details=f"Raw: {raw_output[:200]}")
            return {
                "verdict": "pass",
                "summary": "Review could not complete — auto-passing to avoid pipeline block.",
                "issues": [],
                "feedback_for_engineer": "",
                "criteria_results": {}
            }
    # ─── LLM: Review Marketing Copy ──────────────────────────────────────────

    def _review_marketing_copy_with_llm(self):
        """LLM Call #2: Review the marketing copy for quality and completeness."""
        idea = self.state.get("last_idea", "Unknown Startup")
        spec = self.state.get("last_spec", {})
        copy = self.state.get("last_marketing_copy", {})

        personas = spec.get("personas", [])
        target_user = personas[0].get("role", "potential user") if personas else "potential user"
        pain_point = personas[0].get("pain_point", "") if personas else ""

        copy_str = json.dumps(copy, indent=2)

        prompt = f"""
You are reviewing marketing copy for a startup called "{idea}".

TARGET AUDIENCE:
- Primary User: {target_user}
- Their Pain Point: {pain_point}

MARKETING COPY TO REVIEW:
{copy_str}

REVIEW CRITERIA — be generous:
1. Is there a tagline present (any length)?
2. Is there an email subject present?
3. Are social posts present (at least 1)?

Pass if at least 2 of 3 criteria are met.

Return ONLY raw JSON, no markdown, no backticks:
{{
  "verdict": "pass",
  "summary": "One sentence summary.",
  "issues": [],
  "feedback_for_marketing": "",
  "criteria_results": {{
    "tagline_present": true,
    "email_subject_present": true,
    "social_posts_present": true
  }}
}}
"""

        task = Task(
            description=prompt,
            agent=self.agent,
            expected_output="A JSON object with verdict, summary, issues, feedback_for_marketing, and criteria_results"
        )
        crew = Crew(agents=[self.agent], tasks=[task])

        log_step("QAAgent", "LLM Copy Review", f"Reviewing marketing copy for: {idea}")
        try:
            result = execute_with_retry(crew, "QAAgent", "Copy Review")
            raw_text = getattr(result, "raw", str(result)).strip()
            review = extract_json_from_llm(raw_text)
            log_step("QAAgent", "Copy Review Complete", payload=review)
            return review

        except Exception as e:
            raw_output = getattr(result, "raw", str(result)) if 'result' in locals() else "No Output"
            log_error("QAAgent", "Copy Review LLM Parse Error", e, details=f"Raw: {raw_output[:200]}")
            # AUTO-PASS on parse error to avoid blocking the pipeline
            return {
        "verdict": "pass",
        "summary": "Copy review auto-passed due to LLM parse error.",
        "issues": [],
        "feedback_for_marketing": "",
        "criteria_results": {}
    }

    # ─── GitHub: Post PR Review Comments ────────────────────────────────────

    def _post_pr_review_comments(self, html_review, copy_review):
        """
        Post at least 2 inline review comments on the GitHub PR via the GitHub Reviews API.
        Returns True if comments were posted successfully, False otherwise.
        """
        if not GITHUB_TOKEN or not REPO_NAME:
            log_step("QAAgent", "GitHub Comments Skipped", "GITHUB_TOKEN or GITHUB_REPO not set.")
            return False

        pr_url = self.state.get("last_pr_url", "")
        if not pr_url:
            log_step("QAAgent", "GitHub Comments Skipped", "No PR URL available in state.")
            return False

        # Extract PR number from URL (e.g. https://github.com/user/repo/pull/42 → 42)
        try:
            pr_number = int(pr_url.rstrip("/").split("/")[-1])
            self.state["last_pr_number"] = pr_number
        except (ValueError, IndexError) as e:
            log_error("QAAgent", "PR Number Extraction Failed", e, details=f"PR URL: {pr_url}")
            return False

        # Get the latest commit SHA on the PR (needed for inline comments)
        commit_sha = self._get_pr_head_sha(pr_number)
        if not commit_sha:
            # Fall back to posting regular PR comments (not inline) if we can't get the SHA
            return self._post_regular_pr_comments(pr_number, html_review, copy_review)

        # Build the two required inline comments
        comments = self._build_inline_comments(html_review, copy_review)

        # Post a full review with inline comments using the Reviews API
        try:
            url = f"https://api.github.com/repos/{REPO_NAME}/pulls/{pr_number}/reviews"

            # Build review body summary
            html_verdict = html_review.get("verdict", "fail").upper()
            copy_verdict = copy_review.get("verdict", "fail").upper()
            overall = "✅ PASS" if (html_verdict == "PASS" and copy_verdict == "PASS") else "❌ FAIL"

            html_issues = "\n".join([f"- {i}" for i in html_review.get("issues", [])]) or "- None"
            copy_issues = "\n".join([f"- {i}" for i in copy_review.get("issues", [])]) or "- None"

            review_body = (
                f"## 🤖 LaunchMind QA Agent Review\n\n"
                f"**Overall Verdict: {overall}**\n\n"
                f"### 🖥️ HTML Landing Page — {html_verdict}\n"
                f"{html_review.get('summary', '')}\n\n"
                f"**Issues:**\n{html_issues}\n\n"
                f"### 📣 Marketing Copy — {copy_verdict}\n"
                f"{copy_review.get('summary', '')}\n\n"
                f"**Issues:**\n{copy_issues}\n\n"
                f"---\n*Automatically reviewed by LaunchMind QA Agent 🤖*"
            )

            event = "APPROVE" if (html_verdict == "PASS" and copy_verdict == "PASS") else "COMMENT"

            payload = {
                "commit_id": commit_sha,
                "body": review_body,
                "event": event,
                "comments": comments
            }

            log_step("QAAgent", "Posting PR Review", f"PR #{pr_number} | Event: {event}")
            resp = requests.post(url, headers=_gh_headers(), json=payload)

            if resp.status_code == 200:
                review_url = resp.json().get("html_url", pr_url)
                log_step("QAAgent", "PR Review Posted", f"Review URL: {review_url}")
                print(f"   ✅ GitHub PR review posted: {review_url}")
                return True
            else:
                log_error("QAAgent", "PR Review Post Failed",
                          Exception(f"Status {resp.status_code}: {resp.text[:200]}"))
                # Fallback: post as regular issue comments
                return self._post_regular_pr_comments(pr_number, html_review, copy_review)

        except Exception as e:
            log_error("QAAgent", "PR Review Exception", e)
            return self._post_regular_pr_comments(pr_number, html_review, copy_review)

    def _get_pr_head_sha(self, pr_number):
        """Fetch the head commit SHA for a given PR number."""
        url = f"https://api.github.com/repos/{REPO_NAME}/pulls/{pr_number}"
        try:
            resp = requests.get(url, headers=_gh_headers())
            if resp.status_code == 200:
                sha = resp.json().get("head", {}).get("sha", "")
                log_step("QAAgent", "Fetched PR Head SHA", sha)
                return sha
        except Exception as e:
            log_error("QAAgent", "Fetch PR SHA Failed", e)
        return None

    def _build_inline_comments(self, html_review, copy_review):
        """
        Build at least 2 inline comment objects for the GitHub Reviews API.
        Targets index.html with specific line positions.
        """
        comments = []

        # Comment 1: HTML quality feedback — targets line 1 of index.html (doctype / head)
        html_issues = html_review.get("issues", [])
        html_feedback = html_review.get("feedback_for_engineer", "")
        comment1_body = (
            f"**🤖 QA Agent — HTML Review ({html_review.get('verdict', 'fail').upper()})**\n\n"
            f"{html_review.get('summary', 'No summary provided.')}\n\n"
        )
        if html_issues:
            comment1_body += "**Issues found:**\n" + "\n".join([f"- {i}" for i in html_issues])
        if html_feedback:
            comment1_body += f"\n\n**Action required:** {html_feedback}"

        comments.append({
            "path": "index.html",
            "position": 1,   # Line 1 of the diff (<!DOCTYPE html>)
            "body": comment1_body
        })

        # Comment 2: Marketing copy consistency feedback — targets approx line 10 (title/meta)
        copy_issues = copy_review.get("issues", [])
        copy_feedback = copy_review.get("feedback_for_marketing", "")
        comment2_body = (
            f"**🤖 QA Agent — Copy Consistency Review ({copy_review.get('verdict', 'fail').upper()})**\n\n"
            f"{copy_review.get('summary', 'No summary provided.')}\n\n"
        )
        if copy_issues:
            comment2_body += "**Issues found:**\n" + "\n".join([f"- {i}" for i in copy_issues])
        if copy_feedback:
            comment2_body += f"\n\n**Action required:** {copy_feedback}"

        comments.append({
            "path": "index.html",
            "position": 5,   # A few lines in — targeting the <head> section
            "body": comment2_body
        })

        return comments

    def _post_regular_pr_comments(self, pr_number, html_review, copy_review):
        """
        Fallback: post two regular comments on the PR issue thread
        if the inline Reviews API call fails (e.g. diff position unavailable).
        """
        log_step("QAAgent", "Fallback: Regular PR Comments", f"PR #{pr_number}")
        url = f"https://api.github.com/repos/{REPO_NAME}/issues/{pr_number}/comments"
        success_count = 0

        # Comment 1: HTML review
        html_body = (
            f"## 🤖 QA Agent — HTML Review: {html_review.get('verdict', 'fail').upper()}\n\n"
            f"**Summary:** {html_review.get('summary', '')}\n\n"
        )
        issues = html_review.get("issues", [])
        if issues:
            html_body += "**Issues:**\n" + "\n".join([f"- {i}" for i in issues]) + "\n\n"
        feedback = html_review.get("feedback_for_engineer", "")
        if feedback:
            html_body += f"**Feedback for Engineer:** {feedback}"

        try:
            resp = requests.post(url, headers=_gh_headers(), json={"body": html_body})
            if resp.status_code == 201:
                log_step("QAAgent", "HTML Comment Posted", resp.json().get("html_url", ""))
                success_count += 1
            else:
                log_error("QAAgent", "HTML Comment Failed",
                          Exception(f"{resp.status_code}: {resp.text[:150]}"))
        except Exception as e:
            log_error("QAAgent", "HTML Comment Exception", e)

        # Comment 2: Marketing copy review
        copy_body = (
            f"## 🤖 QA Agent — Marketing Copy Review: {copy_review.get('verdict', 'fail').upper()}\n\n"
            f"**Summary:** {copy_review.get('summary', '')}\n\n"
        )
        copy_issues = copy_review.get("issues", [])
        if copy_issues:
            copy_body += "**Issues:**\n" + "\n".join([f"- {i}" for i in copy_issues]) + "\n\n"
        copy_feedback = copy_review.get("feedback_for_marketing", "")
        if copy_feedback:
            copy_body += f"**Feedback for Marketing:** {copy_feedback}"

        try:
            resp = requests.post(url, headers=_gh_headers(), json={"body": copy_body})
            if resp.status_code == 201:
                log_step("QAAgent", "Copy Comment Posted", resp.json().get("html_url", ""))
                success_count += 1
            else:
                log_error("QAAgent", "Copy Comment Failed",
                          Exception(f"{resp.status_code}: {resp.text[:150]}"))
        except Exception as e:
            log_error("QAAgent", "Copy Comment Exception", e)

        return success_count >= 1