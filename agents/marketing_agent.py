import os
import json
import requests
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from message_bus import receive_messages, send_message, create_message
from utils.logger import log_step, log_error, log_message_flow, execute_with_retry, extract_json_from_llm

# ─── Config ──────────────────────────────────────────────────────────────────
SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")


class MarketingAgent:
    """
    Marketing Agent — Assignment Role:
      1. Receives product spec from Product agent
      2. Uses LLM to generate: tagline, description, cold-outreach email, 3 social posts
      3. Sends email via SendGrid (LLM-generated subject + body)
      4. Posts Slack message (Block Kit) with tagline, description, PR link
      5. Sends all copy back to CEO as structured JSON result
    """

    def __init__(self):
        self.agent = Agent(
            role="Growth Marketing Manager",
            goal="Create compelling marketing materials that convert potential users into customers",
            backstory=(
                "You are a growth-focused marketing expert who has launched dozens of SaaS products. "
                "You write punchy taglines, data-driven cold emails, and viral social media posts "
                "that speak directly to user pain points."
            ),
            verbose=os.getenv("DEBUG", "False") == "True",
            llm="groq/llama-3.3-70b-versatile",
            max_rpm=2
        )

        self.state = {
            "last_idea": None,
            "last_spec": None,
            "last_pr_url": None,
            "last_focus": None
        }

    # ─── Message Loop ────────────────────────────────────────────────────────

    def process_messages(self):
        messages = receive_messages("marketing")
        if not messages:
            return

        for msg in messages:
            try:
                msg_type = msg.get("message_type")
                log_step("MarketingAgent", "Process Message",
                        f"Received '{msg_type}' from '{msg['from_agent']}'")

                if msg_type == "task":
                    payload = msg["payload"]
                    focus = payload.get("focus", "")

                    # NEW: Agar sirf PR URL update hai toh poora run mat karo
                    if "Update PR URL" in focus:
                        incoming_pr = payload.get("pr_url", "")
                        if incoming_pr:
                            self.state["last_pr_url"] = incoming_pr
                            log_step("MarketingAgent", "PR URL Updated", incoming_pr)
                        continue  # skip _run(), sirf URL update karo

                    self.state["last_idea"] = payload.get("idea", "Unknown Startup")
                    self.state["last_spec"] = payload.get("spec", {})
                    self.state["last_pr_url"] = payload.get("pr_url", "https://github.com")
                    self.state["last_focus"] = focus
                    self._run(parent_msg_id=msg["message_id"])

                elif msg_type == "revision_request":
                    feedback = msg["payload"].get("feedback", "Please improve the marketing copy.")
                    log_step("MarketingAgent", "Handling Revision", f"Feedback: {feedback}")
                    self.state["last_focus"] = (
                        f"REVISION FEEDBACK TO APPLY: {feedback} | "
                        f"Original Task: Marketing for {self.state['last_idea']}"
                    )
                    self._run(parent_msg_id=msg["message_id"])

            except Exception as e:
                log_error("MarketingAgent", "Message Processing Failure", e,
                        details=f"Message ID: {msg.get('message_id')}")

    # ─── Core Pipeline ───────────────────────────────────────────────────────

    def _run(self, parent_msg_id):
        """Generate all marketing copy with LLM → send email → post Slack → notify CEO."""
        copy = self._generate_marketing_copy_with_llm()

        # Step 1: Send email via SendGrid
        email_success = self._send_email(
            subject=copy.get("email_subject", f"Introducing {self.state['last_idea']}"),
            html_body=copy.get("email_html", "<p>Check out our new product!</p>")
        )

        # Step 2: Post to Slack (Block Kit)
        pr_url = self.state.get("last_pr_url") or "https://github.com"
        slack_success = self._post_to_slack(copy, pr_url)

        # Step 3: Send result back to CEO
        result_payload = {
            "status": "success" if (email_success and slack_success) else "partial",
            "tagline": copy.get("tagline", ""),
            "description": copy.get("description", ""),
            "email_subject": copy.get("email_subject", ""),
            "email_html": copy.get("email_html", ""),  # <--- YE LINE MISSING THI
            "social_posts": copy.get("social_posts", {}),
            "email_sent": email_success,
            "slack_posted": slack_success,
            "pr_url": pr_url
        }

        reply = create_message(
            from_agent="marketing",
            to_agent="ceo",
            message_type="result",
            payload=result_payload,
            parent_message_id=parent_msg_id
        )
        send_message(reply)
        log_message_flow(reply["message_id"], "marketing", "ceo", "result")
        log_step("MarketingAgent", "Result Sent to CEO", payload=result_payload)

        print(f"\n{'='*50}")
        print(f"✅ Marketing Agent Complete!")
        print(f"   📧 Email sent: {email_success}")
        print(f"   💬 Slack posted: {slack_success}")
        print(f"   🏷️  Tagline: {copy.get('tagline','')}")
        print(f"{'='*50}\n")

    # ─── LLM: Generate Marketing Copy ────────────────────────────────────────

    def _generate_marketing_copy_with_llm(self):
        """LLM generates all marketing assets from the product spec."""
        idea = self.state.get("last_idea", "Smart Startup")
        spec = self.state.get("last_spec", {})
        pr_url = self.state.get("last_pr_url", "https://github.com")

        value_prop = spec.get("value_proposition", idea)
        features = spec.get("features", [])
        features_text = ", ".join([f.get("name", "") for f in features[:5]])
        personas = spec.get("personas", [])
        target_user = personas[0].get("role", "potential user") if personas else "potential user"
        pain_point = personas[0].get("pain_point", "") if personas else ""

        prompt = f"""
You are an elite Silicon-Valley Growth Marketer and highly-paid D2C copywriter. Guarantee maximum psychological engagement and conversion for the following startup.

Startup: {idea}
Value Proposition: {value_prop}
Key Features: {features_text}
Primary Target User: {target_user}
Their Pain Point: {pain_point}

Return ONLY a raw JSON object with NO markdown formatting, NO backticks, and NO conversational text.
You MUST use this EXACT schema and escape all double quotes inside strings:

{{
  "tagline": "Under 8 words. Extremely punchy, memorable, and entirely benefit-focused hook.",
  "description": "2-3 sentences. Do not use generic startup jargon. Speak directly to their pain point and offer an irresistible solution.",
  "email_subject": "High-converting cold outreach email subject line (create curiosity, urgency, or extreme personalized value).",
  "email_html": "Full HTML body of an elite cold outreach email using the AIDA (Attention, Interest, Desire, Action) psychological framework. Embed pattern-interrupting openers, agitate their precise pain point, cleanly format 3 bullet-point benefits with emojis, and close with an irresistible CTA hyperlink to: {pr_url}",
  "social_posts": {{
    "twitter": "Under 280 characters. Start with a viral or contrarian hook. Format tightly with spacing and emojis. Drive clicks.",
    "linkedin": "Professional, story-driven 3-sentence post designed for LinkedIn virality. Focus on the 'Why' behind solving the pain point.",
    "instagram": "Highly aesthetic, casual hook for Instagram. Use bullet points and strategic hashtags. Drive viewers to the link in bio."
  }}
}}

All content must be hyper-specific to {idea} — completely avoid generic marketing filler (Do NOT say "We are excited to announce...").
"""

        task = Task(
            description=prompt,
            agent=self.agent,
            expected_output="A JSON object containing tagline, description, email_subject, email_html, and social_posts"
        )
        crew = Crew(agents=[self.agent], tasks=[task])

        log_step("MarketingAgent", "LLM Generate Copy", f"Calling LLM to generate marketing copy for: {idea}")
        try:
            result = execute_with_retry(crew, "MarketingAgent", "Marketing Copy Generation")
            raw_text = getattr(result, "raw", str(result)).strip()

            copy = extract_json_from_llm(raw_text)
            
            log_step("MarketingAgent", "Copy Generation Success", payload=copy)
            return copy

        except Exception as e:
            raw_output = getattr(result, "raw", str(result)) if 'result' in locals() else "No Output"
            error_msg = f"JSON Parsing Error: {str(e)}. The LLM output was unparseable. Raw text snippets: {raw_output[:200]}..."
            log_error("MarketingAgent", "LLM Copy Generation Failed", e, details=error_msg)
            
            return {
                "status": "failed",
                "error": error_msg
            }

    # ─── SendGrid Email ───────────────────────────────────────────────────────

    def _send_email(self, subject, html_body):
        """Send email via SendGrid HTTP API."""
        if not SENDGRID_API_KEY:
            log_step("MarketingAgent", "SendGrid Skipped", "SENDGRID_API_KEY not set.")
            print("   ⚠️  Email skipped: SENDGRID_API_KEY not configured.")
            return False
        if not FROM_EMAIL or not TO_EMAIL:
            log_step("MarketingAgent", "Email Skipped", "FROM_EMAIL or TO_EMAIL not set.")
            print("   ⚠️  Email skipped: FROM_EMAIL / TO_EMAIL not configured.")
            return False

        payload = {
            "personalizations": [{"to": [{"email": TO_EMAIL}]}],
            "from": {"email": FROM_EMAIL},
            "subject": subject,
            "content": [{"type": "text/html", "value": html_body}]
        }

        headers = {
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json"
        }

        log_step("MarketingAgent", "Sending Email", f"To: {TO_EMAIL} | Subject: {subject}")
        try:
            resp = requests.post("https://api.sendgrid.com/v3/mail/send",
                                 headers=headers, json=payload)
            if resp.status_code == 202:
                log_step("MarketingAgent", "Email Sent", f"✅ Delivered to {TO_EMAIL}")
                print(f"   ✅ Email sent to: {TO_EMAIL}")
                return True
            else:
                log_error("MarketingAgent", "Email Send Failed",
                          Exception(f"Status {resp.status_code}: {resp.text}"))
                print(f"   ❌ Email failed: {resp.status_code} — {resp.text[:100]}")
                return False
        except Exception as e:
            log_error("MarketingAgent", "Email Exception", e)
            print(f"   ❌ Email exception: {e}")
            return False

    # ─── Slack Block Kit ──────────────────────────────────────────────────────

    def _post_to_slack(self, copy, pr_url):
        """Post a Block Kit message to Slack #launches channel."""
        if not SLACK_TOKEN:
            log_step("MarketingAgent", "Slack Skipped", "SLACK_BOT_TOKEN not set.")
            print("   ⚠️  Slack skipped: SLACK_BOT_TOKEN not configured.")
            return False

        tagline = copy.get("tagline", self.state.get("last_idea", "New Product Launch"))
        description = copy.get("description", "Check out our new product!")
        twitter_post = copy.get("social_posts", {}).get("twitter", "")

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚀 New Product Launch!"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{tagline}*\n\n{description}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*GitHub PR:*\n<{pr_url}|View Pull Request>"},
                    {"type": "mrkdwn", "text": "*Status:*\nReady for Review ✅"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Social Post (Twitter):*\n{twitter_post}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Pull Request"},
                        "url": pr_url,
                        "style": "primary"
                    }
                ]
            }
        ]

        payload = {"channel": "#launches", "blocks": blocks}
        headers = {
            "Authorization": f"Bearer {SLACK_TOKEN}",
            "Content-Type": "application/json"
        }

        log_step("MarketingAgent", "Posting to Slack", "Channel: #launches")
        try:
            resp = requests.post("https://slack.com/api/chat.postMessage",
                                 headers=headers, json=payload)
            data = resp.json()
            if data.get("ok"):
                log_step("MarketingAgent", "Slack Posted", "✅ Message posted to #launches")
                print(f"   ✅ Slack message posted to #launches")
                return True
            else:
                err = data.get("error", "unknown_error")
                log_error("MarketingAgent", "Slack Post Failed", Exception(err))
                print(f"   ❌ Slack error: {err}")
                return False
        except Exception as e:
            log_error("MarketingAgent", "Slack Exception", e)
            print(f"   ❌ Slack exception: {e}")
            return False
