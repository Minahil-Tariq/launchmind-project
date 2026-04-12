import os
import json
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from message_bus import receive_messages, send_message, create_message
from utils.logger import log_step, log_error, log_message_flow, execute_with_retry, extract_json_from_llm

class CEOAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "fallback_key_if_none")
        
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            groq_api_key=api_key
        )

        self.agent = Agent(
            role="CEO Orchestrator",
            goal="Oversee the entire startup lifecycle from idea to execution",
            backstory=(
                "You are an expert startup CEO who excels at breaking down "
                "ideas into actionable tasks for different departments, and critically "
                "assessing their work for quality and iteration."
            ),
            verbose=os.getenv("DEBUG", "False") == "True",
            llm="groq/llama-3.3-70b-versatile",
            max_rpm=2
        )
        
        # State tracking — extended with QA fields
        self.state = {
            "product_accepted": False,
            "engineer_accepted": False,
            "marketing_accepted": False,
            "qa_accepted": False,                  # NEW: tracks QA approval
            "current_idea": None,
            "completed": False,
            "product_spec": None,                  # NEW: cached for forwarding to QA
            "engineer_result_payload": None,       # NEW: cached for forwarding to QA
            "marketing_result_payload": None,      # NEW: cached for forwarding to QA
            "revision_counts": {
                "product": 0,
                "engineer": 0,
                "marketing": 0,
                "qa": 0                            # NEW: QA revision guard
            }
        }

    def process_messages(self):
        """Polls for new messages and triggers appropriate handlers."""
        messages = receive_messages("ceo")
        
        for msg in messages:
            try:
                log_step("CEOAgent", "Process Message", f"Received '{msg['message_type']}' from '{msg['from_agent']}'")
                
                if msg["message_type"] == "idea":
                    self.state["current_idea"] = msg["payload"]["idea"]
                    self.decompose_idea(self.state["current_idea"], msg["message_id"])
                    
                elif msg["message_type"] == "result":
                    self.review_output(msg)
                    
            except Exception as e:
                log_error("CEOAgent", "Message Processing Failure", e, details=f"Message ID: {msg.get('message_id')}")

    def decompose_idea(self, idea, parent_msg_id):
        """LLM Call #1: Decompose the high-level idea into actionable components."""
        log_step("CEOAgent", "Decompose Idea", f"Breaking down: {idea}")
        
        prompt = f"""
        Startup Idea: {idea}
        
        Break this idea down into three specific focus areas/tasks:
        1. A task for the Product Manager to define specs, features, personas.
        2. A task for the Engineer to build the landing page based on specs.
        3. A task for the Marketing lead to define a tagline, write emails, and craft social posts.

        Return ONLY a raw JSON object with NO markdown formatting, NO backticks, and NO conversational text.
        You MUST use this EXACT schema where all values are simple strings:
        {{
            "product_task": "1-2 sentences describing the product manager's specific focus.",
            "engineer_task": "1-2 sentences describing the engineer's specific focus.",
            "marketing_task": "1-2 sentences describing the marketing specific focus."
        }}
        """
        
        task = Task(description=prompt, agent=self.agent, expected_output="A JSON object containing the three tasks.")
        crew = Crew(agents=[self.agent], tasks=[task])
        
        try:
            result = execute_with_retry(crew, "CEOAgent", "Idea Decomposition")
            raw_text = getattr(result, "raw", str(result)).strip()
            tasks_json = extract_json_from_llm(raw_text)
            log_step("CEOAgent", "Decompose Idea Success", payload=tasks_json)
            
        except BaseException as e:
            raw_output = getattr(result, "raw", str(result)) if 'result' in locals() else "No Output"
            log_error("CEOAgent", "Decompose JSON Parse Error", e, details=f"Raw text: {raw_output}")
            tasks_json = {
                "product_task": "Define target audience, personas, features, and value proposition.",
                "engineer_task": "Build responsive HTML landing page capturing value proposition.",
                "marketing_task": "Draft social media strategy, outreach emails, and a catchy tagline."
            }

        # Dispatch ONLY to Product first — engineer and marketing wait for CEO approval
        self._dispatch("product", idea, tasks_json.get("product_task", "Define Spec"), parent_msg_id)

    def _dispatch(self, target, idea, focus, parent_msg_id):
        payload = {"idea": idea, "focus": focus}
        msg = create_message("ceo", target, "task", payload, parent_message_id=parent_msg_id)
        send_message(msg)
        log_message_flow(msg["message_id"], "ceo", target, "task")

    def _dispatch_to_qa(self, parent_msg_id):
        """Forward engineer + marketing outputs to the QA agent for review."""
        eng_payload = self.state.get("engineer_result_payload", {})
        mkt_payload = self.state.get("marketing_result_payload", {})
        spec        = self.state.get("product_spec", {})

        qa_task_payload = {
            "idea":         self.state.get("current_idea", ""),
            "spec":         spec,
            "pr_url":       eng_payload.get("pr_url", ""),
            "html_preview": eng_payload.get("html_preview", ""),
            "marketing_copy": {
                "tagline":       mkt_payload.get("tagline", ""),
                "email_subject": mkt_payload.get("email_subject", ""),
                "social_posts":  mkt_payload.get("social_posts", {}),
                "description":   mkt_payload.get("description", "")
            }
        }

        qa_msg = create_message(
            from_agent="ceo",
            to_agent="qa",
            message_type="task",
            payload=qa_task_payload,
            parent_message_id=parent_msg_id
        )
        send_message(qa_msg)
        log_message_flow(qa_msg["message_id"], "ceo", "qa", "task")
        log_step("CEOAgent", "Dispatched to QA", "Forwarded engineer + marketing outputs for review.")

    def review_output(self, msg):

        source = msg["from_agent"]
        payload = msg["payload"]

        log_step("CEOAgent", "Review Output", source)

        if source == "product":
            self.state["product_accepted"] = True
            self.state["product_spec"] = payload

            eng_payload = {
                "idea": self.state["current_idea"],
                "spec": payload,
                "focus": "Build Landing Page"
            }
            eng_msg = create_message("ceo", "engineer", "task", eng_payload,
                                    parent_message_id=msg["message_id"])
            send_message(eng_msg)
            log_message_flow(eng_msg["message_id"], "ceo", "engineer", "task")

            mkt_payload = {
                "idea": self.state["current_idea"],
                "spec": payload,
                "focus": "Marketing Copy"
            }
            mkt_msg = create_message("ceo", "marketing", "task", mkt_payload,
                                    parent_message_id=msg["message_id"])
            send_message(mkt_msg)
            log_message_flow(mkt_msg["message_id"], "ceo", "marketing", "task")

        elif source == "engineer":
            if payload.get("status") == "failed":
                log_step("CEOAgent", "Engineer Failed", payload.get("error", ""))
                return
            self.state["engineer_accepted"] = True
            self.state["engineer_result_payload"] = payload
            log_step("CEOAgent", "Engineer Accepted", "PR and HTML received.")
            
            # NEW: Marketing ko actual PR URL bhejo
            actual_pr_url = payload.get("pr_url", "")
            if actual_pr_url:
                pr_update_msg = create_message(
                    from_agent="ceo",
                    to_agent="marketing",
                    message_type="task",
                    payload={
                        "idea": self.state["current_idea"],
                        "spec": self.state.get("product_spec", {}),
                        "focus": "Update PR URL only — regenerate Slack post with correct PR link",
                        "pr_url": actual_pr_url
                    },
                    parent_message_id=msg["message_id"]
                )
                send_message(pr_update_msg)
                log_message_flow(pr_update_msg["message_id"], "ceo", "marketing", "task (PR URL update)")

        elif source == "marketing":
            # Accept directly — no LLM review needed, QA handles this
            self.state["marketing_accepted"] = True
            self.state["marketing_result_payload"] = payload
            log_step("CEOAgent", "Marketing Accepted", "Copy received.")

        elif source == "qa":
            self.state["qa_accepted"] = True

            needs_eng_revision = payload.get("needs_engineer_revision", False)
            needs_mkt_revision = payload.get("needs_marketing_revision", False)

            if needs_eng_revision:
                rev = create_message("ceo", "engineer", "revision_request",
                                    {"feedback": payload.get("engineer_feedback", "")},
                                    parent_message_id=msg["message_id"])
                send_message(rev)
                log_message_flow(rev["message_id"], "ceo", "engineer", "revision_request (QA)")
                self.state["engineer_accepted"] = False
                self.state["engineer_result_payload"] = None
                self.state["qa_accepted"] = False

            if needs_mkt_revision:
                rev = create_message("ceo", "marketing", "revision_request",
                                    {"feedback": payload.get("marketing_feedback", "")},
                                    parent_message_id=msg["message_id"])
                send_message(rev)
                log_message_flow(rev["message_id"], "ceo", "marketing", "revision_request (QA)")
                self.state["marketing_accepted"] = False
                self.state["marketing_result_payload"] = None
                self.state["qa_accepted"] = False

            if not needs_eng_revision and not needs_mkt_revision:
                if not self.state["completed"]:
                    self.send_slack_summary()
                    self.state["completed"] = True

        # Dispatch to QA once engineer AND marketing both done
        eng_payload = self.state.get("engineer_result_payload") or {}
        mkt_payload = self.state.get("marketing_result_payload") or {}

        if (
            self.state["product_accepted"]
            and eng_payload.get("status") == "success"
            and mkt_payload.get("tagline", "")
            and not self.state["qa_accepted"]
        ):
            self._dispatch_to_qa(msg["message_id"])
    def send_slack_summary(self):
        log_step("CEOAgent", "Final Summary", "All agents have completed correctly.")
        try:
            import requests
            token = os.environ.get("SLACK_BOT_TOKEN")
            if not token:
                return

            eng_payload = self.state.get("engineer_result_payload") or {}
            pr_url = eng_payload.get("pr_url", "")
            issue_url = eng_payload.get("issue_url", "")

            payload = {
                "channel": "#launches",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Startup Orchestration Complete! 🚀"}
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"*Idea:* {self.state['current_idea']}\n"
                                "All agents (Product, Engineer, Marketing, QA) completed successfully."
                            )
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*GitHub PR:*\n<{pr_url}|View Pull Request>"},
                            {"type": "mrkdwn", "text": f"*GitHub Issue:*\n<{issue_url}|View Issue>"}
                        ]
                    }
                ]
            }

            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json=payload
            )
            log_step("CEOAgent", "Slack API Hit", payload={"status": response.status_code})

        except Exception as e:
            log_error("CEOAgent", "Slack Publish Error", e)
