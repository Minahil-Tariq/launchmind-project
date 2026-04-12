import os
import json
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from message_bus import receive_messages, send_message, create_message
from utils.logger import log_step, log_error, log_message_flow, execute_with_retry, extract_json_from_llm

load_dotenv()

class ProductAgent:
    def __init__(self):
        # We try to use os.getenv, but fallback safely
        api_key = os.getenv("GROQ_API_KEY", "fallback_key_if_none")
        
        self.agent = Agent(
            role="Product Manager",
            goal="Define product specifications for startup ideas",
            backstory=(
                "You are an experienced product manager who specializes in "
                "defining value propositions, personas, product features, "
                "and user stories for new startups."
            ),
            verbose=os.getenv("DEBUG", "False") == "True",
            llm="groq/llama-3.1-8b-instant"
        )
        
        # State tracking block to protect against pipeline hangs
        self.state = {
            "last_idea": None,
            "last_focus": None
        }

    def process_messages(self):
        messages = receive_messages("product")

        if not messages:
            return

        for msg in messages:
            try:
                msg_type = msg.get("message_type")
                log_step("ProductAgent", "Process Message", f"Received '{msg_type}' from '{msg['from_agent']}'")

                if msg_type == "task":
                    # Handle normal first-time tasks natively 
                    self.state["last_idea"] = msg["payload"].get("idea", "Unknown Startup Idea")
                    self.state["last_focus"] = msg["payload"].get("focus", "Generate generic specs")
                    
                    self.generate_spec(
                        idea=self.state["last_idea"], 
                        focus=self.state["last_focus"], 
                        parent_msg_id=msg["message_id"]
                    )
                    
                elif msg_type == "revision_request":
                    # Fixes Silent Breaking Error: It must handle revisions!
                    feedback = msg["payload"].get("feedback", "Please improve the spec.")
                    log_step("ProductAgent", "Handling Revision", f"Received feedback: {feedback}")
                    
                    self.generate_spec(
                        idea=self.state["last_idea"], 
                        focus=f"CRITICAL FEEDBACK TO APPLY: {feedback} | Original Focus: {self.state['last_focus']}",
                        parent_msg_id=msg["message_id"]
                    )
            except Exception as e:
                log_error("ProductAgent", "Message Processing Failure", e, details=f"Message ID: {msg.get('message_id')}")

    def generate_spec(self, idea, focus, parent_msg_id):
        log_step("ProductAgent", "Generate Spec", f"Drafting spec for: {idea}")

        prompt = f"""
        Startup Idea: {idea}
        Focus: {focus}

        Create a structured product specification. Return ONLY valid JSON with these exact keys:

        {{
          "value_proposition": "One sentence: what the product does and for whom.",
          "personas": [
            {{"name": "...", "role": "...", "pain_point": "..."}},
            {{"name": "...", "role": "...", "pain_point": "..."}}
          ],
          "features": [
            {{"name": "...", "description": "...", "priority": 1}},
            {{"name": "...", "description": "...", "priority": 2}},
            {{"name": "...", "description": "...", "priority": 3}},
            {{"name": "...", "description": "...", "priority": 4}},
            {{"name": "...", "description": "...", "priority": 5}}
          ],
          "user_stories": [
            "As a [user], I want [action] so that [benefit].",
            "As a [user], I want [action] so that [benefit].",
            "As a [user], I want [action] so that [benefit]."
          ]
        }}

        Fill in all fields based on the startup idea. No markdown. No commentary. Pure JSON only.
        """

        task = Task(description=prompt, agent=self.agent, expected_output="Structured JSON product specification")
        crew = Crew(agents=[self.agent], tasks=[task])

        try:
            result = execute_with_retry(crew, "ProductAgent", "Generate Specification")
            raw_text = getattr(result, "raw", str(result)).strip()
            
            spec_json = extract_json_from_llm(raw_text)
            log_step("ProductAgent", "Generate Spec Success", payload=spec_json)
            
        except BaseException as e:
            raw_output = getattr(result, "raw", str(result)) if 'result' in locals() else "No Output"
            log_error("ProductAgent", "Spec JSON Parse Error", e, details=f"Raw text: {raw_output}")
            spec_json = {"error": "Failed to generate proper spec layout", "raw_output": raw_output}

        idea_str = self.state.get("last_idea") or idea

        # Send result back to CEO for review (The CEO will gatekeep and pass to downstream agents)
        ceo_reply = create_message(
            from_agent="product",
            to_agent="ceo",
            message_type="result",
            payload=spec_json,
            parent_message_id=parent_msg_id
        )
        send_message(ceo_reply)
        log_message_flow(ceo_reply["message_id"], "product", "ceo", "result")
        log_step("ProductAgent", "Dispatched to CEO", "Spec sent to CEO for review.")