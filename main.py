import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from utils.logger import log_step, log_error, get_system_logger
from message_bus import send_message, create_message, message_bus
from agents.ceo_agent import CEOAgent

# Load all sub-agents safely
try:
    from agents.product_agent import ProductAgent
    has_product = True
except ImportError as e:
    has_product = False
    print(f"⚠️  ProductAgent not found: {e}")

try:
    from agents.engineer_agent import EngineerAgent
    has_engineer = True
except ImportError as e:
    has_engineer = False
    print(f"⚠️  EngineerAgent not found: {e}")

try:
    from agents.marketing_agent import MarketingAgent
    has_marketing = True
except ImportError as e:
    has_marketing = False
    print(f"⚠️  MarketingAgent not found: {e}")
try  :
    from agents.qa_agent import QAAgent
    has_qa = True
except ImportError as e:
    has_qa = False
    print(f"⚠️  QAAgent not found: {e}")

def simulate_idea_injection(idea: str):
    """Injects the initial startup idea into the CEO's message queue."""
    log_step("System", "Inject Idea", f"Injecting startup idea: {idea}")
    msg = create_message(
        from_agent="system",
        to_agent="ceo",
        message_type="idea",
        payload={"idea": idea}
    )
    send_message(msg)
    print(f"\n📨  [System → CEO] Idea injected: {idea}\n")


def run_orchestration():
    logger = get_system_logger()
    logger.info("=" * 60)
    logger.info("  LaunchMind Multi-Agent System — Starting Up")
    logger.info("=" * 60)

    print("\n" + "🎯" * 30)
    print("  LAUNCHMIND — MULTI-AGENT SYSTEM")
    print("🎯" * 30)

    # ── Inject the startup idea ──────────────────────────────────────────────
    startup_idea = (
        "Smart Study Planner that auto-generates personalised study schedules "
        "based on exam deadlines, so students can stop stressing and start acing their exams"
    )
    simulate_idea_injection(startup_idea)

    # ── Initialise agents ────────────────────────────────────────────────────
    ceo_agent = CEOAgent()
    agents_list = [ceo_agent]

    if has_product:
        agents_list.append(ProductAgent())
        print("✅  ProductAgent loaded")
    else:
        logger.warning("Running without ProductAgent")

    if has_engineer:
        agents_list.append(EngineerAgent())
        print("✅  EngineerAgent loaded")
    else:
        logger.warning("Running without EngineerAgent — GitHub actions will not execute")
    
    if has_marketing:
        agents_list.append(MarketingAgent())
        print("✅  MarketingAgent loaded")
    else:
        logger.warning("Running without MarketingAgent — Email/Slack will not run")
    if has_qa:
        agents_list.append(QAAgent())
        print("✅  QAAgent loaded")
    else:
        logger.warning("Running without QAAgent — No quality checks will be performed")      
    print(f"\n🚦  Starting event loop with {len(agents_list)} agents...\n")

    # ── Main event loop ──────────────────────────────────────────────────────
    max_loops = 80          # Enough for multi-revision cycles
    loop_count = 0

    try:
        while loop_count < max_loops:
            loop_count += 1

            # Check if CEO has finished orchestration
            if ceo_agent.state.get("completed", False):
                log_step("System", "Orchestration Complete",
                         "CEO agent signalled all sub-agents have been accepted.")
                print("\n🎉  All agents completed successfully! Orchestration done.\n")
                break

            # Let every agent process its inbox
            for agent in agents_list:
                try:
                    agent.process_messages()
                except Exception as e:
                    agent_name = agent.__class__.__name__
                    log_error("System", f"{agent_name} Process Failure", e)

            # Small pause to avoid CPU spinning and give API calls time to breathe
            time.sleep(1)

    except KeyboardInterrupt:
        log_step("System", "Orchestration Interrupted", "User manually stopped the process.")
        print("\n⛔  Interrupted by user.")
    except Exception as e:
        log_error("System", "Global Orchestration Crash", e, "Fatal error in main loop.")
    finally:
        logger.info(f"LaunchMind shutting down. Total event loops: {loop_count}")
        print(f"\n{'='*60}")
        print(f"  LaunchMind shut down after {loop_count} event loop(s).")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    run_orchestration()
