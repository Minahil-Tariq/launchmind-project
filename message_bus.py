import uuid
from datetime import datetime

# Shared message bus
message_bus = {
    "ceo": [],
    "product": [],
    "engineer": [],
    "marketing": [],
    "qa": []
}


def generate_message_id():
    return str(uuid.uuid4())


def current_timestamp():
    return datetime.utcnow().isoformat() + "Z"


def send_message(message):
    """
    Send message to the appropriate agent queue
    """
    to_agent = message["to_agent"]

    if to_agent not in message_bus:
        raise ValueError(f"Unknown agent: {to_agent}")

    message_bus[to_agent].append(message)


def receive_messages(agent_name):
    """
    Retrieve messages for an agent
    """
    if agent_name not in message_bus:
        raise ValueError(f"Unknown agent: {agent_name}")

    messages = message_bus[agent_name]
    message_bus[agent_name] = []  # clear inbox
    return messages


def create_message(from_agent, to_agent, message_type, payload, parent_message_id=None):
    """
    Create message according to assignment schema
    """
    VALID_TYPES = ["task", "result", "revision_request", "confirmation", "idea"]
    if message_type not in VALID_TYPES:
        raise ValueError(f"Invalid message_type '{message_type}'. MUST be one of {VALID_TYPES}")

    message = {
        "message_id": generate_message_id(),
        "from_agent": from_agent,
        "to_agent": to_agent,
        "message_type": message_type,
        "payload": payload,
        "timestamp": current_timestamp()
    }

    if parent_message_id:
        message["parent_message_id"] = parent_message_id

    return message