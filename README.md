# 🚀 LaunchMind: Smart Study Planner (Multi-Agent System)

## 📌 Startup Idea

Smart Study Planner is an AI-powered tool designed to help students manage their time effectively by automatically generating study schedules based on deadlines, subjects, and priorities. It reduces stress caused by poor time management and helps students stay consistent with their study goals.

---

## 🤖 Agent Architecture

This system is built using a Multi-Agent System (MAS) where different AI agents collaborate to simulate a startup workflow.

### Agents:

* **CEO Agent (Orchestrator)**
  Breaks down the startup idea into tasks, assigns them to other agents, reviews outputs, and manages feedback loops.

* **Product Agent**
  Creates a structured product specification including value proposition, user personas, features, and user stories.

* **Engineer Agent**
  Generates a landing page and interacts with GitHub by creating commits and pull requests.

* **Marketing Agent**
  Generates marketing content, sends emails, and posts launch updates to Slack.

* **QA Agent (Reviewer)**
  Reviews outputs from Engineer and Marketing agents and provides feedback to ensure quality.

---

### 🔁 Communication Flow

```
User → CEO Agent
      ↓
Product Agent → Engineer Agent → Marketing Agent
      ↓                    ↓
     QA Agent (Review & Feedback)
      ↓
      CEO Agent (Final Decision & Summary)
```

Agents communicate using structured JSON messages via a message bus.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/Minahil-Tariq/launchmind-project.git
cd launchmind-project
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Set Environment Variables

Create a `.env` file and add:

```
OPENAI_API_KEY=your_key
GITHUB_TOKEN=your_token
SLACK_BOT_TOKEN=your_token
SENDGRID_API_KEY=your_key
GITHUB_REPO=yourusername/repo-name
```

---

### 4. Run the System

```
python main.py
```

---

## 🔗 Platform Integrations

This project integrates with real-world platforms:

* **GitHub**

  * Engineer Agent creates branches, commits code, and opens pull requests.

* **Slack**

  * Marketing Agent posts launch announcements to the `#launches` channel.

* **SendGrid (Email)**

  * Marketing Agent sends outreach emails to a test inbox.

---

## 💬 Slack Workspace

Screenshots:
<img width="959" height="537" alt="image" src="https://github.com/user-attachments/assets/e75708b6-ea72-4980-9bc6-ead76bd4d2a2" />


---

## 🔗 GitHub Pull Request

Engineer Agent PR:
https://github.com/Minahil-Tariq/launchmind-project/pull/128

---

## 📁 Project Structure

```
launchmind/
│
├── agents/
│   ├── ceo_agent.py
│   ├── product_agent.py
│   ├── engineer_agent.py
│   ├── marketing_agent.py
│   └── qa_agent.py
│
├── main.py
├── message_bus.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ✅ Features Implemented

* Multi-agent collaboration using structured JSON messages
* Real GitHub integration (commit + pull request)
* Slack bot messaging using API
* Email sending using SendGrid
* Dynamic decision-making via CEO agent

---

## 👥 Team Members

* Student 1 → CEO Agent
* Student 2 → Product + QA Agent
* Student 3 → Engineer + Marketing Agent

---

## 🎯 Notes

* API keys are stored securely in `.env` and not committed
* System demonstrates real-world agent collaboration
* Designed for academic and learning purposes

---
