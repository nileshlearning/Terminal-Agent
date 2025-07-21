# 🤖 Terminal Agent

![Terminal Agent Logo](https://img.icons8.com/fluency/96/console.png)
![Everyday Agents Family](https://img.icons8.com/color/96/artificial-intelligence.png)

## Overview

**Terminal Agent** is an intelligent, AI-powered command-line assistant designed to help users interact with their system terminal using natural language. It is a core member of the "Everyday Agents" family—a suite of agents built to simplify daily digital tasks through automation and conversational AI.

Terminal Agent leverages Large Language Models (LLMs) to understand user queries, plan terminal command execution, and provide direct answers or execute commands as needed. It supports multiple LLM providers and can be configured for local or cloud-based models.

---

## Features

- **Natural Language Interface:** Ask questions or request system actions in plain English.
- **Intelligent Planning:** The agent analyzes your query, determines if terminal commands are needed, and plans steps accordingly.
- **Safe Execution:** Commands are only run with your explicit confirmation.
- **Multi-Provider Support:** Configure with OpenAI, OpenRouter, Together, or Local (Ollama) models.
- **Error Handling & Replanning:** If a command fails, the agent can replan and retry.
- **Rich Output:** Uses the `rich` library for beautiful, readable terminal output.
- **Extensible:** Designed as part of the "Everyday Agents" family for future integration.

---

## Supported LLM Providers & Models

- **OpenAI:**  
  - Example: `gpt-3.5-turbo`, `gpt-4`
- **OpenRouter:**  
  - Example: `openai/gpt-3.5-turbo`, `google/gemini-pro`
- **Together:**  
  - Example: `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-Instruct-v0.2`
- **Local (Ollama):**  
  - Example: `llama3`, `mistral`, `phi3`

You can select and configure your preferred provider and model during setup.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/terminal-agent.git
   cd terminal-agent
   ```


2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Docker Usage

You can run Terminal Agent in a containerized environment using Docker:

1. **Build the Docker image:**
   ```sh
   docker build -t terminal-agent .
   ```

2. **Run the container:**
   ```sh
   docker run -it terminal-agent
   ```

This will launch the agent inside the container. You can mount your config or data as needed using Docker volume options.

3. **(Optional) Install Local LLM (Ollama):**
   - Visit [Ollama](https://ollama.com/) and follow installation instructions.

---

## Configuration

On first run, Terminal Agent will prompt you to configure your LLM provider and model:

- **Provider:** Choose from OpenAI, OpenRouter, Together, or Local.
- **API Key:** Enter your API key for cloud providers (leave blank for Local).
- **Model:** Select the model you wish to use.

Configuration is saved in `config/config.json` for future sessions.

## Configuration File: `config/config.json`

The `config/config.json` file stores your LLM provider, API key, and model selection. It is automatically created and updated when you run the agent and complete the setup prompts.

**Example `config/config.json`:**
```json
{
  "provider": "Together",
  "api_key": "your_api_key_here",
  "model": "meta-llama/Llama-2-7b-chat-hf"
}
```

**Fields:**
- `provider`: The LLM provider to use. Options: `"OpenAI"`, `"OpenRouter"`, `"Together"`, `"Local"`
- `api_key`: Your API key for the selected provider (leave blank for `"Local"`)
- `model`: The model name for your provider (e.g., `"gpt-3.5-turbo"`, `"llama3"`)

**How to edit:**
- You can manually edit `config/config.json` with any text editor.
- Make sure to use valid JSON format.
- If you change providers or models, update the fields accordingly.

**Tip:**  
If you delete `config/config.json`, the agent will prompt you to reconfigure on next run.

---

## Usage

Run the agent from your terminal:

```sh
python src/main.py
```

You'll see a welcome panel. Enter your requests in natural language, such as:

- `check if python is installed on my system`
- `list all files in the current directory`
- `install numpy using pip`
- `show me the current git branch`

The agent will analyze your query, decide if terminal commands are needed, and either answer directly or ask for permission to run commands. Outputs are displayed in a rich, readable format.

**Exit:**  
Type `exit`, `quit`, or `bye` to end the session.

---

## Example Session

```text
🤖 Terminal Agent
Your AI-powered terminal assistant
Type 'exit' or 'quit' to end session

❯ Enter your request: check if python is installed on my system
Do you want to run this command?
python --version [y/n] (y):
Success:
Python 3.11.2
```

---

## Advanced Features

- **Replanning:** If a command fails, the agent will attempt to fix and retry up to 5 times.
- **Model Optimization:** Automatically adjusts token usage for smaller models.
- **Extensible Design:** Easily add new providers or integrate with other agents in the "Everyday Agents" family.

---

## Project Structure

```
terminal_agent/
├── src/
│   └── main.py           # Main application file
├── config/
│   └── config.json       # LLM configuration
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build file
├── README.md             # Project documentation
├── LICENSE               # MIT License
```

---

## Part of "Everyday Agents" Family

Terminal Agent is one module in the "Everyday Agents" suite—a collection of smart agents designed to automate and simplify daily digital tasks. Stay tuned for more agents and integrations!

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests to help improve Terminal Agent and the Everyday Agents family.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author & Contact

Created and maintained by **Nilesh Maske**  
Email: [nileshmaske09@gmail.com](mailto:nileshmaske09@gmail.com)

For questions, suggestions, or collaboration, feel free to reach out!

---

**Empower your terminal. Automate your day. Welcome to Everyday Agents!**
