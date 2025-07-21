from rich.prompt import Prompt, Confirm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import subprocess
import sys
import json
import os
import requests
import re
import time
from typing import Dict, List, Optional, Tuple
import platform


# Together API import with fallback
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

console = Console()
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def setup_config():
    console.print("[bold yellow]LLM Configuration Required[/]")
    
    # Build provider choices based on availability
    provider_choices = ["OpenAI", "OpenRouter", "Local", "Together"]
    if TOGETHER_AVAILABLE:
        provider_choices.append("Together")
    
    provider = Prompt.ask("LLM Provider", choices=provider_choices, default="OpenAI")
    api_key = Prompt.ask("API Key (leave blank for Local)", default="") if provider != "Local" else ""
    
    if provider == "OpenAI":
        model = Prompt.ask("Model", default="gpt-3.5-turbo")
    elif provider == "OpenRouter":
        model = Prompt.ask("Model", default="openai/gpt-3.5-turbo")
    elif provider == "Together":
        model = Prompt.ask("Model", default="meta-llama/Llama-2-7b-chat-hf")
    else:
        model = Prompt.ask("Model", default="llama3")
    
    config = {"provider": provider, "api_key": api_key, "model": model}
    save_config(config)
    return config

def call_llm(prompt: str, config: Dict, max_tokens: int = 150, system_prompt: str = None) -> str:
    """Enhanced LLM call with better error handling and token management"""
    provider = config["provider"]
    api_key = config.get("api_key", "")
    model = config["model"]
    
    # Debug: Print the prompt being sent to LLM
    console.print(f"[cyan]DEBUG - LLM Prompt:[/]")
    console.print(f"[dim]{prompt}[/]")
    console.print(f"[cyan]--- End Prompt ---[/]")
    
    # Optimize prompt for smaller models
    if "1b" in model.lower() or "small" in model.lower():
        max_tokens = min(max_tokens, 80)
    
    # Default system prompt for command generation
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant. Answer questions in a friendly and informative way."
    
    
    if provider == "OpenAI":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        response = resp.json()["choices"][0]["message"]["content"]
        
        # Debug: Print the LLM response
        console.print(f"[cyan]DEBUG - LLM Response:[/]")
        console.print(f"[dim]{response}[/]")
        console.print(f"[cyan]--- End Response ---[/]")
        
        return response
        
    elif provider == "OpenRouter":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "TerminalAgent"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        response = resp.json()["choices"][0]["message"]["content"]
        
        # Debug: Print the LLM response
        console.print(f"[cyan]DEBUG - LLM Response:[/]")
        console.print(f"[dim]{response}[/]")
        console.print(f"[cyan]--- End Response ---[/]")
        
        return response
        
    elif provider == "Together":
        if not TOGETHER_AVAILABLE:
            raise Exception("Together API is not available. Please install: pip install together")
        
        try:
            client = Together(api_key=api_key)
            
            # Together API uses chat completion format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Try streaming first, fallback to non-streaming if it fails
            try:
                # Use streaming for better user experience
                response_text = ""
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    stream=True
                )
                
                console.print(f"[cyan]DEBUG - Together streaming response:[/]")
                for chunk in stream:
                    # Safely extract content from streaming response
                    try:
                        if (hasattr(chunk, 'choices') and 
                            chunk.choices and 
                            len(chunk.choices) > 0 and
                            hasattr(chunk.choices[0], 'delta') and
                            hasattr(chunk.choices[0].delta, 'content') and
                            chunk.choices[0].delta.content):
                            content = chunk.choices[0].delta.content
                            response_text += content
                            console.print(f"[dim]{content}[/]", end="")
                    except (AttributeError, IndexError) as e:
                        # Skip chunks that don't have expected structure
                        console.print(f"[dim yellow]Skipping chunk: {chunk}[/]")
                        continue
                
                console.print(f"\n[cyan]--- End Response ---[/]")
                return response_text.strip()
                
            except Exception as stream_error:
                console.print(f"[yellow]Streaming failed, trying non-streaming mode: {stream_error}[/]")
                # Fallback to non-streaming mode
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    stream=False
                )
                
                response_text = response.choices[0].message.content
                
                # Debug: Print the LLM response
                console.print(f"[cyan]DEBUG - Together non-streaming response:[/]")
                console.print(f"[dim]{response_text}[/]")
                console.print(f"[cyan]--- End Response ---[/]")
                
                return response_text.strip()
            
        except Exception as e:
            console.print(f"[red]Together API error: {e}[/]")
            raise
        
    elif provider == "Local":
        data = {
            "model": model,
            "prompt": f"System: {system_prompt}\nUser: {prompt}\nAssistant:",
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1
            }
        }
        resp = requests.post("http://192.168.137.148:11434/api/generate", json=data, timeout=60)
        resp.raise_for_status()
        response = resp.json().get("response", "")
        
        # Debug: Print the LLM response
        console.print(f"[cyan]DEBUG - LLM Response:[/]")
        console.print(f"[dim]{response}[/]")
        console.print(f"[cyan]--- End Response ---[/]")
        
        return response
    
    raise Exception(f"Unsupported provider: {provider}. Supported providers: OpenAI, OpenRouter, Together, Local")


class TerminalAgent:
    """Enhanced Terminal Agent with better planning and error handling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session_context = []
        self.is_small_model = self._is_small_model()
        
    def _is_small_model(self) -> bool:
        """Check if we're using a small model that needs special handling"""
        model = self.config.get("model", "").lower()
        return any(term in model for term in ["1b", "small", "tiny", "mini"])
    
    def process_query(self, query: str) -> str:
        
        system_prompt = f"""You are a terminal agent.
        Analyze the user's query and determine if answering it requires running a terminal command.
        If a terminal command is needed, rewrite the user's query to be more specific and clear (do not answer or provide a command), and respond with: {{"need_terminal_command": "yes", "query": "rewritten user query"}}
        If no terminal command is needed, answer the query directly and respond with: {{"need_terminal_command": "no", "answer": "direct answer"}}

        Respond in JSON only.
        """
        response = call_llm(query, self.config, max_tokens=1000, system_prompt=system_prompt)

        # from reponse extract the json data just take data between { and } using regex and then parse reponse as json
        try:
            json_data = re.search(r"\{.*\}", response).group()
            response_json = json.loads(json_data)
            return response_json
        except (json.JSONDecodeError, AttributeError) as e:
            console.print(f"[red]Error parsing LLM response: {e}[/]")
            return {"error": "Error processing your request. Please try again."}

    
    # def process_general_query(self, query, system_prompt = None):
    #     """Process a general query that does not require terminal commands"""
    #     if system_prompt is None:
    #         system_prompt = """
    #         You are a helpful assistant. Respond to the user's query in a concise and informative manner.
    #         """
    #     response = call_llm(query, self.config, max_tokens=3000, system_prompt=system_prompt)

    #     return response
    
    def plan_steps(self, query: str) -> str:

        system_prompt = f"""
        You are a terminal agent. Create a terminal execution plan for the query.

        - Only include sequential steps if each step depends on the previous one.
        - Do NOT list alternative commands for the same operation as separate steps.
        - For checks (e.g., verifying if Python is installed), provide only one command, unless multiple are strictly required.
        - Respond in JSON only, like: {{\"step1\": \"terminal command for step 1\", \"step2\": \"terminal command for step 2\", ...}}
        """
        response = call_llm(query, self.config, max_tokens=3000, system_prompt=system_prompt)
        cleaned_response = response.strip()
        if cleaned_response.startswith('```'):
            cleaned_response = re.sub(r'^```[a-zA-Z]*', '', cleaned_response)
            cleaned_response = cleaned_response.rstrip('`').strip()
        # Extract all JSON objects
        json_blocks = re.findall(r'\{.*?\}', cleaned_response, re.DOTALL)
        for block in json_blocks:
            try:
                response_json = json.loads(block)
                return response_json
            except json.JSONDecodeError:
                continue
        console.print(f"[red]Error parsing LLM response: No valid JSON found.[/]")
        console.print(f"[yellow]Raw response:[/] {cleaned_response}")
        return {"error": "Error processing your request. Please try again.", "raw_response": cleaned_response}

    def update_plan_steps(self, query: str, failed_command) -> str:
        """Update the execution plan based on a failed command"""
        system_prompt = f"""
        You are a terminal agent. Fix the execution plan due to error in command:

        these are the previous plan and results:
        {failed_command}

        Respond in JSON only.
        
        like this {{\"step\": \"fixed terminal command for step 1\", \"step\": \"fixed terminal command for step 2\", ...}}
        """
        response = call_llm(query, self.config, max_tokens=3000, system_prompt=system_prompt)
        cleaned_response = response.strip()
        if cleaned_response.startswith('```'):
            cleaned_response = re.sub(r'^```[a-zA-Z]*', '', cleaned_response)
            cleaned_response = cleaned_response.rstrip('`').strip()
        # Extract all JSON objects
        json_blocks = re.findall(r'\{.*?\}', cleaned_response, re.DOTALL)
        for block in json_blocks:
            try:
                response_json = json.loads(block)
                return response_json
            except json.JSONDecodeError:
                continue
        console.print(f"[red]Error parsing LLM response: No valid JSON found.[/]")
        console.print(f"[yellow]Raw response:[/] {cleaned_response}")
        return {"error": "Error processing your request. Please try again.", "raw_response": cleaned_response}
        
    
    def execute_command(self,response: Dict[str, str]):
        failed_command = {"status": "ok"}

        for command in response.values():
            failed_command[command] = "Not run yet"
            try:
                # Ask user permission to run the command
                if not Confirm.ask(f"Do you want to run this command?\n[bold yellow]{command}[/]", default=True):
                    console.print("[red]Command execution cancelled by user.[/]")
                    break
                # Show command in a new terminal window (Windows only)
                if platform.system() == "Windows":
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    console.print(f"[green]Success:[/]\n{stdout.decode()}")
                    failed_command[command] = stdout.decode()
                else:
                    console.print(f"[red]Error:[/]\n{stderr.decode()}")
                    failed_command[command] = stderr.decode()
                    failed_command["status"] = "error"
                    break
                # Display output in a panel
                console.print(Panel(
                    json.dumps(stdout.decode(), indent=2),
                    title="[bold green]Response[/]",
                    border_style="green"
                ))
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error:[/]\n{e}")
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/]")
                continue
        return failed_command
    
    def evaluate_response(self, query: str, terminal_output: str) -> Dict:
        """Evaluate if the terminal output answers the user's query using LLM."""
        system_prompt = (
            "You are a terminal agent. Given the user's query and the terminal output, "
            "determine if the output answers the query. "
            "If yes, respond with: {\"answered\": true, \"answer\": \"final answer to user\"}. "
            "If not, respond with: {\"answered\": false}. Respond in JSON only."
        )
        prompt = f"User Query: {query}\nTerminal Output: {terminal_output}"
        response = call_llm(prompt, self.config, max_tokens=500, system_prompt=system_prompt)
        try:
            json_data = re.search(r"\{.*\}", response).group()
            response_json = json.loads(json_data)
            return response_json
        except (json.JSONDecodeError, AttributeError) as e:
            console.print(f"[red]Error parsing evaluation response: {e}[/]")
            return {"answered": False, "error": "Could not evaluate response."}


def main():
    """Main application entry point"""
    try:
        # Load configuration
        config = load_config()
        if not config:
            config = setup_config()
        
        # Create terminal agent
        agent = TerminalAgent(config)
        
        # Display welcome message
        console.print(Panel.fit(
            "[bold cyan]🤖 Terminal Agent[/]\n"
            "[green]Your AI-powered terminal assistant[/]\n"
            "[yellow]Type 'exit' or 'quit' to end session[/]",
            title="Welcome",
            border_style="blue"
        ))
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold green]❯[/] Enter your request")
                # Check for exit commands
                if query.strip().lower() in ("exit", "quit", "bye"):
                    console.print("[yellow]👋 Goodbye![/]")
                    break
                # Process the query
                console.print(f"[dim]Processing: {query}[/]")
                response = agent.process_query(query)
                if response["need_terminal_command"] == "no":
                    console.print(f"[bold blue]Terminal Command Needed:[/]\n{response['answer']}")
                    response = response["answer"]
                    # Display response
                    console.print(Panel(
                        response,
                        title="[bold green]Response[/]",
                        border_style="green"
                    ))
                else:
                    console.print(f"[bold blue]Terminal Command Needed:[/]\n{response['query']}")
                    query = response["query"]
                    max_retries = 5
                    retry_count = 0
                    while retry_count < max_retries:
                        plan = agent.plan_steps(query)
                        failed_command = agent.execute_command(plan)
                        # Collect all terminal outputs
                        outputs = [v for k, v in failed_command.items() if k != "status"]
                        terminal_output = "\n".join(outputs)
                        eval_result = agent.evaluate_response(query, terminal_output)
                        if eval_result.get("answered"):
                            console.print(Panel(
                                eval_result.get("answer", "Query answered."),
                                title="[bold green]Final Answer[/]",
                                border_style="green"
                            ))
                            break
                        if failed_command["status"] == "ok":
                            console.print(f"[green]All commands succeeded, but query not answered. Output:[/]\n{failed_command}")
                            break
                        else:
                            console.print(f"[red]Command failed:[/]\n{failed_command}")
                            retry_count += 1
                            if retry_count < max_retries:
                                console.print(f"[yellow]Re-planning... Attempt {retry_count+1}/{max_retries}[/]")
                                plan = agent.update_plan_steps(query, failed_command)
                                continue
                            else:
                                console.print(f"[red]Maximum retries reached. Please check your commands or query.[/]")
                                break
            except KeyboardInterrupt:
                console.print("[yellow]👋 Goodbye![/]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                continue
    
    except KeyboardInterrupt:
        console.print("[yellow]👋 Goodbye![/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/]")
        sys.exit(1)
    

if __name__ == "__main__":
    main()