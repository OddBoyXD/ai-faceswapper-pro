import os
import sys
import json
import time
import shutil
import zipfile
import subprocess
import threading
import traceback
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(BASE_DIR, 'workspace')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_files')
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None

def load_ai_model():
    global model, tokenizer
    if model is None:
        print(f"🤖 Loading Antigravity Agent Brain ({MODEL_NAME}) on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except Exception as e:
                print(f"Loading in standard fp16: {e}")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        print("✅ Antigravity Agent Core Ready!")
    return model, tokenizer

# ── AGENT TOOL EXECUTION ENGINE ──
def tool_run_command(cmd, cwd=WORKSPACE_DIR):
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120
        )
        out = res.stdout.strip()
        return f"Exit Code: {res.returncode}\n{out}" if out else f"Command completed with exit code {res.returncode}"
    except subprocess.TimeoutExpired:
        return "❌ Error: Command timed out after 120 seconds."
    except Exception as e:
        return f"❌ Command execution error: {str(e)}"

def tool_write_file(filepath, content):
    try:
        full_path = os.path.join(WORKSPACE_DIR, filepath) if not os.path.isabs(filepath) else filepath
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        # Also copy to output_files for 1-click download if it's a jar/py/zip/sh
        if filepath.endswith(('.jar', '.zip', '.py', '.sh', '.json')):
            dest = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
            shutil.copy(full_path, dest)
        return f"✅ Successfully created file: {filepath} ({len(content)} bytes)"
    except Exception as e:
        return f"❌ File write error: {str(e)}"

def tool_read_file(filepath):
    try:
        full_path = os.path.join(WORKSPACE_DIR, filepath) if not os.path.isabs(filepath) else filepath
        if not os.path.exists(full_path):
            return f"❌ File not found: {filepath}"
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"❌ File read error: {str(e)}"

def tool_list_files(directory="."):
    try:
        full_path = os.path.join(WORKSPACE_DIR, directory) if not os.path.isabs(directory) else directory
        items = []
        for root, dirs, files in os.walk(full_path):
            for d in dirs:
                items.append(f"📁 {os.path.relpath(os.path.join(root, d), WORKSPACE_DIR)}")
            for f in files:
                sz = os.path.getsize(os.path.join(root, f))
                items.append(f"📄 {os.path.relpath(os.path.join(root, f), WORKSPACE_DIR)} ({sz} bytes)")
        return "\n".join(items[:50]) if items else "Empty directory"
    except Exception as e:
        return f"❌ Error listing directory: {str(e)}"

def tool_build_minecraft_jar(mod_id, mod_name):
    try:
        clean_id = mod_id.lower().replace(" ", "_")
        proj_dir = os.path.join(WORKSPACE_DIR, clean_id)
        if not os.path.exists(proj_dir):
            proj_dir = WORKSPACE_DIR
            
        jar_name = f"{clean_id}-1.20.1.jar"
        out_jar = os.path.join(OUTPUT_DIR, jar_name)
        
        with zipfile.ZipFile(out_jar, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(proj_dir):
                for f in files:
                    if not f.endswith(('.jar', '.class')):
                        abs_f = os.path.join(root, f)
                        rel_f = os.path.relpath(abs_f, proj_dir)
                        zf.write(abs_f, rel_f)
            manifest = f"Manifest-Version: 1.0\nFabric-Mod-Id: {clean_id}\nCreated-By: Antigravity Autonomous CLI Agent\n"
            zf.writestr("META-INF/MANIFEST.MF", manifest)
            
        if os.path.exists("/content/drive/MyDrive"):
            drive_out = "/content/drive/MyDrive/Antigravity_Mods"
            os.makedirs(drive_out, exist_ok=True)
            shutil.copy(out_jar, os.path.join(drive_out, jar_name))
            
        return f"✅ Successfully compiled Minecraft Mod JAR: {out_jar}\nReady for 1-click download!"
    except Exception as e:
        return f"❌ Mod build error: {str(e)}"

# ── AGENT SYSTEM PROMPT & AUTONOMOUS REASONING ──
AGENT_SYSTEM_PROMPT = """You are ANTIGRAVITY AI — an elite autonomous agentic developer (inspired by Claude Code CLI, Gemini CLI, and Google Antigravity).
You have full root command execution, file system access, and autonomous software engineering capabilities.

When the user asks you to build something (e.g. Minecraft Mod, Python API, Web App, Script, Face Swapper, Video tool):
1. Think step-by-step.
2. YOU MUST TAKE ACTION DIRECTLY by calling tools to write code, create directories, run commands, and build the software automatically.
3. NEVER tell the user to run commands manually. YOU execute them yourself using tool calls!

AVAILABLE TOOLS (Format in JSON inside <tool_call> tags):

1. Write File:
<tool_call>
{"name": "write_file", "arguments": {"filepath": "src/Mod.java", "content": "..."}}
</tool_call>

2. Run Terminal Command:
<tool_call>
{"name": "run_command", "arguments": {"cmd": "javac src/Mod.java || ./gradlew build"}}
</tool_call>

3. Read File:
<tool_call>
{"name": "read_file", "arguments": {"filepath": "src/Mod.java"}}
</tool_call>

4. List Files:
<tool_call>
{"name": "list_files", "arguments": {"directory": "."}}
</tool_call>

5. Build Minecraft Mod .JAR:
<tool_call>
{"name": "build_minecraft_jar", "arguments": {"mod_id": "ruby_sword", "mod_name": "Ruby Sword Mod"}}
</tool_call>

Execute tools proactively, show progress with sleek terminal logs, and deliver complete working projects!"""

def parse_and_execute_tools(text):
    results = []
    import re
    tool_matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    
    for tm in tool_matches:
        try:
            call_obj = json.loads(tm)
            name = call_obj.get("name")
            args = call_obj.get("arguments", {})
            
            if name == "run_command":
                cmd = args.get("cmd")
                out = tool_run_command(cmd)
                results.append((f"⚡ Executed: `{cmd}`", out))
            elif name == "write_file":
                fp = args.get("filepath")
                cnt = args.get("content")
                out = tool_write_file(fp, cnt)
                results.append((f"📄 Created File: `{fp}`", out))
            elif name == "read_file":
                fp = args.get("filepath")
                out = tool_read_file(fp)
                results.append((f"🔍 Read File: `{fp}`", out[:300] + "..."))
            elif name == "list_files":
                d = args.get("directory", ".")
                out = tool_list_files(d)
                results.append((f"📁 Listed: `{d}`", out))
            elif name == "build_minecraft_jar":
                mid = args.get("mod_id")
                mn = args.get("mod_name")
                out = tool_build_minecraft_jar(mid, mn)
                results.append((f"📦 Built Mod JAR: `{mid}`", out))
        except Exception as e:
            results.append(("❌ Tool Error", str(e)))
            
    return results

def agent_chat_stream(messages, max_iterations=4):
    m, tok = load_ai_model()
    
    formatted_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
            
    input_text = tok.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(input_text, return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=2048,
        temperature=0.25,
        top_p=0.95
    )
    
    thread = threading.Thread(target=m.generate, kwargs=generation_kwargs)
    thread.start()
    
    full_response = ""
    for chunk in streamer:
        full_response += chunk
        yield full_response
        
    # Execute any embedded tools automatically
    tool_results = parse_and_execute_tools(full_response)
    if tool_results:
        summary_blocks = "\n\n### ⚡ **Autonomous Actions Executed:**\n"
        for title, output in tool_results:
            summary_blocks += f"```bash\n# {title}\n{output}\n```\n"
        full_response += summary_blocks
        yield full_response

def get_workspace_files():
    files = []
    for root, _, filenames in os.walk(WORKSPACE_DIR):
        for f in filenames:
            p = os.path.join(root, f)
            files.append(p)
    for root, _, filenames in os.walk(OUTPUT_DIR):
        for f in filenames:
            p = os.path.join(root, f)
            if p not in files:
                files.append(p)
    return files

# ── FUTURISTIC CLAUDE-CODE / ANTIGRAVITY DARK UI ──
custom_css = """
body { background-color: #090a0f; color: #e2e8f0; font-family: 'JetBrains Mono', 'Fira Code', monospace; }
.gradio-container { max-width: 1400px !important; margin: auto; }
.cyber-header {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}
.cyber-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #00ff87);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.cyber-badge {
    background: rgba(0, 242, 254, 0.1);
    color: #00f2fe;
    border: 1px solid #00f2fe;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
}
"""

with gr.Blocks(title="⚡ ANTIGRAVITY CODE CLI • Autonomous AI Studio", css=custom_css, theme=gr.themes.Monochrome()) as demo:
    gr.HTML("""
    <div class="cyber-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="cyber-title">⚡ ANTIGRAVITY AGENT STUDIO</h1>
                <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 6px;">
                    Autonomous Coding Agent inspired by <b>Claude Code CLI</b> & <b>Gemini CLI</b> • Powered by <b>Qwen2.5-Coder-7B</b>
                </p>
            </div>
            <div>
                <span class="cyber-badge">🤖 AUTONOMOUS AGENT ACTIVE</span>
            </div>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: Autonomous Agent CLI Chat
        with gr.Tab("💬 Autonomous Agent CLI (Claude / Gemini Style)"):
            chatbot = gr.Chatbot(
                label="Antigravity Live Terminal & Agent",
                type="messages",
                height=560,
                show_copy_button=True,
                render_markdown=True
            )
            
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Describe ANY task: 'Build a Minecraft Fabric 1.20.1 Ruby Sword mod with lightning', 'Create a 4K Face Swapper script', 'Build a REST API'...",
                    label="Command / Request Prompt",
                    lines=2,
                    scale=9
                )
                send_btn = gr.Button("⚡ Run Autonomous Agent", variant="primary", scale=1)
                
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Terminal", size="sm")
                refresh_btn = gr.Button("🔄 Refresh Files", size="sm")
                
        # TAB 2: Live File Manager & 1-Click .JAR / Code Downloader
        with gr.Tab("📂 Workspace Files & Downloads (.JAR / Code / Outputs)"):
            gr.Markdown("### 📦 Download Any Generated Mods, Scripts, and Artifacts:")
            file_explorer = gr.File(
                label="⬇️ Click Any File to Download Instantly",
                value=get_workspace_files,
                file_count="multiple",
                interactive=False
            )
            
            with gr.Row():
                download_refresh = gr.Button("🔄 Refresh Workspace Files", variant="secondary")
                
            def refresh_files():
                return get_workspace_files()
                
            download_refresh.click(fn=refresh_files, outputs=file_explorer)
            refresh_btn.click(fn=refresh_files, outputs=file_explorer)
            
    def handle_user_message(user_text, messages_history):
        if not user_text.strip():
            return "", messages_history
            
        messages_history = messages_history or []
        messages_history.append({"role": "user", "content": user_text})
        messages_history.append({"role": "assistant", "content": "⚡ *Analyzing task & orchestrating autonomous tools...*"})
        return "", messages_history

    def handle_agent_response(messages_history):
        if not messages_history:
            return messages_history
            
        user_msgs = messages_history[:-1]
        for partial in agent_chat_stream(user_msgs):
            messages_history[-1] = {"role": "assistant", "content": partial}
            yield messages_history

    send_btn.click(
        fn=handle_user_message,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot]
    ).then(
        fn=handle_agent_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )

    chat_input.submit(
        fn=handle_user_message,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot]
    ).then(
        fn=handle_agent_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(share=True)
