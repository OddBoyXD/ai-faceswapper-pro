import json
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Titan AI Studio")
OLLAMA_BASE = "http://127.0.0.1:11434"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Titan AI Studio • Supercluster</title>
    <!-- Tailwind CSS & Marked (Markdown parser) & Highlight.js -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .chat-container { height: calc(100dvh - 145px); }
        pre code { border-radius: 8px; font-size: 0.9rem; }
        .prose pre { padding: 0 !important; background: transparent !important; }
        .code-header { background: #1e293b; color: #94a3b8; padding: 6px 12px; font-size: 0.8rem; border-top-left-radius: 8px; border-top-right-radius: 8px; display: flex; justify-content: space-between; align-items: center; }
        .copy-btn { cursor: pointer; padding: 2px 8px; background: #334155; border-radius: 4px; color: #e2e8f0; font-size: 0.75rem; transition: all 0.2s; }
        .copy-btn:hover { background: #475569; color: white; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 flex flex-col h-screen overflow-hidden">

    <!-- Header -->
    <header class="h-14 border-b border-slate-800 bg-slate-900/80 backdrop-blur px-4 flex items-center justify-between shrink-0 z-10">
        <div class="flex items-center gap-2">
            <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center font-bold text-white shadow-lg">⚡</div>
            <div>
                <h1 class="font-bold text-base bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent leading-none">Titan AI Studio</h1>
                <span class="text-[10px] text-slate-400">48 EPYC Cores • 377 GB Supercluster</span>
            </div>
        </div>
        
        <div class="flex items-center gap-2">
            <select id="modelSelect" class="bg-slate-800 border border-slate-700 text-xs text-purple-300 font-medium rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-purple-500">
                <option value="qwen2.5-coder:7b">🌟 Qwen 2.5 Coder (Senior Dev)</option>
                <option value="dolphin-llama3:8b">🔓 Dolphin LLaMA3 (Uncensored)</option>
                <option value="llama3.2:3b">⚡ LLaMA 3.2 (Instant Fast)</option>
            </select>
            
            <button onclick="clearChat()" class="p-1.5 text-slate-400 hover:text-red-400 bg-slate-800/80 hover:bg-slate-800 rounded-lg transition" title="Clear Chat">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            </button>
        </div>
    </header>

    <!-- Chat Messages Container -->
    <main id="chatBox" class="chat-container flex-1 overflow-y-auto p-4 space-y-4 max-w-4xl w-full mx-auto">
        <div class="flex gap-3 max-w-[90%] md:max-w-[80%]">
            <div class="w-8 h-8 rounded-full bg-purple-600/30 border border-purple-500/50 flex items-center justify-center text-xs shrink-0">🤖</div>
            <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-3.5 text-sm text-slate-200 leading-relaxed shadow-sm">
                👋 Hello! I am your <b>Titan AI Assistant</b>. How can I help you today? Ask me to write code, solve problems, or discuss anything without restrictions!
            </div>
        </div>
    </main>

    <!-- Input Footer -->
    <footer class="border-t border-slate-800/80 bg-slate-900/90 backdrop-blur p-3 max-w-4xl w-full mx-auto shrink-0">
        <form id="chatForm" onsubmit="sendMessage(event)" class="flex items-center gap-2">
            <div class="relative flex-1">
                <textarea id="userInput" rows="1" placeholder="Ask anything, request code, or explore ideas..." class="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none leading-normal" onkeydown="handleKeyDown(event)"></textarea>
            </div>
            <button type="submit" id="sendBtn" class="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90 active:scale-95 text-white font-medium p-2.5 rounded-xl shadow-md transition shrink-0 flex items-center justify-center">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
            </button>
        </form>
    </footer>

    <script>
        let chatHistory = [];
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const modelSelect = document.getElementById('modelSelect');

        function scrollToBottom() {
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('chatForm').dispatchEvent(new Event('submit'));
            }
        }

        function copyCode(btn) {
            const codeBlock = btn.parentElement.nextElementSibling.querySelector('code');
            navigator.clipboard.writeText(codeBlock.innerText);
            btn.innerText = 'Copied!';
            setTimeout(() => { btn.innerText = 'Copy'; }, 2000);
        }

        function formatMarkdown(text) {
            const rawHtml = marked.parse(text);
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = rawHtml;
            
            tempDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
                const pre = block.parentElement;
                const wrapper = document.createElement('div');
                wrapper.className = 'my-3 rounded-lg overflow-hidden border border-slate-700 bg-slate-900';
                
                const lang = block.className.match(/language-(\w+)/)?.[1] || 'code';
                const header = document.createElement('div');
                header.className = 'code-header';
                header.innerHTML = `<span>${lang}</span><button class="copy-btn" onclick="copyCode(this)">Copy</button>`;
                
                pre.parentNode.insertBefore(wrapper, pre);
                wrapper.appendChild(header);
                wrapper.appendChild(pre);
            });
            return tempDiv.innerHTML;
        }

        function clearChat() {
            chatHistory = [];
            chatBox.innerHTML = `
                <div class="flex gap-3 max-w-[90%] md:max-w-[80%]">
                    <div class="w-8 h-8 rounded-full bg-purple-600/30 border border-purple-500/50 flex items-center justify-center text-xs shrink-0">🤖</div>
                    <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-3.5 text-sm text-slate-200 leading-relaxed shadow-sm">
                        🗑️ Chat cleared! Ask a new question or select a different model above.
                    </div>
                </div>
            `;
        }

        async function sendMessage(e) {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;
            
            userInput.value = '';
            userInput.disabled = true;
            sendBtn.disabled = true;
            sendBtn.classList.add('opacity-50');

            // Append User Message
            const userHtml = `
                <div class="flex justify-end">
                    <div class="bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-2xl rounded-tr-none p-3 text-sm max-w-[85%] md:max-w-[75%] shadow-md whitespace-pre-wrap leading-relaxed">
                        ${text.replace(/</g, "&lt;").replace(/>/g, "&gt;")}
                    </div>
                </div>
            `;
            chatBox.innerHTML += userHtml;
            chatHistory.push({ role: 'user', content: text });
            scrollToBottom();

            // Append Bot Message Container
            const botMsgId = 'bot-' + Date.now();
            const botHtml = `
                <div class="flex gap-3 max-w-[90%] md:max-w-[85%]">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-600 to-pink-600 flex items-center justify-center text-xs text-white shrink-0 shadow-md">⚡</div>
                    <div id="${botMsgId}" class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-3.5 text-sm text-slate-200 leading-relaxed shadow-sm flex-1 overflow-x-auto">
                        <span class="animate-pulse text-purple-400">Thinking...</span>
                    </div>
                </div>
            `;
            chatBox.innerHTML += botHtml;
            scrollToBottom();

            const botDiv = document.getElementById(botMsgId);
            const modelName = modelSelect.value;
            let fullReply = '';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: modelName,
                        messages: chatHistory
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                botDiv.innerHTML = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') break;
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.content) {
                                    fullReply += parsed.content;
                                    botDiv.innerHTML = formatMarkdown(fullReply);
                                    scrollToBottom();
                                }
                            } catch (err) {}
                        }
                    }
                }
                chatHistory.push({ role: 'assistant', content: fullReply });
            } catch (error) {
                botDiv.innerHTML = `<span class="text-red-400">⚠️ Error connecting to AI model: ${error.message}</span>`;
            }

            userInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.classList.remove('opacity-50');
            userInput.focus();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_TEMPLATE

@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    model = body.get("model", "qwen2.5-coder:7b")
    messages = body.get("messages", [])

    async def event_generator():
        ollama_payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
            "options": {"temperature": 0.6}
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=ollama_payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                json_data = json.dumps({"content": content})
                                yield f"data: {json_data}\n\n"
                        except Exception:
                            pass
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
