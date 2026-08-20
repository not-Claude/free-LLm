const messagesElement = document.getElementById("messages");
const form = document.getElementById("chat");
const input = document.getElementById("input");
const status = document.getElementById("status");

let messages = [];

function addMessage(role, text) {
    const element = document.createElement("div");
    element.className = "message " + role;
    element.textContent = text;
    messagesElement.appendChild(element);
    messagesElement.scrollTop = messagesElement.scrollHeight;
    return element;
}

async function checkServer() {
    try {
        const response = await fetch("/health");
        const data = await response.json();
        status.textContent = `RAM: ${data.ram_available_gb} GB`;
    } catch {
        status.textContent = "Сервер недоступен";
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = input.value.trim();
    if (!text) return;

    input.value = "";

    messages.push({ role: "user", content: text });
    addMessage("user", text);

    const assistant = addMessage("assistant", "Думаю...");

    try {
        const response = await fetch("/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: "local-llm",
                messages: messages,
                max_tokens: 256,
                temperature: 0.7
            })
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();
        const answer = data.choices[0].message.content;

        assistant.textContent = answer;
        messages.push({ role: "assistant", content: answer });
    } catch (error) {
        assistant.textContent = "Ошибка: " + error.message;
    }

    checkServer();
});

checkServer();
