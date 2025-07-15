const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(text, sender) {
  const msg = document.createElement('div');
  msg.classList.add('message', sender);

  const bubble = document.createElement('div');
  bubble.classList.add('bubble');
  bubble.textContent = text;

  const icon = document.createElement('div');
  icon.classList.add('icon');
  const iconElem = document.createElement('i');
  iconElem.classList.add('fa-solid', sender === 'user' ? 'fa-user' : 'fa-robot');
  icon.appendChild(iconElem);

  if (sender === 'user') {
    msg.appendChild(bubble);
    msg.appendChild(icon);
  } else {
    msg.appendChild(icon);
    msg.appendChild(bubble);
  }

  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msg;
}

function appendTypingIndicator() {
  const msg = document.createElement('div');
  msg.classList.add('message', 'bot');
  msg.id = 'typing-indicator';

  const icon = document.createElement('div');
  icon.classList.add('icon');
  const iconElem = document.createElement('i');
  iconElem.classList.add('fa-solid', 'fa-robot');
  icon.appendChild(iconElem);

  const bubble = document.createElement('div');
  bubble.classList.add('bubble', 'typing');
  bubble.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';

  msg.appendChild(icon);
  msg.appendChild(bubble);
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendQuery() {
  const query = userInput.value.trim();
  if (!query) return;
  appendMessage(query, 'user');
  userInput.value = '';

  // Show typing indicator
  appendTypingIndicator();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await response.json();

    // Remove typing indicator
    const typingEl = document.getElementById('typing-indicator');
    if (typingEl) typingEl.remove();

    appendMessage(data.answer, 'bot');
  } catch (err) {
    console.error(err);
    const typingEl = document.getElementById('typing-indicator');
    if (typingEl) typingEl.remove();
    appendMessage('Error fetching response.', 'bot');
  }
}

sendBtn.addEventListener('click', sendQuery);
userInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendQuery(); });