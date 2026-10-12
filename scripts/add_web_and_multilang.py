"""Web Development (HTML/CSS Landing Pages), Multi-Language Coding, and Multilingual Dataset."""

import os
import json

WEB_AND_MULTILANG_DATA = [
    # 1. HTML5 & CSS Landing Pages
    {
        "prompt": "Create a simple HTML and CSS landing page for a website.",
        "response": """```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Ari-LLM Landing Page</title>
  <style>
    body { font-family: sans-serif; margin: 0; background: #0f172a; color: #fff; text-align: center; }
    header { padding: 60px 20px; }
    h1 { font-size: 42px; margin-bottom: 12px; }
    p { font-size: 18px; color: #94a3b8; max-width: 600px; margin: 0 auto 30px; }
    .cta-btn { background: #3b82f6; color: white; padding: 12px 28px; border-radius: 8px; text-decoration: none; font-weight: bold; }
    .features { display: flex; justify-content: center; gap: 20px; margin-top: 50px; }
    .card { background: #1e293b; padding: 24px; border-radius: 12px; width: 220px; }
  </style>
</head>
<body>
  <header>
    <h1>Supercharge Your Workflow</h1>
    <p>Ari-LLM is the fast, local AI model running directly on your CPU with zero external dependencies.</p>
    <a href="#get-started" class="cta-btn">Get Started Free</a>
  </header>
  <div class="features">
    <div class="card"><h3>Ultra-Fast</h3><p>90+ tokens/second on CPU</p></div>
    <div class="card"><h3>Private</h3><p>100% offline and secure</p></div>
    <div class="card"><h3>Open</h3><p>Trained from scratch</p></div>
  </div>
</body>
</html>
```"""
    },
    {
        "prompt": "Write a clean HTML and CSS hero section.",
        "response": """```html
<section style="background: #111827; color: #f9fafb; padding: 80px 24px; text-align: center;">
  <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 16px;">Next-Gen Local AI</h1>
  <p style="font-size: 1.25rem; color: #9ca3af; margin-bottom: 32px;">Experience private, sub-15ms AI inference directly on your machine.</p>
  <button style="background: #2563eb; color: white; font-size: 1rem; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer;">Explore Docs</button>
</section>
```"""
    },
    {
        "prompt": "How do I create a responsive flexbox card in CSS?",
        "response": """```css
.card-container {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  justify-content: center;
}

.card {
  flex: 1 1 300px;
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 20px;
}
```"""
    },

    # 2. JavaScript / Web Development
    {
        "prompt": "Write a JavaScript function to debounce a call.",
        "response": """```javascript
function debounce(func, delay = 300) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}
```"""
    },
    {
        "prompt": "How do I make an async fetch request in JavaScript with error handling?",
        "response": """```javascript
async function fetchData(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Fetch failed:", error);
    return null;
  }
}
```"""
    },
    {
        "prompt": "How do I add a click event listener in JavaScript?",
        "response": """```javascript
const button = document.querySelector("#my-btn");
button.addEventListener("click", () => {
  alert("Button was clicked!");
});
```"""
    },

    # 3. SQL Database Queries
    {
        "prompt": "Write a SQL query to find employees with salary greater than 50000.",
        "response": """```sql
SELECT employee_id, first_name, last_name, salary, department
FROM employees
WHERE salary > 50000
ORDER BY salary DESC;
```"""
    },
    {
        "prompt": "Write a SQL query joining customers and orders.",
        "response": """```sql
SELECT 
    c.customer_id,
    c.customer_name,
    o.order_id,
    o.order_date,
    o.total_amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2025-01-01';
```"""
    },
    {
        "prompt": "How do I count orders per customer in SQL?",
        "response": """```sql
SELECT customer_id, COUNT(*) AS total_orders
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 5;
```"""
    },

    # 4. C Systems Programming
    {
        "prompt": "Write a C function to reverse an array.",
        "response": """```c
void reverse_array(int arr[], int size) {
    int start = 0;
    int end = size - 1;
    while (start < end) {
        int temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}
```"""
    },
    {
        "prompt": "How do I allocate memory dynamically in C?",
        "response": """```c
#include <stdio.h>
#include <stdlib.h>

int* create_array(int size) {
    int* arr = (int*)malloc(size * sizeof(int));
    if (arr == NULL) {
        perror("Allocation failed");
        return NULL;
    }
    return arr;
}
```"""
    },

    # 5. Multilingual Knowledge & Spanish / French Translation
    {
        "prompt": "How do you say hello in Spanish?",
        "response": "In Spanish, you say 'hello' as 'Hola'. For a more formal greeting, you can say 'Buenos días' (Good morning) or 'Buenas tardes' (Good afternoon)."
    },
    {
        "prompt": "How do you say thank you in French?",
        "response": "In French, 'thank you' is 'Merci'. To say 'thank you very much', say 'Merci beaucoup'."
    },
    {
        "prompt": "Translate 'Hello, how are you?' into Spanish and French.",
        "response": "- Spanish: 'Hola, ¿cómo estás?'\n- French: 'Bonjour, comment allez-vous?'"
    },
    {
        "prompt": "What is photosynthesis?",
        "response": "Photosynthesis is the biological process by which green plants, algae, and cyanobacteria convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll."
    },
]

def append_web_and_multilang():
    sft_file = "data/processed/sft_conversational_v2.jsonl"
    with open(sft_file, "a", encoding="utf-8") as f:
        # Repeat to ensure sufficient gradient weighting
        for _ in range(40):
            for item in WEB_AND_MULTILANG_DATA:
                f.write(json.dumps(item) + "\n")

    print(f"Appended web dev & multi-language data to {sft_file}")
    with open(sft_file, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"Total SFT examples now: {total}")

if __name__ == "__main__":
    append_web_and_multilang()
