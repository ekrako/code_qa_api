import asyncio
from typing import Any

import litellm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from code_qa_api.core.config import settings

# Configure litellm routing strategy if needed, or just use model names directly
# litellm.router_strategy = "simple-shuffle"
# litellm.set_verbose=True

litellm.set_verbose = False  # Reduce litellm logging clutter


# Exponential backoff for API calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_embeddings(texts: list[str], model: str = settings.embedding_model) -> np.ndarray:
    if not texts:
        return np.array([])
    try:
        response = await litellm.aembedding(model=model, input=texts)
        embeddings = [item["embedding"] for item in response["data"]]
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embeddings with model {model}: {e}")
        raise  # Re-raise after logging


# Retry decorator for explanation generation
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def get_chunk_explanation(chunk_content: str, file_path: str, model: str = settings.chunk_explanation_model) -> str:
    system = """\
Your task is to generate clear, concise explanations of Python CODE snippets for both humans and AI to understand. Follow these guidelines:

1. Content:
   - Explain the code's purpose and functionality.
   - Use the active voice and imperative tense throughout.
   - Target junior Python developers familiar with basic concepts.
   - Scale detail based on content complexity: 100-150 words for simple files, 150-200 for moderate complexity, and 200-250 for highly complex content.
   - For larger files, provide a concise overview (2-3 sentences) before delving into specifics.

2. Python Focus:
   - Prioritize information from docstrings and important comments, integrating their essence without direct quoting.
   - Highlight use of standard library, popular third-party packages, and idioms.
   - Explain non-obvious code sections or design choices.
   - Mention use of design patterns or important Python conventions.

3. Boundaries:
   - Focus only on explaining functionality and purpose.
   - Avoid suggesting improvements or commenting on code quality.
   - Don't refer to the code in the third person.

4. Follow-up and Clarification:
   - The aim is to answer once and not drag into a follow-up call.
   - If you are not sure of the answer, it is better to answer "I don't know" according to the requests mentioned before than to ask follow-up questions.

5. Line Number References:
   - Utilize line numbers when present to provide precise references to specific parts of the content.
   - Incorporate line number references naturally into your explanation to enhance clarity and specificity.
   - When referencing multiple consecutive lines, use a range (e.g., lines 05-07).

FORMAT:
<ID>[filename.py]</ID>
<CODE>
[Python code snippet]
</CODE>

[Your explanation adhering to the above guidelines]

EXAMPLES (separated by '---'):
---
Please explain the following Python code snippet:

<ID>data_processor.py</ID>
<CODE>
01 import pandas as pd
02 import numpy as np
03
04 def clean_data(df):
05     df['date'] = pd.to_datetime(df['date'])
06     df['value'] = df['value'].fillna(df['value'].mean())
07     return df[(np.abs(df['value'] - df['value'].mean()) <= 3 * df['value'].std())]
08
09 def process_file(filepath):
10     df = pd.read_csv(filepath)
11     cleaned_df = clean_data(df)
12     result = cleaned_df.groupby('category')['value'].agg(['mean', 'median', 'std'])
13     return result
</CODE>

RESPONSE:
This code defines functions for data cleaning and processing using pandas and numpy.

Import necessary libraries on lines 01-02. The 'clean_data' function (lines 04-07) performs data preprocessing:
1. Convert 'date' column to datetime format.
2. Fill missing values in 'value' column with mean.
3. Remove outliers using 3-sigma rule.

The 'process_file' function (lines 09-13) orchestrates the data processing workflow:
1. Read CSV file into a DataFrame.
2. Clean data using 'clean_data' function.
3. Group data by 'category' and calculate mean, median, and standard deviation of 'value' column.

This script is designed for efficient data cleaning and statistical analysis, utilizing pandas for data manipulation and numpy for numerical operations. 
It implements common data preprocessing techniques and basic statistical aggregations.
---
Please explain the following Python code snippet:

<ID>web_scraper.py</ID>
<CODE>
01 import asyncio
02 import aiohttp
03 from bs4 import BeautifulSoup
04
05 async def fetch_url(session, url):
06     async with session.get(url) as response:
07         return await response.text()
08
09 async def scrape_urls(urls):
10     async with aiohttp.ClientSession() as session:
11         tasks = [fetch_url(session, url) for url in urls]
12         responses = await asyncio.gather(*tasks)
13
14     results = []
15     for html in responses:
16         soup = BeautifulSoup(html, 'html.parser')
17         title = soup.find('title').text if soup.find('title') else 'No title found'
18         results.append(title)
19
20    return results
21
22 if __name__ == "__main__":
23     urls = ["http://example.com", "http://example.org", "http://example.net"]
24     titles = asyncio.run(scrape_urls(urls))
25     for url, title in zip(urls, titles):
26        print(f"{url}: {title}")
</CODE>

RESPONSE:
This code implements an asynchronous web scraper using aiohttp and BeautifulSoup libraries.

Import required libraries for asynchronous operations and HTML parsing on lines 01-03. 
The 'fetch_url' function (lines 05-07) asynchronously retrieves HTML content from a given URL.

The 'scrape_urls' function (lines 09-20) is the core of the scraper:
1. Create an aiohttp session.
2. Concurrently fetch HTML content from multiple URLs using asyncio.gather.
3. Parse each HTML response with BeautifulSoup.
4. Extract page titles, defaulting to 'No title found' if title tag is absent.

The main block (lines 22-26) demonstrates usage:
1. Define a list of URLs to scrape.
2. Run the scraper asynchronously.
3. Print each URL with its corresponding page title.

This script showcases efficient web scraping by leveraging asynchronous I/O to handle multiple requests concurrently, 
potentially improving performance for large-scale scraping tasks.
---
Please explain the following Python code snippet:

<ID>todo_app.py</ID>
<CODE>
01 import tkinter as tk
02 from tkinter import messagebox
03 import json
04
05 class TodoApp:
06     def __init__(self, master):
07         self.master = master
08         self.master.title("Todo App")
09        self.todos = self.load_todos()
10
11        self.task_entry = tk.Entry(master, width=40)
12        self.task_entry.pack(pady=10)
13
14        self.add_button = tk.Button(master, text="Add Task", command=self.add_task)
15        self.add_button.pack()
16
17        self.task_listbox = tk.listbox(master, width=50)
18        self.task_listbox.pack(pady=10)
19
20         self.remove_button = tk.Button(master, text="Remove Task", command=self.remove_task)
21         self.remove_button.pack()
22
23         self.update_listbox()
24
25     def add_task(self):
26         task = self.task_entry.get()
27         if task:
28             self.todos.append(task)
29             self.update_listbox()
30             self.task_entry.delete(0, tk.END)
31             self.save_todos()
32         else:
33             messagebox.showwarning("Warning", "Please enter a task.")
34
35     def remove_task(self):
36         try:
37             index = self.task_listbox.curselection()[0]
38             del self.todos[index]
39             self.update_listbox()
40             self.save_todos()
41        except IndexError:
42            messagebox.showwarning("Warning", "Please select a task to remove.")
43
44    def update_listbox(self):
45        self.task_listbox.delete(0, tk.END)
46        for task in self.todos:
47            self.task_listbox.insert(tk.END, task)
48
49    def load_todos(self):
50        try:
51            with open('todos.json', 'r') as f:
52                return json.load(f)
53        except FileNotFoundError:
54            return []
55
56    def save_todos(self):
57        with open('todos.json', 'w') as f:
58            json.dump(self.todos, f)
59
60 if __name__ == "__main__":
61     root = tk.Tk()
62     app = TodoApp(root)
63     root.mainloop()
</CODE>

RESPONSE:
This code implements a simple Todo application using Tkinter for the GUI and JSON for data persistence.

Import necessary libraries for GUI, message boxes, and JSON handling on lines 01-03. 
The TodoApp class (lines 05-58) encapsulates the application's functionality:

1. Initialize the app (lines 06-23):
  - Set up the main window and title.
  - Load existing todos from a JSON file.
  - Create and pack GUI elements (entry field, buttons, listbox).

2. Task management methods:
  - add_task (lines 25-33): Add new task to the list and update GUI.
  - remove_task (lines 35-42): Remove selected task from the list and update GUI.
  - update_listbox (lines 44-47): Refresh the listbox with current tasks.

3. Data persistence methods:
  - load_todos (lines 49-54): Load tasks from 'todos.json' file, return empty list if file not found.
  - save_todos (lines 56-58): Save current tasks to 'todos.json' file.

The main block (lines 60-63) creates the Tkinter root window, instantiates the TodoApp, and starts the main event loop.

This script demonstrates a basic MVC pattern, separating the data model (todo list) from the view (Tkinter GUI) and controller (TodoApp class methods). 
It provides a user-friendly interface for managing tasks with persistent storage.
---
"""
    prompt = f"""
        f"Explain the following Python code snippet from the file '{file_path}'. "
        <CODE>
        {chunk_content}
        </CODE>
"""

    try:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        response = await litellm.acompletion(model=model, messages=messages, max_tokens=100)
        explanation = response.choices[0].message.content.strip()
        return str(explanation)
    except Exception as e:
        print(f"Error getting chunk explanation with model {model} for {file_path}: {e}")
        # Return a default explanation or raise the error
        return "Could not generate explanation."


async def process_chunks_for_embedding(chunks: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not chunks:
        return np.array([]), []

    # Generate explanations concurrently
    explanation_tasks = [get_chunk_explanation(chunk["content"], chunk["file_path"]) for chunk in chunks]
    explanations = await asyncio.gather(*explanation_tasks)

    texts_to_embed = []
    metadata_list = []

    for i, chunk in enumerate(chunks):
        explanation = explanations[i]
        content = chunk["content"]
        file_path = chunk["file_path"]

        # Combine content and explanation for embedding
        combined_text = f"File: {file_path}\nExplanation: {explanation}\nCode:\n{content}"
        texts_to_embed.append(combined_text)

        # Store metadata (without the potentially large combined text)
        metadata = chunk.copy()
        metadata["explanation"] = explanation  # Add the generated explanation
        # Remove None values from metadata to ensure clean data
        metadata = {k: v for k, v in metadata.items() if v is not None}
        metadata_list.append(metadata)

    if not texts_to_embed:
        return np.array([]), []

    # Get embeddings for the combined texts
    embeddings = await get_embeddings(texts_to_embed)

    return embeddings, metadata_list
