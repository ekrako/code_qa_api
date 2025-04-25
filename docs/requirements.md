# Modelcode GenAI Coding Exercise: QA about Code

Your goal is to write a **REST API** that answers questions about a local repo. As an example, you can take the following repo:
https://drive.google.com/file/d/1s0IgVa_zNEa4Ge9SDL0V-2qDCr23wuLX/view?usp=sharing.

Given the root path of a Python project, your API should take in questions in natural language and answer them as text (potentially with code snippets). The questions should be processed independently of each other (not as a chat). The questions might include things like:
- What does class X do?
- How is service Y implemented?
- How does method X use parameter Y?

To implement this functionality, you should use a **RAG system** where you will define and implement the various aspects such as chunking of the code, storage, indexing and retrieval. You can use existing libraries for any part of the project, as long as you can explain your choices. Your chunks should reflect coherent logical blocks of code (e.g. functions, classes, …) and not just be character/token based.

Your code should support **large repos** that could not fit in a model's context window.

You should implement a way to **automatically measure the quality of your system**. For that purpose, we provide a set of 10 question/answer pairs at https://github.com/Modelcode-ai/grip_qa. These can be considered references to measure against (of course, it is not required for each answer to be strictly identical to the reference one in order to be correct). Your evaluation script should be able to run your implementation of the QA system on the set of questions, compare the answers to the reference one and provide a numeric score capturing the quality of the system's answers. It's up to you to define the details of the approach and metrics you use.

Once you complete the requirements above, as a bonus, you can extend your submission to implement additional features, such as creating a tool-using agent that leverages the RAG, implementing a client application to your RAG service, etc.

Write your program in any language and use any LLM available (e.g. OpenAI's GPT-4o, Anthropic's Claude, Llama, …), whether through an API or locally. If you need a temporary account with OpenAI for the purpose of this exercise, contact Antoine (araux@modelcode.ai).

At the follow up interview, be ready to demonstrate your submission, explain your approach, describe the main challenges and how you overcame them, and discuss possible extensions to the project.