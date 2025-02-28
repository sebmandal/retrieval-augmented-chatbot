# Retrieval-augmented chatbot starter

This is a simplified starter template package for a project that uses OpenAI embeddings and chat completions. It reads text from a file, splits it into chunks, creates embeddings, and starts an interactive chat session that finds the most similar text passages.

## Setup

1. **Install dependencies:**

   ```bash
   pip install openai
   ```

2. **Configure your API key:**

   Set your OpenAI API key as an environment variable:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Prepare your data:**

   Place your input text in `data/input.txt`.

## Usage

Run the main script:

```bash
python -m rag_starter.main
```

The script will prompt you to decide whether to overwrite existing embeddings, then enter into an interactive chat session.

Happy coding!
