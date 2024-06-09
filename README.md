# Run instructions

```
pip install -r requirements.txt

streamlit run main.py
```

# LLM and Embeddings:

LLM used: gpt-3.5-turbo-0125 (also used cohere command r+, and phi3:medium)

phi3:medium can't follow the default prompts in Ragas, mostly because it was trained on a very weird prompt sturcture.

For Cohere command, I was severely rate limited.

Embedding used: avsolatorio/GIST-Embedding-v0 (smallest i could find on the MTEB leaderboard)

# Dataset construction:

Corpus: [StreamLit documentation](https://docs.streamlit.io/)

Documents scraped: ~170

Using Ragas for question generation

Ragas interally generates a docstore that is filled up using embeddings and LLM calls that return keywords in that particular chunk.

Chunks are randomly selected, a base evolution (simple category) is created first that generates a seed question. All other 3 categories are complex evolutions that work on this seed question to make this more complex. Ragas can automatically fix invalid  questions. Complex evolutions may use information in the same chunk or different chunk altogether.

Question generation is expensive, 1 LLM call + 1 embedding call for the store creation per chunk, and atleast 2 LLM calls per question. The cost is justified, question generation need to generally be as complex as possible so that there's less room for error.

Distribution of evolutions:

-   simple: 0.1
-   reasoning: 0.3
-   multi_context: 0.3 
-   conditional: 0.3

Total questions generated: 37


Output format: `questions.csv` 

Attributes: `question, metadata, contexts, answer, evolution_type` 

This is a comprehensive dataset, since the nodes for question generation are randomly chosen, and we obtain different types of questiosn, I'm considering a worst case scenario having very low weightage to simple questions (10%). 

# RAG

LLMOps library used: LangChain

VectorStore used: Chroma

I first tried using normal RAG. Didn't work too well. Switched to Parent Document retriever chain using LCEL and message history. I don't think I can come up with other. Wanted to use Cohere Rerank but I get only get like 2 llm calls per min.

Based on my experience, for GPT:

System message + user history + bot message containing contexts gives highest scores, but only by a narrow margin.

# Evaluation

I don't think regular metrics like perplexity, rouge, bleu help when it comes to answers generated using RAG since RAG generated answers are highly dependant on the contexts, the structure of the prompt, and the LLM being used.

Evaluation done by Ragas.

Metrics used:

-   Answer Correctness: Uses LLM calls to determine factual similarity (score between 0-1)

-   Answer similarity: Semantic similarity using embeddings


```
Metric: answer_correctness 	Value: 0.4608059358059359
Metric: answer_relevancy 	Value: 0.8482784726386126
```