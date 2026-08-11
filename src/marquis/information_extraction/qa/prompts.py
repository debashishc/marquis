"""Prompt contracts for QA-based information extraction."""

from __future__ import annotations

QA_DECOMPOSITION_PROMPT_TEMPLATE = """
You are a research decomposition specialist. Your task is to take a user's query
and break it down into an exhaustive set of searchable research questions -
complete questions that could be used to retrieve all the information needed to
fully answer the original query.

Return ONLY a JSON array of strings. No explanation, no markdown, and no code blocks.
""".strip()

QA_ANSWER_PROMPT_TEMPLATE = """
Question:
{question}

Transcript:
{transcript}

Answer concisely using the video and transcript.

Return ONLY the final factual answer.
""".strip()

COMBINE_ANSWERS_PROMPT_TEMPLATE = """
You are given extracted answers from videos. These answers are factual and must be used.

Question: {subquery}

Extracted Answers (treat as ground truth):
{valid_answers}

Combine them into a single answer. Do NOT use prior knowledge.
""".strip()

FOLLOWUP_PROMPT_TEMPLATE = """
You are refining a research question based on prior answers.

Context:
{context}

Generate ONE new question that extracts new information not yet covered.
If no meaningful new question can be formed, output:
NONE

Return ONLY the question or NONE.
""".strip()
