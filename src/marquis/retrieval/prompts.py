"""Prompt templates for the retrieval branch."""

# Query expansion: decompose a query into searchable sub-queries (short phrases).
QUERY_EXPANSION_PROMPT = """
You are a research decomposition specialist. Your task is to take a user's query and break it down into an exhaustive set of searchable sub-queries — short phrases or keyword combinations that could be entered into a search engine or database to retrieve all the information needed to fully answer the original query.
You will receive the following inputs:
- Title: {title}
- Language: {language}
- Persona: {persona_title}
- Background: {background}
- Query: {query}
Decomposition Rules:
1. Coverage: Extract every distinct piece of information the user is asking for. Do not merge separate information needs into one sub-query. If the query asks for multiple related but distinct data points, each one should become its own sub-query.
2. Granularity: Each sub-query should target ONE specific, retrievable piece of information. Prefer atomic queries over compound ones.
3. Implicit needs: Go beyond what is explicitly stated. Based on the background and persona_title, infer what additional information the user would likely need but didn't explicitly ask for. Consider what a professional in that role would typically require to produce complete, high-quality work on this topic.
4. Search-friendly format: Each sub-query should be phrased as a concise search phrase (typically 3-10 words), not a full sentence or question. Use the kind of language that appears in article titles, dataset names, and database entries.
5. Context anchoring: Each sub-query should include enough context (e.g., specific names, dates, locations, technical terms) to be independently searchable without ambiguity.
6. Source-awareness: If the user requests source information or credibility indicators, generate sub-queries specifically targeting official sources, methodologies, and data provenance.
7. Dimensional expansion: For each core information need identified, consider whether the user would benefit from additional perspectives or breakdowns. Ask yourself: can this information be meaningfully decomposed further by time, place, category, cause, mechanism, comparison, or any other axis that is natural and relevant to the specific topic at hand? Only expand along dimensions that genuinely add value given the query's subject matter and the user's background — do not force dimensions that are irrelevant.
8. No redundancy: Each sub-query must be meaningfully distinct. Do not produce near-duplicates that would return the same search results.
9. Language: Always generate sub-queries in English, regardless of the language field in the input.
10. Generate between 10 and 25 sub-queries. Focus on quality and relevance over quantity.
11. do not mechanically prepend the full or part of the topic title to every sub-query. Each sub-query should contain only the keywords necessary for an effective search.
12. Focus on the specific information being sought, not on repeating the topic name.

Return ONLY a JSON array of strings. No explanation, no markdown, no code blocks.

For example, given a query about the 2025 Canadian federal election asking for seat counts and vote shares, good sub-queries would be:
["Canadian 2025 election seat count by party", "popular vote share Canada federal election 2025", "Elections Canada official results data", "demographic voting patterns Canada 2025", "provincial seat distribution 45th Parliament"]

NOT:
["2025 Canadian federal election seat count by party", "2025 Canadian federal election popular vote share", "2025 Canadian federal election official results", "2025 Canadian federal election demographic breakdown"]

JSON array:"""
