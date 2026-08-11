# Dual retrieval paths

`marquis.retrieval` contains two separate retrieval paths that must not be
conflated.

## Path A: tevatron first-stage retrieval

Classic dense retrieval pipeline:

```
marquis-retrieve expand → prepare-subqueries → fusion/rrf → qrels → evaluate
```

Produces `.trec` run files. Used for the paper's retrieval evaluation.

## Path B: OmniEmbed dense search (QA system)

```
marquis-retrieve retrieve
```

Feeds the QA extraction system directly. Uses OmniEmbed embeddings for
dense search.

## Why it matters

- Config changes for Path A (tevatron tile sizes, fusion weights) should
  not affect Path B.
- When adding retrieval features, identify which path is the target before
  making changes.
