# [ArXiv Troler](https://arxiv-troller.com/)

![Alt text](assets/search.png?raw=true "Arxiv Troller Search")

A web application for discovering, organizing, and tracking machine learning papers from arXiv. Built as a successor to [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) with semantic search capabilities and personal paper management. Another way to think of it is as a simpler, less-featured [Zeta Alpha](https://www.zeta-alpha.com/) tool.

## Overview

This tool helps researchers manage the overwhelming flow of papers on arXiv by providing:

- **Semantic similarity search** using embeddings to find papers related to your interests
- **Personal tagging system** to organize papers into custom collections
- **Flexible filtering** by date, category, and keywords

## Key Features

### Search Modes

The application supports three complementary search approaches:

**Keyword Search**: Traditional title-based search with date and category filters

**Single Paper Similarity**: Find papers semantically similar to a specific paper 

**Tag-Based Discovery**: Search for papers similar to an entire collection you've tagged

### Paper Management

**Tags**: Create custom collections of papers (e.g., "reinforcement learning theory", "Vision SNN"). Papers can belong to multiple tags.

**Organized Views**: View your tagged papers alongside search results

### Filtering

All search modes support filtering by:
- **Time period**: Last week, month, 3/6/12/24 months, or all time
- **arXiv category**: Filter to specific research areas (cs.LG, cs.AI, etc.)

Default filters differ by search type. Keyword searches default to "all time" since you're looking for specific content. Similarity searches default to "last week" since you're typically looking for recent related work.

## How It Works

### Data Pipeline

Papers are imported from arXiv with their metadata (title, authors, abstract, categories, publication date). Abstracts are embedded using a semantic embedding model.

PGVector is used to store and search among the embeddings in a PostgreSQL database.

### Similarity Search

When you search for papers similar to a single paper, the system:
1. Retrieves the embedding for the source paper
2. Excludes the source paper from results
3. Applies date and category filters to the candidate set
4. Computes distances between the source embedding and candidates
5. Returns the closest matches ordered by semantic similarity

When searching based on a tag collection, the system queries for similar papers from each tagged paper independently, then interleaves the results. This prevents one paper from dominating recommendations. 

## Local Installation

```bash
git clone https://github.com/igm503/arxiv-troller.git
cd arxiv-troller
# optional: create a virtual environment
# conda
pip install -r requirements.txt
python manage.py migrate
python manage.py harvest_records
python manage.py add_embeddings
python manage.py add_citations
```

## Technical Stack

- [**Django**](https://github.com/django/django) for the web framework
- [**PostgreSQL**](https://github.com/postgres/postgres) with [**PGVector**](https://github.com/pgvector/pgvector) for embedding storage and similarity search

## Future Directions

Planned enhancements include:
- Enhanced tag management features
- Citation-based search to discover papers through reference networks
