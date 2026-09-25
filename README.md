# [ArXiv Troller](https://arxiv-troller.com/)

![Alt text](assets/search.png?raw=true "Arxiv Troller Search")

A web application for discovering, organizing, and tracking machine learning papers from arXiv. Built as a successor to [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) with semantic search capabilities and personal paper management. Another way to think of it is as a simpler, less-featured [Zeta Alpha](https://www.zeta-alpha.com/) tool.

## Overview

This tool helps researchers manage the overwhelming flow of papers on arXiv by providing:

- **Semantic similarity search** using embeddings to find papers related to your interests
- **Personal tagging system** to organize papers into custom collections
- **Flexible filtering** by date, category, and keywords

## Key Features

### Search Modes

The application supports four complementary search approaches:

**Keyword Search**: Full-text search over titles and abstracts, with date and category filters

**Title Search**: Substring search over titles (`title: ...` in the search box)

**Single Paper Similarity**: Find papers semantically similar to a specific paper 

**Tag-Based Discovery**: Search for papers similar to an entire collection you've tagged

### Paper Management

**Tags**: Create custom collections of papers (e.g., "reinforcement learning theory", "Vision SNN"). Papers can belong to multiple tags.

**Organized Views**: View your tagged papers alongside search results

### Filtering

All search modes support filtering by:
- **Time period**: Last day, 3 days, week, month, 3/6/12 months, or all time
- **arXiv category**: Filter to specific research areas (cs.LG, cs.AI, etc.)

All searches default to the last week.

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

## API

A JSON API at `/api/ingestion/` exposes the same search and tags as the site. It uses the site's session login: send the `sessionid` cookie, and for POSTs also the `csrftoken` cookie plus an `X-CSRFToken` header. Pick an operation with `action`. Reads are GET with query parameters; writes are POST with a JSON body.

| Action | Method | Parameters | Returns |
|---|---|---|---|
| `tags` | GET | | your tags: `[{id, name}]` |
| `tag` | GET | `tag`, `cursor` | papers in a tag, by arXiv ID, 100 per page |
| `papers` | GET | `since`, `cursor` | papers created or updated since `since`, newest first, 100 per page |
| `search` | GET | `type`, then `q` (keyword, title), `paper` (arXiv ID) or `tag`; optional `since` or `date_filter`, `category`, `cursor` | the site's search, 20 per page, up to 400 results |
| `similar` | GET | `tag`, `since`, `cursor` | for five tagged papers per page, each one's 20 nearest papers created since `since` |
| `bulk_add` | POST | `tag`, `arxiv_ids` (up to 200) | adds papers to a tag, creating it if needed |
| `bulk_remove` | POST | `tag`, `arxiv_ids` (up to 200) | removes papers from a tag |
| `copy_tag` | POST | `source`, `target` | adds every paper in `source` to `target` |

- `search` types match the site's search modes: `keyword`, `title`, `paper` (similar to one paper) and `tag` (similar to a tag's papers, never returning papers already in the tag). Tag search samples the tag's papers randomly, so repeated calls can differ.
- `since` is an ISO 8601 time with a timezone, e.g. `2026-09-01T00:00:00+00:00`. For `search`, pass either `since` or a site `date_filter` (`1day`, `3day`, `1week`, `1month`, `3months`, `6months`, `1year`, `2years`, `all`); the default is `1week`.
- Papers are returned as `{arxiv_id, title, abstract, created, updated, categories}`.
- Lists include `next_cursor`: pass it back as `cursor` for the next page. It is `null` on the last page.
- Writes return `{tag, count, missing}`, where `missing` lists arXiv IDs that aren't in the database.
- Errors return `{"ok": false, "error": ...}` with status 400 (bad input), 401 (not logged in) or 405 (wrong method).

```bash
curl -b "sessionid=$SESSION" "https://arxiv-troller.com/api/ingestion/?action=search&type=tag&tag=backbones&since=2026-09-01T00:00:00%2B00:00"
```

## Local Installation

```bash
git clone https://github.com/igm503/arxiv-troller.git
cd arxiv-troller
# optional: create a virtual environment
# conda
pip install -r requirements.txt
python manage.py migrate
python manage.py harvest_records
python manage.py add_voyage4_embeddings  # or add_voyage3_embeddings / add_gemini_embeddings
```

### Made with

- [**Django**](https://github.com/django/django) for the web framework
- [**PostgreSQL**](https://github.com/postgres/postgres) with [**PGVector**](https://github.com/pgvector/pgvector) for embedding storage and similarity search

## Todo

- Citation-based search to discover papers through reference networks
- Way to indicate paper has already been seen
- Use hidden papers to improve search results
- recommendation generation
- allow actual tag filtering
