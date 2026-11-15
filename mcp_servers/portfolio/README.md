# Portfolio MCP Server

Read-only MCP access to a freelancer's portfolio: case studies, repository READMEs and
code, client reviews, a CV, and past proposals labelled with whether they won.

Built for [GigBid](../../README.md), but it depends on nothing in GigBid. It reads a
directory of portfolio files and an index, so you can point it at your own work and use it
in Claude Desktop, Cursor, or any MCP host.

## Why you might want it

Searching your own portfolio is a surprisingly bad experience. The same project exists as
a case study, a README, a commit history and a client review, and none of them is the
version you need at the moment a client asks "have you done this before?". This server
answers that question with the specific item, its date, and whether the proposal that used
it actually won.

It also tells you what you *cannot* claim. `list_skills` returns `unbacked_skills`: skills
on your profile with no evidence behind them. That list is usually uncomfortable and
always useful.

## Install

Three ways, depending on how permanent you want it.

**uvx**, nothing installed permanently:

```bash
uvx --from git+https://github.com/<your-username>/gigbid gigbid-portfolio-mcp
```

**Docker**, pointed at your own portfolio:

```bash
docker build -f infra/docker/Dockerfile.portfolio-mcp -t gigbid-portfolio-mcp .
docker run --rm -p 8701:8701 -v /path/to/your_portfolio:/portfolio:ro -e PERSONAS_DIR=/portfolio gigbid-portfolio-mcp
```

The image carries only this server: no torch, no sentence-transformers, no LangGraph,
because the default ranker is BM25. That is about 150MB instead of several gigabytes. Run
it without the volume to try the tools against the synthetic corpora in this repository.

**From a clone**, for development:

```bash
git clone https://github.com/<your-username>/gigbid
cd gigbid
pip install -e .
```

Then add it to your host. For Claude Desktop, in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "portfolio": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/<your-username>/gigbid",
               "gigbid-portfolio-mcp"],
      "env": { "PERSONAS_DIR": "/absolute/path/to/your_portfolio" }
    }
  }
}
```

For Cursor, the same block in `.cursor/mcp.json`. To check it before wiring a host:

```bash
npx @modelcontextprotocol/inspector python -m mcp_servers.portfolio.server
```

Streamable HTTP, if you want it over the network:

```bash
python -m mcp_servers.portfolio.server --http --port 8701
```

## Tools

| Tool | What it does |
|---|---|
| `search_evidence(requirement, persona, k, filters, outcome_aware)` | Evidence for **one** requirement. Filter by type, skill, date, industry or proposal outcome. `outcome_aware=true` prefers phrasing from proposals that won. |
| `get_case_study(case_id, persona)` | One case study in full, plus the repo, code and review for the same project. Takes an evidence id (`ai-p01-case`) or a project id (`ai-p01`). |
| `list_skills(persona, min_evidence)` | Skills with evidence counts, won-proposal counts, your rate and floor, and `unbacked_skills`. |
| `portfolio_health()` | Is the corpus loadable, and what is indexed. |

Search one requirement at a time. A query blending five requirements returns the average
of every project and nothing precisely — the corpus describes each project several ways on
purpose, and that is the weakness to work with rather than around.

## Resources

| URI | Returns |
|---|---|
| `portfolio://case/{case_id}` | One case study with provenance |
| `portfolio://proposal/{proposal_id}` | One past proposal, what it cited, whether it won |
| `portfolio://project/{project_id}` | Every description of one project at once |
| `portfolio://profile/{persona}` | Rates, floor, availability, declared skills |
| `portfolio://wins/{persona}` | The proposals that won, newest first |

## Prompts

`answer_screening_question`, `prove_requirement`, `draft_proposal_paragraph`,
`should_i_bid`, `portfolio_gaps`.

Each one constrains the answer to retrieved evidence and tells the model to say so plainly
when the portfolio does not support a claim. That instruction lives in the prompt because
these are installable in hosts with nothing downstream to catch a fabricated claim.

## Using your own portfolio

Point `PERSONAS_DIR` at your own directory, laid out like this:

```
your_portfolio/
├── evidence_index.json          # the catalogue (see below)
└── ai_ml_engineer/              # must be one of the three names below
    ├── profile.json             # rates, floor, skills, availability
    ├── cv.md
    ├── case_studies/*.md        # YAML frontmatter + prose
    ├── repos/<name>/README.md
    ├── repos/<name>/*.py
    ├── reviews.json             # [{id, project_id, rating, text, date}]
    └── proposals.json           # [{id, text, outcome, cited_project_ids, date}]
```

The `persona` argument is a closed enum — `ai_ml_engineer`, `fullstack_web_developer`,
`data_analyst` — so your folder has to use whichever of those three fits you, or you edit
`PersonaName` in [`tools.py`](tools.py) and re-record the schema snapshot with
`make snapshot-update`. The enum is deliberate rather than lazy: a closed set stops a model
inventing a persona name, and it is one of the things R&D question A measures.

`evidence_index.json` is a flat list, one entry per item:

```json
{
  "id": "ai-p01-case",
  "project_id": "ai-p01",
  "persona": "your_name",
  "type": "case_study",
  "project": "Support ticket triage classifier",
  "skills": ["Machine Learning", "Python", "scikit-learn"],
  "date": "2023-12-15",
  "client_industry": "SaaS",
  "outcome": "delivered",
  "path": "case_studies/ai-p01.md"
}
```

`type` is one of `case_study`, `proposal`, `review`, `readme`, `code`, `cv`. For a
proposal, `outcome` is `won`, `replied` or `ignored`, and `cited_project_ids` names the
projects it argued from — that link is what makes "I have won work with Celery" a
checkable statement.

## Ranking

The default ranker is BM25 over item text plus an exact-match bonus on indexed skills.
BM25 rather than embeddings because this corpus turns on exact library names: a job asking
for Celery should surface the project that used Celery, and a dense embedding blurs
"Celery" into "task queue".

If `rag.retriever.get_ranker` exists it is used instead, which is how GigBid swaps in the
retrieval design its own evaluation picked. The tool signature does not change either way.

## Tests

```bash
pytest mcp_servers/portfolio/tests -q
```

15 tests, all through an in-memory MCP client rather than against the Python functions,
because a server whose functions work but whose schema is malformed is broken for every
host and only a protocol-level test notices.

## Licence and data

Code: MIT. The persona corpora shipped in this repository are synthetic, generated for
AI-4016, and describe no real person or client.
