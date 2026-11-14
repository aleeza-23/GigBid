---
id: "ai-p02-case"
project_id: "ai-p02"
persona: "ai_ml_engineer"
type: "case_study"
project: "Retrieval-augmented assistant over insurance policy documents"
skills: ["Python", "OpenAI API", "LangChain", "Natural Language Processing"]
date: "2024-01-05"
client_industry: "Insurance"
outcome: "delivered"
data_origin: "synthetic"
---
# Retrieval-augmented assistant over insurance policy documents

*SYNTHETIC - generated for AI-4016 GigBid; not a real person or client.*

Client: an insurance company (2024).

## Problem
The claims agents searched roughly 600 PDF policies to answer coverage questions.

## What I did
I chunked and embedded the policies, combined keyword and vector search, and forced every answer to cite the clause it came from.

## Result
Average answer time dropped from 6 minutes to 40 seconds with 92% citation precision on a 150-question test set.

## Tools
Python, OpenAI API, LangChain, Natural Language Processing
