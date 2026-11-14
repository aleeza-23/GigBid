---
id: "ai-p11-case"
project_id: "ai-p11"
persona: "ai_ml_engineer"
type: "case_study"
project: "LLM evaluation harness for prompt regressions"
skills: ["Python", "AI Model Development", "OpenAI API"]
date: "2024-01-05"
client_industry: "AI Startup"
outcome: "delivered"
data_origin: "synthetic"
---
# LLM evaluation harness for prompt regressions

*SYNTHETIC - generated for AI-4016 GigBid; not a real person or client.*

Client: an ai startup company (2024).

## Problem
The prompt changes shipped without tests and silently broke answers.

## What I did
I built a 300-item golden set, an LLM judge calibrated against human labels, and a CI gate.

## Result
The judge agreed with human labels at kappa 0.72 and caught 11 regressions before release.

## Tools
Python, AI Model Development, OpenAI API
