# Worker

`scheduler.py` polls saved searches on the configured schedule and delegates new jobs to
the graph. `tasks.py` keeps a persistent deduplication ledger so repeated sweeps do not
analyse reposts or duplicate jobs twice. Run `make worker-once` for a single sweep,
`make worker` for the scheduler, and `make worker-stats` to inspect the ledger.

The current worker uses the repository's lightweight scheduler and SQLite ledger rather
than Celery, Arq or Dramatiq. In the Docker stack, run one worker replica so two workers
cannot both pass the seen check before either records the job.
