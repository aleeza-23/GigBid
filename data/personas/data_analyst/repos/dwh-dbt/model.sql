-- dwh-dbt: example model (synthetic). Uses dbt, PostgreSQL, Airbyte.
{{ config(materialized='table') }}
with source as (select * from {{ ref('stg_events') }}),
final as (select date_trunc('month', event_at) as month, count(distinct user_id) as users
          from source group by 1)
select * from final
