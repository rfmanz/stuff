-- Query for Socure data pull
with map as (select distinct thirdparty_raw_id, event_id
             from decision)
select e.user_id,
       tpr.created_dt,
       case when tpr.response::jsonb->'fraud'#>'{scores, 0}'->>'version'='3.0' then tpr.response::json->'fraud'#>'{scores, 0}'->>'score' else tpr.response::json->'fraud'#>'{scores, 1}'->>'score' end as fraud_score_1_v1,
       case when tpr.response::jsonb->'fraud'#>'{scores, 0}'->>'version'='1.0' then tpr.response::json->'fraud'#>'{scores, 0}'->>'score' else tpr.response::json->'fraud'#>'{scores, 1}'->>'score' end as fraud_score_2_v1,
       case when tpr.response::jsonb->'fraud'#>'{scores, 0}'->>'name'='generic' then tpr.response::json->'fraud'#>'{scores, 0}'->>'score' 
          when tpr.response::jsonb->'fraud'#>'{scores, 1}'->>'name'='generic' then tpr.response::json->'fraud'#>'{scores, 1}'->>'score'
          else null end as fraud_score_1,
       case when tpr.response::jsonb->'fraud'#>'{scores, 0}'->>'name'='generic' then tpr.response::json->'fraud'#>'{scores, 0}'->>'version' 
          when tpr.response::jsonb->'fraud'#>'{scores, 1}'->>'name'='generic' then tpr.response::json->'fraud'#>'{scores, 1}'->>'version'
          else null end as fraud_score_1_version,
       case when tpr.response::jsonb->'fraud'#>'{scores, 0}'->>'name'='sigma' then tpr.response::json->'fraud'#>'{scores, 0}'->>'score' 
          when tpr.response::jsonb->'fraud'#>'{scores, 1}'->>'name'='sigma' then tpr.response::json->'fraud'#>'{scores, 1}'->>'score'
          else null end as fraud_score_2,
       case when tpr.response::jsonb->'fraud'#>'{scores, 0}'->>'name'='sigma' then tpr.response::json->'fraud'#>'{scores, 0}'->>'version' 
          when tpr.response::jsonb->'fraud'#>'{scores, 1}'->>'name'='sigma' then tpr.response::json->'fraud'#>'{scores, 1}'->>'version'
          else null end as fraud_score_2_version,
       tpr.response::jsonb->'addressRisk'->>'score' as address_risk_score,        
       tpr.response::jsonb->'emailRisk'->>'score' as email_risk_score,
       tpr.response::jsonb->'phoneRisk'->>'score' as phone_risk_score,
       tpr.response::jsonb->'nameAddressCorrelation'->>'score' as name_address_correlation,
       tpr.response::jsonb->'nameEmailCorrelation'->>'score' as name_email_correlation,
       tpr.response::jsonb->'namePhoneCorrelation'->>'score' as name_phone_correlation,
       tpr.response::jsonb->'social'->>'profilesFound' as social_profiles_found
from event e
join map on e.event_id=map.event_id
join thirdparty_raw tpr on tpr.thirdparty_raw_id = map.thirdparty_raw_id
join thirdparty_provider tpp on tpr.thirdparty_provider_id=tpp.thirdparty_provider_id
where e.kob='BANKING' and tpp.code='SOCURE'