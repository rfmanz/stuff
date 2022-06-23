-- Query for Threat Metrix data pull
with map as (select distinct thirdparty_raw_id, event_id
             from decision)
select e.user_id,
       tpr.created_dt,
       tpr.response::jsonb->'results'->>'os' as os,
       tpr.response::jsonb->'results'->>'time_zone' as time_zone,
       tpr.response::jsonb->'results'->>'dns_ip_geo' as dns_ip_geo,
       tpr.response::jsonb->'results'->>'enabled_ck' as enabled_ck,
       tpr.response::jsonb->'results'->>'enabled_fl' as enabled_fl,
       tpr.response::jsonb->'results'->>'enabled_im' as enabled_im,
       tpr.response::jsonb->'results'->>'enabled_js' as enabled_js,
       tpr.response::jsonb->'results'->>'screen_res' as screen_res,
       tpr.response::jsonb->'results'->>'agent_brand' as agent_brand,
       tpr.response::jsonb->'results'->>'device_name' as device_name,
       tpr.response::jsonb->'results'->>'page_time_on' as page_time_on,
       tpr.response::jsonb->'results'->>'dns_ip_region' as dns_ip_region,
       tpr.response::jsonb->'results'->>'agent_language' as agent_language,
       tpr.response::jsonb->'results'->>'tmx_risk_rating' as tmx_risk_rating,
       tpr.response::jsonb->'results'->>'browser_language' as browser_language
from event e
join map on e.event_id=map.event_id
join thirdparty_raw tpr on tpr.thirdparty_raw_id = map.thirdparty_raw_id
join thirdparty_provider tpp on tpr.thirdparty_provider_id=tpp.thirdparty_provider_id
where e.kob='BANKING' and tpp.code='TMX'
