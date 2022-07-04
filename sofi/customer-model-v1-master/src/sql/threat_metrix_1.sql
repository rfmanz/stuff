-- Query for Threat Metrix data pull
with
    map
    as
    (
        select distinct thirdparty_raw_id, event_id
        from TDM_RISK_MGMT_HUB.CLEANSED.decision
    )
select e.user_id,
       tpr.created_dt,
       parse_json(tpr.response):results.os as os,
       parse_json(tpr.response):results.time_zone as time_zone,
       parse_json(tpr.response):results.dns_ip_geo as dns_ip_geo,
       parse_json(tpr.response):results.enabled_ck as enabled_ck,
       parse_json(tpr.response):results.enabled_fl as enabled_fl,
       parse_json(tpr.response):results.enabled_im as enabled_im,
       parse_json(tpr.response):results.enabled_js as enabled_js,
       parse_json(tpr.response):results.screen_res as screen_res,
       parse_json(tpr.response):results.agent_brand as agent_brand,
       parse_json(tpr.response):results.device_name  as device_name,
       parse_json(tpr.response):results.page_time_on as page_time_on,
       parse_json(tpr.response):results.dns_ip_region as dns_ip_region,
       parse_json(tpr.response):results.agent_language as agent_language,
       parse_json(tpr.response):results.tmx_risk_rating as tmx_risk_rating,
       parse_json(tpr.response):results.browser_language as browser_language
from TDM_RISK.CLEANSED.event e
    join map on e.event_id=map.event_id
    join TDM_RISK.CLEANSED.thirdparty_raw tpr on tpr.thirdparty_raw_id = map.thirdparty_raw_id
    join TDM_RISK.CLEANSED.thirdparty_provider tpp on tpr.thirdparty_provider_id=tpp.thirdparty_provider_id
where e.kob='BANKING' and tpp.code='TMX'
