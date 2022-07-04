-- Query for Socure data pull

with map as (select distinct thirdparty_raw_id, event_id
             from TDM_RISK_MGMT_HUB.CLEANSED.decision) -- is this right ? 
select e.user_id
       , tpr.created_dt
       , case when parse_json(tpr.response):fraud.scores[0].version = '3.0' 
        then parse_json(tpr.response):fraud.scores[0].score 
        else parse_json(tpr.response):fraud.scores[1].score end as fraud_score_1_v1

       , case when parse_json(tpr.response):fraud.scores[0].version = '1.0'  
        then parse_json(tpr.response):fraud.scores[0].score 
        else parse_json(tpr.response):fraud.scores[1].score end as fraud_score_2_v1

       , case when parse_json(tpr.response):fraud.scores[0].name='generic' 
        then parse_json(tpr.response):fraud.scores[0].score 
        when parse_json(tpr.response):fraud.scores[1].name='generic' 
        then parse_json(tpr.response):fraud.scores[1].score
        else null end as fraud_score_1

       , case when parse_json(tpr.response):fraud.scores[0].name='generic' 
        then parse_json(tpr.response):fraud.scores[0].version
        when parse_json(tpr.response):fraud.scores[1].name='generic' 
        then parse_json(tpr.response):fraud.scores[1].version
        else null end as fraud_score_1_version

       , case when parse_json(tpr.response):fraud.scores[0].name='sigma' 
        then parse_json(tpr.response):fraud.scores[0].score 
        when parse_json(tpr.response):fraud.scores[1].name='sigma' 
        then parse_json(tpr.response):fraud.scores[1].score
        else null end as fraud_score_2        
           
       , case when parse_json(tpr.response):fraud.scores[0].name='sigma' 
        then parse_json(tpr.response):fraud.scores[0].version
        when parse_json(tpr.response):fraud.scores[1].name='sigma' 
        then parse_json(tpr.response):fraud.scores[1].version
        else null end as fraud_score_2_version

       , parse_json(tpr.response):addressRisk.score as address_risk_score
       , parse_json(tpr.response):emailRisk.score as email_risk_score
       , parse_json(tpr.response):phoneRisk.score as phone_risk_score
       , parse_json(tpr.response):nameAddressCorrelation.score as name_address_correlation
       , parse_json(tpr.response):nameEmailCorrelation.score as name_email_correlation
       , parse_json(tpr.response):namePhoneCorrelation.score as name_phone_correlation
       , parse_json(tpr.response):social.profilesFound as social_profiles_found
       
from TDM_RISK.CLEANSED.event e
join map on e.event_id=map.event_id
join TDM_RISK.CLEANSED.thirdparty_raw tpr on tpr.thirdparty_raw_id = map.thirdparty_raw_id
join TDM_RISK.CLEANSED.thirdparty_provider tpp on tpr.thirdparty_provider_id=tpp.thirdparty_provider_id
where e.kob='BANKING' and tpp.code='SOCURE'
