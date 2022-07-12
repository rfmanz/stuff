-- Query for Giact data pulls
with
    map
    as
    (
        select distinct thirdparty_raw_id, event_id
        from TDM_RISK_MGMT_HUB.CLEANSED.decision
    )
select e.user_id, 
       tpr.created_dt,
       parse_json(tpr.request):sofiAccountNumber as business_account_number,
       parse_json(tpr.response):bankName as giact_bank_name,
       parse_json(tpr.response):createdDate as giact_created_date,
       parse_json(tpr.response):itemReferenceId as giact_item_reference_id,
       parse_json(tpr.response):accountAddedDate as giact_account_added_date,
       parse_json(tpr.response):accountResponseCode as giact_account_response_code,
       parse_json(tpr.response):customerResponseCode as giact_customer_response_code, 
       parse_json(tpr.response):verificationResponse as giact_verification_response,
       parse_json(tpr.response):accountLastUpdatedDate as giact_account_last_updated_date
from TDM_RISK.CLEANSED.event e
    join map on e.event_id=map.event_id
    join TDM_RISK.CLEANSED.thirdparty_raw tpr on tpr.thirdparty_raw_id = map.thirdparty_raw_id
    join TDM_RISK.CLEANSED.thirdparty_provider tpp on tpr.thirdparty_provider_id=tpp.thirdparty_provider_id
where e.kob='BANKING' and tpp.code='GIACT'
