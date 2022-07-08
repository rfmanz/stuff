-- Query for Giact data pulls
with map as (select distinct thirdparty_raw_id, event_id
             from decision)
select e.user_id, 
       tpr.created_dt,
       tpr.request::jsonb->>'sofiAccountNumber' as business_account_number,
       tpr.response::jsonb->>'bankName' as giact_bank_name,
       tpr.response::jsonb->>'createdDate' as giact_created_date,
       tpr.response::jsonb->>'itemReferenceId' as giact_item_reference_id,
       tpr.response::jsonb->>'accountAddedDate' as giact_account_added_date,
       tpr.response::jsonb->>'accountResponseCode' as giact_account_response_code,
       tpr.response::jsonb->>'customerResponseCode' as giact_customer_response_code, 
       tpr.response::jsonb->>'verificationResponse' as giact_verification_response,
       tpr.response::jsonb->>'accountLastUpdatedDate' as giact_account_last_updated_date
from event e
join map on e.event_id=map.event_id
join thirdparty_raw tpr on tpr.thirdparty_raw_id = map.thirdparty_raw_id
join thirdparty_provider tpp on tpr.thirdparty_provider_id=tpp.thirdparty_provider_id
where e.user_id in <USER_ID>
and e.kob='BANKING' and tpp.code='GIACT'