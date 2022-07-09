
with outputs as (
    select *
    from tdm_bank.cleansed.profile_daily_transaction_journal j
    left join tdm_bank.modeled.profile_daily_transaction_journal_transaction_source_tags jst
        on j.key_hash = jst.key_hash
    )
select
system_processing_date tjd 
,o.clock_time as time 
, o.account_number AS business_account_number
, o.calendar_date AS created_dt
, o.ending_balance as endbal
, o.external_transaction_code AS transaction_code
, o.transaction_amount AS transaction_amount
, trn.debit_or_credit as is_credit
, cif.sofi_user_id_reference as user_id

,o.json_payload:RECIPIENT_ACCOUNT__RCID::varchar as external_account_number
,o.json_payload:RECIPIENT_INSTITUTION__RINS::varchar as external_institution_id
,ach_company_id__achcoid AS originating_company_id
,ach_id__achid AS external_institution_trans_id
,ach_originating_dfi_identification__achodfi AS originator_dfi_id

, auth.merchant_name AS merchant_name

from outputs o

LEFT JOIN tdm_bank.cleansed.profile_authorizations auth 
ON auth.authorization_id = o.authorization_id__authid -- AND dtj.transaction_source_tags LIKE '%%AUTHID#%%'

JOIN tdm_bank.cleansed.profile_deposits dep ON dep.account_number = o.account_number
JOIN tdm_bank.cleansed.profile_customers cif ON cif.customer_number = dep.customer_number
LEFT JOIN tdm_bank.cleansed.profile_transaction_codes trn on o.external_transaction_code = trn.transaction_code

WHERE endbal IS NOT NULL
    and o.transaction_amount IS NOT NULL
    and o.transaction_amount NOT LIKE '%%#%%'





