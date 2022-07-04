
-- banking transactions by business account.
SELECT system_processing_date
    , dtj.clock_time
    , dtj.account_number AS business_account_number
    , dtj.calendar_date AS created_dt
    , dtj.ending_balance as endbal
    , dtj.external_transaction_code AS transaction_code
    , dtj.transaction_amount AS transaction_amount
    , trn.debit_or_credit as is_credit
    , cif.sofi_user_id_reference as user_id
    , split_part(split_part(dtj.transaction_source_tags, 'RCID#', 2), '~', 1) AS external_account_number
    , split_part(split_part(dtj.transaction_source_tags, 'RINS#', 2), '~', 1) AS external_institution_id 
    , split_part(split_part(dtj.transaction_source_tags, 'ACHCOID#', 2), '~', 1) AS originating_company_id
    , split_part(split_part(dtj.transaction_source_tags, 'ACHID#', 2), '~', 1) AS external_institution_trans_id
    , split_part(split_part(dtj.transaction_source_tags, 'ACHODFI#', 2), '~', 1) AS originator_dfi_id
    , auth.merchant_name AS merchant_name
FROM tdm_bank.cleansed.profile_daily_transaction_journal dtj
    LEFT JOIN tdm_bank.cleansed.profile_authorizations auth ON auth.authorization_id = split_part(split_part(dtj.transaction_source_tags, 'AUTHID#', 2), '~', 1) AND dtj.transaction_source_tags LIKE '%%AUTHID#%%'
    JOIN tdm_bank.cleansed.profile_deposits dep ON dep.account_number = dtj.account_number
    JOIN tdm_bank.cleansed.profile_customers cif ON cif.customer_number = dep.customer_number
    LEFT JOIN tdm_bank.cleansed.profile_transaction_codes trn on dtj.external_transaction_code = trn.transaction_code
WHERE endbal IS NOT NULL
    and dtj.transaction_amount IS NOT NULL
    and dtj.transaction_amount NOT LIKE '%%#%%'
