-- temporarily pull from DW, eventually will pull data from
-- BankingTransactionOccured Kafka topic (when it is fixed in Data Lake)
SELECT borrower_id,
    btf.banking_account_id,
    btf.banking_transaction_details_id,
    ba.business_account_number,
    transaction_created_date_id,
    time_of_day,
    transaction_code,
    transaction_comment,
    transaction_amount,
    account_ending_balance,
    card_present_ind,
    external_account_number
FROM TDM_RISK_MGMT_HUB.modeled.banking_transactions_facts btf
    LEFT JOIN TDM_RISK_MGMT_HUB.modeled.banking_transactions_details btd ON btf.banking_transaction_details_id=btd.banking_transaction_details_id
    LEFT JOIN TDM_RISK_MGMT_HUB.modeled.time_of_day tod on btf.transaction_created_time_id = tod.time_id
    LEFT JOIN TDM_RISK_MGMT_HUB.modeled.banking_accounts ba on btf.banking_account_id = ba.banking_account_id
WHERE transaction_code IS NOT NULL
    AND account_ending_balance IS NOT NULL
    AND btd.active_flag = TRUE
    AND btd.transaction_code NOT IN ('DIAD', 'IIAD', 'IIPD', 'DIPD')
