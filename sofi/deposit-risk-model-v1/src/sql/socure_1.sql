WITH
    bid
    AS
    -- get borrower_id, banking_account_id for all SoFi money accounts 30+ days old opened after Jan 1st 2019
    (
        SELECT DISTINCT badf.borrower_id
        FROM TDM_RISK_MGMT_HUB.modeled.banking_account_daily_facts badf
            JOIN TDM_RISK_MGMT_HUB.modeled.banking_accounts ba ON badf.banking_account_id = ba.banking_account_id
            JOIN TDM_RISK_MGMT_HUB.modeled.borrowers b ON b.borrower_id=badf.borrower_id
        WHERE product_type='SoFi Money Cash Account'
            AND b.sofi_employee_ind=FALSE
            AND ba.active_flag=TRUE
    )
SELECT bid.borrower_id,
    so.socure_pull_date,
    so.fraud_score_1,
    so.fraud_score_2,
    so.email_risk_score,
    so.phone_risk_score,
    so.address_risk_score
FROM TDM_RISK_MGMT_HUB.modeled.socure_pull so
    JOIN bid ON bid.borrower_id = so.borrower_id
