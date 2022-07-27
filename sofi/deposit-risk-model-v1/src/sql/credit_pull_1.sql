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
    cp.credit_pull_date,
    cp.fico_score,
    cp.vantage_score,
    parse_json(cp.experian_attribute_json)
:BCC7120::int as BCC7120,
	   parse_json
(cp.experian_attribute_json):BCC2800::int as BCC2800,
	   parse_json
(cp.experian_attribute_json):ILN5520::int as ILN5520,
	   parse_json
(cp.experian_attribute_json):BCX3423::int as BCX3423,
	   parse_json
(cp.experian_attribute_json):STU5031::int as STU5031,
        parse_json
(cp.experian_attribute_json):IQT9415::int as IQT9415,
	   parse_json
(cp.experian_attribute_json):IQT9413::int as IQT9413,
      parse_json
(cp.experian_attribute_json):MTF5820::int as MTF5820,
       parse_json
(cp.experian_attribute_json):ALL7120::int as ALL7120,	   
       parse_json
(cp.experian_attribute_json):ALL8220::int as ALL8220
FROM TDM_RISK_MGMT_HUB.modeled.credit_pull cp
JOIN bid ON bid.borrower_id = cp.borrower_id
