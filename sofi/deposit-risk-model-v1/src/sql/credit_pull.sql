WITH bid AS -- get borrower_id, banking_account_id for all SoFi money accounts 30+ days old opened after Jan 1st 2019
    (SELECT DISTINCT badf.borrower_id
     FROM banking_account_daily_facts badf
     JOIN banking_accounts ba ON badf.banking_account_id = ba.banking_account_id
     JOIN borrowers b ON b.borrower_id=badf.borrower_id
     WHERE product_type='SoFi Money Cash Account'
         AND b.sofi_employee_ind=FALSE
         AND ba.active_flag=TRUE)
SELECT bid.borrower_id,
       cp.credit_pull_date,
	   cp.fico_score,
	   cp.vantage_score,
	   cp.experian_attribute_json::json->>'BCC7120' AS BCC7120,
	   cp.experian_attribute_json::json->>'BCC2800' AS BCC2800,
	   cp.experian_attribute_json::json->>'ILN5520' AS ILN5520,
	   cp.experian_attribute_json::json->>'BCX3423' AS BCX3423,
	   cp.experian_attribute_json::json->>'STU5031' AS STU5031,
	   cp.experian_attribute_json::json->>'IQT9415' AS IQT9415,
	   cp.experian_attribute_json::json->>'IQT9413' AS IQT9413,
	   cp.experian_attribute_json::json->>'MTF5820' AS MTF5820,
	   cp.experian_attribute_json::json->>'ALL7120' AS ALL7120,
	   cp.experian_attribute_json::json->>'ALL8220' AS ALL8220
FROM credit_pull cp
JOIN bid ON bid.borrower_id = cp.borrower_id
