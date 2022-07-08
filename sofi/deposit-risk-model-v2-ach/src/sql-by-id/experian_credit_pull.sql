-- Query for Experian credit pull
SELECT b.user_id,
       cp.credit_pull_date,
       cp.fico_score,
       cp.vantage_score,
       cp.experian_attribute_json::json->>'ALL7120' AS ALL7120,
       cp.experian_attribute_json::json->>'ALL8220' AS ALL8220,
       cp.experian_attribute_json::json->>'BCC2800' AS BCC2800,
       cp.experian_attribute_json::json->>'BCC7120' AS BCC7120,
       cp.experian_attribute_json::json->>'BCX3423' AS BCX3423,
       cp.experian_attribute_json::json->>'ILN5520' AS ILN5520,
       cp.experian_attribute_json::json->>'IQT9415' AS IQT9415,
       cp.experian_attribute_json::json->>'IQT9413' AS IQT9413,
       cp.experian_attribute_json::json->>'MTF5820' AS MTF5820,
       cp.experian_attribute_json::json->>'STU5031' AS STU5031,
       cp.credit_card_loan_amount,
       cp.delinquencies_90_days,
       cp.education_loan_amount,
       cp.mortgage_loan_amount,
       cp.secured_loan_amount,
       cp.total_outstanding_balance,
       cp.total_tradelines_open,
       cp.unsecured_loan_amount
FROM credit_pull cp
JOIN borrowers b ON b.borrower_id = cp.borrower_id
WHERE b.borrower_id in <BORROWER_ID>
and cp.pull_type = 'MONEY'