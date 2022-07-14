-- Query for Experian credit pull
SELECT b.user_id,
       cp.credit_pull_date,
       cp.fico_score,
       cp.vantage_score,
       parse_json(cp.experian_attribute_json):ALL7120::int as ALL7120,
       parse_json(cp.experian_attribute_json):ALL8220::int as ALL8220,
       parse_json(cp.experian_attribute_json):BCC2800::int as BCC2800,
       parse_json(cp.experian_attribute_json):BCC7120::int as BCC7120,
       parse_json(cp.experian_attribute_json):BCX3423::int as BCX3423,
       parse_json(cp.experian_attribute_json):ILN5520::int as ILN5520,
       parse_json(cp.experian_attribute_json):IQT9415::int as IQT9415,
       parse_json(cp.experian_attribute_json):IQT9413::int as IQT9413,
       parse_json(cp.experian_attribute_json):MTF5820::int as MTF5820,
       parse_json(cp.experian_attribute_json):STU5031::int as STU5031,
       cp.credit_card_loan_amount,
       cp.delinquencies_90_days,
       cp.education_loan_amount,
       cp.mortgage_loan_amount,
       cp.secured_loan_amount,
       cp.total_outstanding_balance,
       cp.total_tradelines_open,
       cp.unsecured_loan_amount
FROM TDM_RISK_MGMT_HUB.modeled.credit_pull cp
JOIN TDM_RISK_MGMT_HUB.modeled.borrowers b ON b.borrower_id = cp.borrower_id
WHERE cp.pull_type = 'MONEY'