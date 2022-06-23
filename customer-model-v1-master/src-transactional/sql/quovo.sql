-- Pull quovo data
WITH transactions AS
    (SELECT account_id,
            count(*) AS nr_transactions
     FROM account_transactions
     GROUP BY 1)
SELECT im.party_id,
       aa.response::jsonb->>'type' AS account_type,
       aa.response::jsonb->>'availableBalance' AS available_balance,
       aa.fetched_from_quovo_dt,
       t.nr_transactions
FROM account_auth aa
LEFT JOIN transactions t ON t.account_id = aa.account_id
JOIN connection_accounts ca ON aa.account_id = ca.account_id
JOIN customer_connections cc ON ca.connection_id = cc.connection_id
JOIN identity_mapping im ON cc.quovo_id = im.quovo_id