WITH
    last_unrestricted_date
    AS
    (
        SELECT business_account_number,
            max(effective_date) AS last_unrestricted_date
        FROM banking_accounts
        WHERE account_restricted_ind=FALSE
            AND account_closed_date is NULL
        GROUP BY 1
    ),
    first_restricted_by_risk_date
    AS
    (
        SELECT business_account_number,
            min(effective_date) AS first_restricted_by_risk_date
        FROM banking_accounts
        WHERE account_restricted_reason LIKE '%%No%%'
        GROUP BY 1
    )
SELECT ba.business_account_number,
    max(last_unrestricted_date) AS last_unrestricted_date,
    min(first_restricted_by_risk_date) AS first_restricted_by_risk_date
FROM banking_accounts ba
    LEFT JOIN last_unrestricted_date lud ON lud.business_account_number=ba.business_account_number
    LEFT JOIN first_restricted_by_risk_date frbrd ON frbrd.business_account_number=ba.business_account_number
GROUP BY 1