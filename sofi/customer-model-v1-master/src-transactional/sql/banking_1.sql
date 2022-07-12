-- Up to date banking account info from profile.
WITH
    full_data
    AS
    (
        SELECT dep.account_number AS business_account_number,
            sofi_user_id_reference as zsofiid,
            account_opened_date as odt,
            account_closed_date as dtc,
            balance_ledger as bal,
            account_first_deposit_amount as afdep,
            date_of_birth as dob,
            state_mailing_address as mstate,
            zip_code_mailing_address as mzip,
            uc.description AS closed_reason,
            rest_acn."DESC" as restricted_reason
        FROM tdm_bank.cleansed.profile_deposits dep
            JOIN tdm_bank.cleansed.profile_customers cif ON cif.customer_number = dep.customer_number
            LEFT JOIN tdm_bank.cleansed.profile_utility_closeout_reason_codes uc ON uc.closeout_reason_code = dep.account_closeout_reason_code
                AND uc.product_class = dep.product_class
            LEFT JOIN
            (SELECT listagg(DISTINCT utblrflg.description, ', ') AS "DESC"      ,
            restrict_account_number
    
    
          FROM tdm_bank.cleansed.profile_customer_restrict_customer_level res_acn
            JOIN tdm_bank.cleansed.profile_utility_restrict_flags utblrflg
            ON utblrflg.userclass = res_acn.restrict_flag
        GROUP BY restrict_account_number
        ORDER BY restrict_account_number
    ) rest_acn ON rest_acn.restrict_account_number = dep.account_number
     WHERE account_name='Cash Account'
     ORDER BY odt DESC)
SELECT business_account_number,
                   zsofiid,
                   odt,
                   dtc,
                   bal,
                   afdep,
                   dob,
                   mstate,
                   mzip,
                   closed_reason,
                   restricted_reason
    FROM full_data
    QUALIFY ROW_NUMBER() OVER (PARTITION BY business_account_number ORDER BY business_account_number) = 1 
