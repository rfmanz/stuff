
-- get data at a banking account level.
WITH
    primary_account_holders
    AS
    (
        SELECT
            cif.sofi_user_id_reference  AS user_id
            , dep.account_number AS business_account_number
            , cif.date_account_opened AS date_account_opened
            , dep.account_closed_date
 AS date_account_closed
            , dep.product_type AS account_type
            , cldr.description AS account_closed_reason
            , joint.sofi_user_id_reference AS joint_user_id
        FROM tdm_bank.cleansed.profile_customers cif
            JOIN tdm_bank.cleansed.profile_deposits dep
            ON cif.customer_number = dep.customer_number
            LEFT JOIN tdm_bank.cleansed.profile_utility_closeout_reason_codes cldr ON cldr.closeout_reason_code = dep.account_closeout_reason_code
                AND cldr.product_class = dep.product_class
            LEFT JOIN
            (SELECT rc.account_number
                , cif.sofi_user_id_reference
                , rc.customer_number
            FROM tdm_bank.cleansed.profile_customer_accounts_mapping rc
                JOIN tdm_bank.cleansed.profile_customers cif ON cif.customer_number = rc.customer_number
                JOIN tdm_bank.cleansed.profile_deposits dep ON dep.account_number = rc.account_number
                JOIN tdm_bank.cleansed.profile_utility_valid_relationships_by_groups
 rel ON rel.product_group = dep.product_group
                    AND rel.relationship_code  = dep.account_relationship_code
                    AND rc.role_code_for_relationship=2) joint ON joint.account_number = dep.account_number
        WHERE dep.product_type=410
    ),
    joint_account_holders
    AS
    (
        SELECT joint_user_id AS user_id,
            business_account_number,
            date_account_opened,
            date_account_closed,
            account_type,
            account_closed_reason,
            user_id AS joint_user_id
        FROM primary_account_holders
        WHERE joint_user_id IS NOT NULL
    )
SELECT *
FROM
    (                                                                                                                SELECT *
        FROM primary_account_holders
    UNION
        SELECT *
        FROM joint_account_holders) AS internal_banking_accounts
