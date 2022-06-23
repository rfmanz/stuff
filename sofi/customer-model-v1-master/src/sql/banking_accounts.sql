-- get data at a banking account level.
WITH primary_account_holders AS
    (SELECT cif.zsofiid AS user_id,
            dep.cid AS business_account_number,
            dao AS date_account_opened,
            dtc AS date_account_closed,
            dep.type AS account_type,
            cldr.des AS account_closed_reason,
            joint.zsofiid AS joint_user_id
     FROM cif
     JOIN dep ON cif.acn = dep.acn
     LEFT JOIN utblcloser cldr ON cldr.clr = dep.clr AND cldr.cls = dep.cls
     LEFT JOIN
         (SELECT rc.cid,
                 cif.zsofiid,
                 rc.acn
          FROM relcif rc
          JOIN cif ON cif.acn = rc.acn
          JOIN dep ON dep.cid = rc.cid
          JOIN utblrel2 rel ON rel.grp = dep.grp
          AND rel.key = dep.acnrelc
          AND rc.role=2) joint ON joint.cid = dep.cid
     WHERE dep.type=410 ),
     joint_account_holders AS
    (SELECT joint_user_id AS user_id,
            business_account_number,
            date_account_opened,
            date_account_closed,
            account_type,
            account_closed_reason,
            user_id AS joint_user_id
     FROM primary_account_holders
     WHERE joint_user_id IS NOT NULL )
SELECT *
FROM
    (SELECT *
     FROM primary_account_holders
     UNION SELECT *
     FROM joint_account_holders) AS internal_banking_accounts
