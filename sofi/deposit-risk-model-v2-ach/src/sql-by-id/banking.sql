-- Up to date banking account info from profile.
WITH full_data AS
    ( SELECT dep.cid AS business_account_number,
             zsofiid as user_id,
             odt AS acct_open_date,
             dtc AS date_acct_closed,
             bal,
             afdep AS first_deposit_amount,
             dob AS date_of_birth,
             mstate,
             mzip,
             uc.des AS closed_reason,
             rest_acn."DESC" as restricted_reason
     FROM dep
     JOIN cif ON cif.acn = dep.acn
     LEFT JOIN utblcloser uc ON uc.clr = dep.clr
     AND uc.cls = dep.cls
     LEFT JOIN
         (SELECT string_agg(DISTINCT utblrflg."DESC", ', ') AS "DESC",
                 cid
          FROM rflgccid res_acn
          JOIN utblrflg ON utblrflg.rflg = res_acn.rflg
          GROUP BY cid
          ORDER BY cid ) rest_acn ON rest_acn.cid = dep.cid
     WHERE dep.cid in <BUSINESS_ACCOUNT_NUMBER>
     AND acctname='Cash Account'
     ORDER BY odt DESC)
SELECT DISTINCT ON (1) business_account_number,
                   user_id,
                   acct_open_date,
                   date_acct_closed,
                   bal,
                   first_deposit_amount,
                   date_of_birth,
                   mstate,
                   mzip,
                   closed_reason,
                   restricted_reason
FROM full_data