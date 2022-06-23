-- banking transactions by business account.
SELECT tjd,
       dtj.time,
       dtj.cid AS business_account_number,
       cdt AS created_dt,
       endbal,
       dtj.etc AS transaction_code,
       tamt AS transaction_amount,
       trn.dc as is_credit,
       cif.zsofiid as user_id,
       split_part(split_part(tso, 'RCID#', 2), '~', 1) AS external_account_number,
       split_part(split_part(dtj.tso, 'RINS#', 2), '~', 1) AS external_institution_id,
       split_part(split_part(dtj.tso, 'ACHCOID#', 2), '~', 1) AS originating_company_id,
       split_part(split_part(dtj.tso, 'ACHID#', 2), '~', 1) AS external_institution_trans_id,
       split_part(split_part(dtj.tso, 'ACHODFI#', 2), '~', 1) AS originator_dfi_id,
       auth.merchnm AS merchant_name
FROM dtj
LEFT JOIN authdtl auth ON auth.authid = split_part(split_part(dtj.tso, 'AUTHID#', 2), '~', 1) AND dtj.tso LIKE '%%AUTHID#%%'
JOIN profile_reporting.dep ON dep.cid = dtj.cid
JOIN cif ON cif.acn = dep.acn
LEFT JOIN trn on dtj.etc = trn.etc
WHERE endbal IS NOT NULL
      and tamt IS NOT NULL
      and tamt NOT LIKE '%%#%%'
