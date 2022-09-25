with a as (
    select base.dw_applicant_id,
           base.id                                                         as current_id,
           base.date_start                                                 as current_date_start,
           lbd.id                                                          as all_time_id,
           lbd.application_type                                            as all_time_application_type,
           lbd.origination_date                                            as all_time_origination_date,
           lbd.original_prin                                               as all_time_orig_bal,
           case when lbm.current_prin > 0 then lbm.current_prin else 0 end as all_time_cur_prin_lbm,
           case
               when coalesce(dlf.prin_amount, lbm.current_prin) > 0 then coalesce(dlf.prin_amount, lbm.current_prin)
               else 0 end                                                  as all_time_cur_prin_dlf,
           lbm.loan_status                                                 as all_time_loan_status,
           lbm.asof_date                                                   as all_time_asof_date
    from dwanalyst.pl_gen4_base_202202 base -- for development period please use: dwanalyst.pl_gen4_base_202105_dat_5908
             left join borrowers b on base.dw_applicant_id = b.borrower_id
             inner join loans_blended_daily lbd
                        on b.user_id = lbd.applicant_id and base.date_start > lbd.origination_date
             left join dwanalyst.hp_asof_date_v1 date
                       on extract(year from base.date_start) = extract(year from date.asof_date) and
                          extract(month from base.date_start) = extract(month from date.asof_date)
             left join loans_blended_monthly lbm on lbd.id = lbm.id and lbm.asof_date = date.asof_date_pre
             left join dates dt on dt.calendar_date = base.date_start - 1
             left join daily_loan_facts dlf on lbd.loan_id = dlf.loan_id and dt.date_id = dlf.reporting_date_id
    where lbd.id <> 0
      and base.date_start >= '2022-01-01'       -- change this condition for development data
      and lbd.application_type = 'PL')
select current_id                                          as id,
       dw_applicant_id,
       sum(all_time_cur_prin_dlf) / sum(all_time_orig_bal) as current_pl_trade_bal_ratio_dlf,
       sum(all_time_cur_prin_lbm) / sum(all_time_orig_bal) as current_pl_trade_bal_ratio_lbm,
       sum(all_time_cur_prin_dlf) / sum(all_time_orig_bal) -
       sum(all_time_cur_prin_lbm) / sum(all_time_orig_bal) as ratio_dif
from a
group by 1, 2