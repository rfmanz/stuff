create table dwanalyst.hp_asof_date_v0 as
select distinct asof_date from loans_blended_monthly
order by 1 asc

create table dwanalyst.hp_asof_date_v1 as
select asof_date, lag(asof_date,1)  over(order by asof_date) as asof_date_pre from dwanalyst.hp_asof_date_v0

create table dwanalyst.pl_gen4_member_raw_202202 as
select base.dw_applicant_id,
       base.id                                                      as current_id,
       base.date_start                                              as current_date_start,
       lbd.id                                                       as all_time_id,
       lbd.application_type                                         as all_time_application_type,
       lbd.origination_date                                         as all_time_origination_date,
       lbd.initial_term                                             as all_time_term,
       lbd.original_prin                                            as all_time_orig_bal,
       lbd.charge_off_date                                          as all_time_charge_off_date,
       lbd.charge_off_prin                                          as all_time_charge_off_prin,
       datediff('month', base.date_start, h.last_status_date) + 1   as all_time_seasoning_extra,
       datediff('month', lbd.origination_date, base.date_start) + 1 as all_time_seasoning,
       lbm.current_prin                                             as all_time_cur_prin,
       lbm.loan_status                                              as all_time_loan_status,
       lbm.asof_date                                                as all_time_asof_date,
       case
           when lbm.current_prin > 0 then 1
           else 0 end                                               as if_active,
       case
           when length(h.status_string) - datediff('month', base.date_start, h.last_status_date) -
                1 < 0 then null
           else substring(h.status_string, 1,
                          length(h.status_string) - datediff('month', base.date_start, h.last_status_date) -
                          1) end                                    as all_time_string_asof_datestart,
       h.last_status_date                                           as all_time_string_latest_date,
       h.status_string                                              as all_time_status_string_latest
from dwanalyst.pl_gen4_base_202202 base
         left join borrowers b on base.dw_applicant_id = b.borrower_id
         inner join loans_blended_daily lbd on b.user_id = lbd.applicant_id and base.date_start > lbd.origination_date
         left join dwanalyst.loan_history_string_202202 h on h.id = lbd.id
         left join dwanalyst.hp_asof_date_v1 date
                   on extract(year from base.date_start) = extract(year from date.asof_date) and
                      extract(month from base.date_start) = extract(month from date.asof_date)
         left join loans_blended_monthly lbm on lbd.id = lbm.id and lbm.asof_date = date.asof_date_pre
where lbd.id <> 0;


create table dwanalyst.pl_gen4_member_agg_202202 as
with a as (
    select *,
           case when all_time_application_type = 'PL' then 1 else 0 end          as pl_flag,
           case
               when all_time_application_type in ('REFI', 'PLUS', 'DENTREFI', 'MEDREFI') then 1
               else 0 end                                                        as slr_flag,
           case when all_time_application_type in ('INSCHOOL') then 1 else 0 end as isl_flag,
           case
               when all_time_application_type in ('PL', 'REFI', 'PLUS', 'DENTREFI', 'MEDREFI', 'INSCHOOL') then 1
               else 0 end                                                        as lending_flag,
           case
               when right(all_time_string_asof_datestart, 3) similar to '%(3|4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever30dpd3m_flag,
           case
               when right(all_time_string_asof_datestart, 3) similar to '%(4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever60dpd3m_flag,
           case
               when right(all_time_string_asof_datestart, 3) similar to '%(5|6|B|D|W)%' then 1
               else 0 end                                                        as ever90dpd3m_flag,
           case
               when right(all_time_string_asof_datestart, 6) similar to '%(3|4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever30dpd6m_flag,
           case
               when right(all_time_string_asof_datestart, 6) similar to '%(4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever60dpd6m_flag,
           case
               when right(all_time_string_asof_datestart, 6) similar to '%(5|6|B|D|W)%' then 1
               else 0 end                                                        as ever90dpd6m_flag,
           case
               when right(all_time_string_asof_datestart, 12) similar to '%(3|4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever30dpd12m_flag,
           case
               when right(all_time_string_asof_datestart, 12) similar to '%(4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever60dpd12m_flag,
           case
               when right(all_time_string_asof_datestart, 12) similar to '%(5|6|B|D|W)%' then 1
               else 0 end                                                        as ever90dpd12m_flag,
           case
               when right(all_time_string_asof_datestart, 24) similar to '%(3|4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever30dpd24m_flag,
           case
               when right(all_time_string_asof_datestart, 24) similar to '%(4|5|6|B|D|W)%' then 1
               else 0 end                                                        as ever60dpd24m_flag,
           case
               when right(all_time_string_asof_datestart, 24) similar to '%(5|6|B|D|W)%' then 1
               else 0 end                                                        as ever90dpd24m_flag,
           case
               when right(all_time_string_asof_datestart, 1) similar to '%(3|4|5|6|B|D|W)%' then 1
               else 0 end                                                        as current_30dpd_flag,
           case
               when right(all_time_string_asof_datestart, 1) similar to '%(4|5|6|B|D|W)%' then 1
               else 0 end                                                        as current_60dpd_flag,
           case
               when right(all_time_string_asof_datestart, 1) similar to '%(F)%' then 1
               else 0 end                                                        as current_forb_flag,
           case
               when right(all_time_string_asof_datestart, 24) similar to '%(F)%' then 1
               else 0 end                                                        as everforb24m_flag,
           case
               when all_time_string_asof_datestart similar to '%(P)%' then 1
               else 0 end                                                        as paidinfull_flag
    from dwanalyst.pl_gen4_member_raw_202202
    where all_time_application_type in ('PL', 'REFI', 'PLUS', 'DENTREFI', 'MEDREFI', 'INSCHOOL')
)
select dw_applicant_id,
       current_id,
       case when sum(pl_flag) > 0 then 1 else 0 end      as all_time_pl_member_flag,
       case when sum(slr_flag) > 0 then 1 else 0 end     as all_time_slr_member_flag,
       case when sum(isl_flag) > 0 then 1 else 0 end     as all_time_isl_member_flag,
       case when sum(lending_flag) > 0 then 1 else 0 end as all_time_lending_member_flag,
       max(all_time_seasoning)                           as all_time_months_oldest_lending_trade,
       min(all_time_seasoning)                           as all_time_months_newest_lending_trade,
       case
           when sum(pl_flag) > 0 then max(all_time_seasoning * pl_flag)
           else null end                                 as all_time_months_oldest_pl_trade,
       case
           when sum(pl_flag) > 0 then min(all_time_seasoning * pl_flag)
           else null end                                 as all_time_months_newest_pl_trade,
       sum(lending_flag)                                 as all_time_num_lending_trade,
       sum(pl_flag)                                      as all_time_num_pl_trade,
       sum(lending_flag * if_active)                     as current_num_lending_trade,
       sum(pl_flag * if_active)                          as current_num_pl_trade,
       sum(all_time_orig_bal * lending_flag)             as all_time_lending_trade_orig_bal,
       case
           when sum(pl_flag) > 0 then sum(all_time_orig_bal * pl_flag)
           else null end                                 as all_time_pl_trade_orig_bal,
       case
           when sum(lending_flag * if_active) > 0 then sum(all_time_orig_bal * lending_flag * if_active)
           else null end                                 as current_lending_trade_orig_bal,
       case
           when sum(pl_flag * if_active) > 0 then sum(all_time_orig_bal * pl_flag * if_active)
           else null end                                 as current_pl_trade_orig_bal,
       case
           when sum(lending_flag * if_active) > 0 then sum(all_time_cur_prin * lending_flag * if_active)
           else null end                                 as current_lending_trade_current_prin,
       case
           when sum(pl_flag * if_active) > 0 then sum(all_time_cur_prin * pl_flag * if_active)
           else null end                                 as current_pl_trade_current_prin,
       case
           when sum(all_time_orig_bal * lending_flag * if_active) > 0 then
                   sum(all_time_cur_prin * lending_flag * if_active) / sum(all_time_orig_bal * lending_flag * if_active)
           else null end                                 as current_lending_trade_bal_ratio,
       case
           when sum(all_time_orig_bal * pl_flag * if_active) > 0 then sum(all_time_cur_prin * pl_flag * if_active) /
                                                                      sum(all_time_orig_bal * pl_flag * if_active)
           else null end                                 as current_pl_trade_bal_ratio,
       case
           when sum(pl_flag) > 0 then avg(all_time_term * pl_flag)
           else null end                                 as all_time_avg_pl_term,
       case
           when sum(slr_flag) > 0 then avg(all_time_term * slr_flag)
           else null end                                 as all_time_avg_sl_term,
       case
           when sum(pl_flag * if_active) > 0 then avg(all_time_term * pl_flag * if_active)
           else null end                                 as current_avg_pl_term,
       case
           when sum(slr_flag * if_active) > 0 then avg(all_time_term * slr_flag * if_active)
           else null end                                 as current_avg_sl_term,
       sum(ever30dpd3m_flag * lending_flag)              as all_time_num_lending_trade_ever30dpd3m,
       sum(ever60dpd3m_flag * lending_flag)              as all_time_num_lending_trade_ever60dpd3m,
       sum(ever90dpd3m_flag * lending_flag)              as all_time_num_lending_trade_ever90dpd3m,
       sum(ever30dpd6m_flag * lending_flag)              as all_time_num_lending_trade_ever30dpd6m,
       sum(ever60dpd6m_flag * lending_flag)              as all_time_num_lending_trade_ever60dpd6m,
       sum(ever90dpd6m_flag * lending_flag)              as all_time_num_lending_trade_ever90dpd6m,
       sum(ever30dpd12m_flag * lending_flag)             as all_time_num_lending_trade_ever30dpd12m,
       sum(ever60dpd12m_flag * lending_flag)             as all_time_num_lending_trade_ever60dpd12m,
       sum(ever90dpd12m_flag * lending_flag)             as all_time_num_lending_trade_ever90dpd12m,
       sum(ever30dpd24m_flag * lending_flag)             as all_time_num_lending_trade_ever30dpd24m,
       sum(ever60dpd24m_flag * lending_flag)             as all_time_num_lending_trade_ever60dpd24m,
       sum(ever90dpd24m_flag * lending_flag)             as all_time_num_lending_trade_ever90dpd24m,
       sum(current_30dpd_flag * lending_flag)            as all_time_num_lending_trade_current_30dpd,
       sum(current_60dpd_flag * lending_flag)            as all_time_num_lending_trade_current_60dpd,
       sum(current_forb_flag * lending_flag)             as all_time_num_lending_trade_current_forb,
       sum(everforb24m_flag * lending_flag)              as all_time_num_lending_trade_everforb24m,
       sum(paidinfull_flag * lending_flag)               as all_time_num_lending_trade_paidinfull,
       case
           when sum(pl_flag) > 0 then sum(ever30dpd3m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever30dpd3m,
       case
           when sum(pl_flag) > 0 then sum(ever60dpd3m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever60dpd3m,
       case
           when sum(pl_flag) > 0 then sum(ever90dpd3m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever90dpd3m,
       case
           when sum(pl_flag) > 0 then sum(ever30dpd6m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever30dpd6m,
       case
           when sum(pl_flag) > 0 then sum(ever60dpd6m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever60dpd6m,
       case
           when sum(pl_flag) > 0 then sum(ever90dpd6m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever90dpd6m,
       case
           when sum(pl_flag) > 0 then sum(ever30dpd12m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever30dpd12m,
       case
           when sum(pl_flag) > 0 then sum(ever60dpd12m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever60dpd12m,
       case
           when sum(pl_flag) > 0 then sum(ever90dpd12m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever90dpd12m,
       case
           when sum(pl_flag) > 0 then sum(ever30dpd24m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever30dpd24m,
       case
           when sum(pl_flag) > 0 then sum(ever60dpd24m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever60dpd24m,
       case
           when sum(pl_flag) > 0 then sum(ever90dpd24m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_ever90dpd24m,
       case
           when sum(pl_flag) > 0 then sum(current_30dpd_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_current_30dpd,
       case
           when sum(pl_flag) > 0 then sum(current_60dpd_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_current_60dpd,
       case
           when sum(pl_flag) > 0 then sum(current_forb_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_current_forb,
       case
           when sum(pl_flag) > 0 then sum(everforb24m_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_everforb24m,
       case
           when sum(pl_flag) > 0 then sum(paidinfull_flag * pl_flag)
           else null end                                 as all_time_num_pl_trade_paidinfull
from a
group by 1, 2