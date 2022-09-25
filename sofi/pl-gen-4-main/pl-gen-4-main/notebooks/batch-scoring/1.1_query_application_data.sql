create table dwanalyst.pl_application_pull_202202 as
select s.app_id, s.applicant_type, coalesce(s.credit_pull_id, h.credit_pull_id) as credit_pull_id
from dwanalyst.pl_application_softpull_v s
         left join dwanalyst.pl_application_hardpull_v h
                   on s.app_id = h.app_id and s.applicant_id = h.applicant_id and s.applicant_type = h.applicant_type
                   
                   
create table dwanalyst.loan_history_string_202202 as
select distinct on (id) id, loan_id, last_status_date, status_string
from dwanalyst.loan_history_string_v
where last_status_date = (select max(asof_date) from loans_blended_monthly)
order by 1, 2 desc


create table dwanalyst.pl_gen4_base_202202 as
select case
           when v.applicant_type = 'PRIMARY' then pri.borrower_id
           else cob.borrower_id end                            as dw_applicant_id,
       af.*,
       h.last_status_date,
       h.status_string,
       (random() * 10::double precision)::integer              AS num_review,
       lbd.confirmed_fraud,
       lbd.charge_off_date,
       lbd.charge_off_prin,
       CASE
           WHEN lbd.charge_off_date IS NOT NULL AND (- lbd.charge_off_prin) >= lbd.original_prin THEN 'Y'::text
           ELSE NULL::text
           END                                                 AS fpd,
       lbd.asof_date,
       (random() * 100::double precision)::integer             AS audit_score,
       lbd.cur_int_rate,
       lbd.gross_borrower_margin,
       lbd.maturity_date,
       cp.business_credit_pull_id,
       CASE
           WHEN af.application_type = 'PL'::text THEN v.applicant_type
           END                                                 AS applicant_type,
       cp.fico_score                                           AS applicant_fico_score,
       cp.vantage_score                                        AS applicant_vantage_score,
       cp.hard_soft,
       case
           when v.applicant_type = 'PRIMARY' then pri.first_name
           else cob.first_name end                             as first,
       case
           when v.applicant_type = 'PRIMARY' then pri.last_name
           else cob.last_name end                              as last,

       case
           when v.applicant_type = 'PRIMARY' then pri.deceased
           else cob.deceased end                               as deceased,
       case
           when v.applicant_type = 'PRIMARY' then pri.deceased_date
           else cob.deceased_date end                          as deceased_date,
       case when current_decision = 'ACCEPT' then 1 else 0 end as is_approved,
       case when date_fund is not null then 1 else 0 end       as is_funded,
       case
           when left(status_string, 6) similar to '%(3|4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever30dpd6m_flag,
       case
           when left(status_string, 6) similar to '%(4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever60dpd6m_flag,
       case
           when left(status_string, 12) similar to '%(3|4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever30dpd12m_flag,
       case
           when left(status_string, 12) similar to '%(4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever60dpd12m_flag,
       case
           when left(status_string, 18) similar to '%(3|4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever30dpd18m_flag,
       case
           when left(status_string, 18) similar to '%(4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever60dpd18m_flag,
       case
           when left(status_string, 18) similar to '%(5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever90dpd18m_flag,
       case
           when left(status_string, 24) similar to '%(3|4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever30dpd24m_flag,
       case
           when left(status_string, 24) similar to '%(4|5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever60dpd24m_flag,
       case
           when left(status_string, 24) similar to '%(5|6|D|W)%' then 1
           when status_string is not null then 0
           else null end                                       as ever90dpd24m_flag,

       case
           when left(status_string, 6) similar to '%(B)%' then 1
           when status_string is not null then 0
           else null end                                       as everbk6m_flag,
       case
           when left(status_string, 12) similar to '%(B)%' then 1
           when status_string is not null then 0
           else null end                                       as everbk12m_flag,
       case
           when left(status_string, 18) similar to '%(B)%' then 1
           when status_string is not null then 0
           else null end                                       as everbk18m_flag,
       case
           when left(status_string, 24) similar to '%(B)%' then 1
           when status_string is not null then 0
           else null end                                       as everbk24m_flag,
       case
           when left(status_string, 6) similar to '%(F)%' then 1
           when status_string is not null then 0
           else null end                                       as everforb6m_flag,
       case
           when left(status_string, 12) similar to '%(F)%' then 1
           when status_string is not null then 0
           else null end                                       as everforb12m_flag,
       case
           when left(status_string, 18) similar to '%(F)%' then 1
           when status_string is not null then 0
           else null end                                       as everforb18m_flag,
       case
           when left(status_string, 24) similar to '%(F)%' then 1
           when status_string is not null then 0
           else null end                                       as everforb24m_flag
from applications_file af
         LEFT JOIN loan_history_string_202202 h ON h.id = af.id
         LEFT JOIN loans_blended_daily lbd ON lbd.id = af.id
         LEFT JOIN pl_application_pull_202202 v ON v.app_id = af.id
         LEFT JOIN credit_pull cp ON v.credit_pull_id = cp.credit_pull_id
         left join borrowers pri on pri.user_id = af.applicant_id
         left join borrowers cob on cob.user_id = af.coborrower_applicant_id
WHERE (af.application_type = ANY
       (ARRAY ['PL'::text]))
  AND af.id <> 0
  AND af.date_start >= '2021-04-01'
  AND af.date_start <= '2022-01-31'