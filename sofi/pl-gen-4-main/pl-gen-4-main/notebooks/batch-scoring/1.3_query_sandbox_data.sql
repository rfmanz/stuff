-- for PL proxy trade data query

create table "pl_gen4_model"."tradeline_202201" as
select e.*, l.*
from "experian_ascend_sandbox_db"."trade" e
         join "pl_gen4_model"."linked_v2" l
              on e.experian_consumer_key = l.xref_c1 and e.base_ts in ('2022-01-29')
                          
              
-- Personal_Loan: Personal installment loans excluding mortgage, auto, and student loans
create table "pl_gen4_model"."pl_tradeline_202201" as
with a as (select *,
                  case
                      when
                              (enhanced_type_code in
                               ('00', '01', '02', '03', '06', '09', '0A', '0F', '10', '11', '12', '13', '14', '17',
                                '1A',
                                '1B', '1C', '20', '21', '22', '23', '30', '31', '3A', '65', '66', '67', '68', '69',
                                '6A',
                                '78', '7B'))
                              and not
                                  ((enhanced_type_code in ('3A')
                                      or
                                    (enhanced_type_code in ('13')
                                        and
                                     kob in
                                     ('AC', 'AF', 'AL', 'AN', 'AU', 'AZ', 'BB', 'BS', 'FA', 'FC', 'FF', 'FP', 'FS',
                                      'FZ')))
                                      or
                                   (enhanced_type_code in ('00')
                                       or
                                    (kob in ('AF', 'AL', 'AN', 'AU', 'AZ', 'FA')
                                        and
                                     enhanced_type_code in
                                     ('00', '01', '02', '03', '06', '09', '0A', '0F', '10', '11', '12', '14', '17',
                                      '1A',
                                      '1B', '1C', '20', '21', '22', '23', '30', '31', '65', '66', '67', '68', '69',
                                      '6A',
                                      '78', '7B'))))
                              and not
                                  (substr(kob, 1, 1) in ('E')
                                      or
                                   enhanced_type_code in ('12')) then 1
                      else 0 end as pl_ind
           from "pl_gen4_model"."tradeline_202201")
select *
from a where pl_ind=1


-- prepare for premier and trended attributes

create table "pl_gen4_model"."month_table_base_ts_v4" as
select distinct base_ts
from "experian_ascend_sandbox_db"."vantage_4"
order by base_ts


create table "pl_gen4_model"."month_table_base_ts_v5" as
select substring(base_ts, 1, 7)                      as base_ts_1,
       lag(base_ts, 1) over (order by base_ts asc)   as base_ts_pre,
       lead(base_ts, 23) over (order by base_ts asc) as base_ts_post24,
       lead(base_ts, 17) over (order by base_ts asc) as base_ts_post18
from "pl_gen4_model"."month_table_base_ts_v4"


create table "pl_gen4_model"."linked_v4" as
select cast(a.xref_c1 as bigint)                       as xref_c1,
       a.applicant_type,
       a.date_start,
       a.loan_info,
       b.base_ts_pre,
       b.base_ts_post24,
       b.base_ts_post18
from "pl_gen4_model"."linked" a
         left join "pl_gen4_model"."month_table_base_ts_v5" b on substring(a.date_start, 1, 7) = b.base_ts_1


-- query post premier attributes in RI model
CREATE TABLE "pl_gen4_model"."premier_v13_post_refresh" AS
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2022-01-29')
              
insert into "pl_gen4_model"."premier_v13_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-12-29')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-11-27')
              
              
insert into "pl_gen4_model"."premier_v13_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-10-30')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-09-29')
              
              
insert into "pl_gen4_model"."premier_v13_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-08-28')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-07-31')
              
insert into "pl_gen4_model"."premier_v13_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-06-30')
              
              
              
-- query post trended attributes in RI model

CREATE TABLE "pl_gen4_model"."trended_3d_v1_1_post_refresh" AS
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2022-01-29')
              
insert into "pl_gen4_model"."trended_3d_v1_1_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-12-29')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-11-27')
              
              
insert into "pl_gen4_model"."trended_3d_v1_1_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-10-30')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-09-29')
              
              
insert into "pl_gen4_model"."trended_3d_v1_1_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-08-28')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-07-31')
              
insert into "pl_gen4_model"."trended_3d_v1_1_post_refresh" 
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-06-30')



--- query post vantage score in RI model

CREATE TABLE "pl_gen4_model"."vantage_post_refresh" AS
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2022-01-29')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-12-29')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-11-27')
              
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-10-30')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-09-29')
              
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-08-28')
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-07-31')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."vantage_4" e
         join "pl_gen4_model"."linked_v4" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_post24 = e.base_ts and e.base_ts in ('2021-06-30')
              
--- query premier attributes for model scoring              
              
create table "pl_gen4_model"."linked_refresh" as
select cast(a.xref_c1 as bigint)                       as xref_c1,
       a.applicant_type,
       a.date_start,
       a.loan_info as id,
       b.base_ts_pre,
       b.base_ts_post24,
       b.base_ts_post18
from "pl_gen4_model"."refresh" a
         left join "pl_gen4_model"."month_table_base_ts_v5" b on substring(a.date_start, 1, 7) = b.base_ts_1
         
         
insert into "pl_gen4_model"."premier_v13_pre_21"
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-03-31')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-04-28')
              
              
insert into "pl_gen4_model"."premier_v13_pre_21"
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-05-29')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-06-30')
              
insert into "pl_gen4_model"."premier_v13_pre_21"
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-07-31')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-08-28')

insert into "pl_gen4_model"."premier_v13_pre_21"
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-09-29')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-10-30')
              
insert into "pl_gen4_model"."premier_v13_pre_21"
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-11-27')


create table "pl_gen4_model"."premier_v13_pre_22" as
select e.*, l.*
from "experian_ascend_sandbox_db"."premier_1_3" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-12-29')
              
              
              
              
--- query trended attributes for model scoring                
              
              
insert into "pl_gen4_model"."2021_trended_3d_v1_1_pre_2"
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-03-31')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-04-28')
              
              
insert into "pl_gen4_model"."2021_trended_3d_v1_1_pre_2"
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-05-29')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-06-30')
              
insert into "pl_gen4_model"."2021_trended_3d_v1_1_pre_2"
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-07-31')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-08-28')

insert into "pl_gen4_model"."2021_trended_3d_v1_1_pre_2"
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-09-29')
              
union all
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-10-30')
              
insert into "pl_gen4_model"."2021_trended_3d_v1_1_pre_2"
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-11-27')


create table "pl_gen4_model"."2022_trended_3d_v1_1_pre" as
select e.*, l.*
from "experian_ascend_sandbox_db"."trended_3d_v1_1" e
         join "pl_gen4_model"."linked_refresh" l
              on e.experian_consumer_key = l.xref_c1 and l.base_ts_pre = e.base_ts and e.base_ts in ('2021-12-29')