select 
af.id,
af.date_fund,
af.applicant_id,
af.dw_application_id
from dwmart.applications_file  af
where  af.date_fund >='${CONDITION}'
and af.date_fund <='${CONDITION}'
and af.application_type='PL'
limit 10


