select im.party_id as user_id,
       aa.response::jsonb->>'availableBalance' as available_bal,
       aa.response::jsonb->>'presentBalance' as current_bal,
--        aa.response::jsonb->>'type' as currency???,
       aa.response::jsonb->>'category' as account_category,
--        aa.response::jsonb->>'type' as account_name,
       aa.response::jsonb->>'type' as account_type,
       aa.response::jsonb->>'accountNumber' as ach_account_number,
       aa.response::jsonb->>'routing' as ach_routing_number,
       aa.fetched_from_quovo_dt as current_as_of_dt
from account_auth aa
join connection_accounts ca on aa.account_id = ca.account_id
join customer_connections cc on ca.connection_id = cc.connection_id
join identity_mapping im on cc.quovo_id = im.quovo_id
where im.party_id in <USER_ID>