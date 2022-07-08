select auth::jsonb->>'partyId' as user_id,
       auth::jsonb->'balances'->>'available' as available_bal,
       auth::jsonb->'balances'->>'current' as current_bal,
       auth::jsonb->'balances'->>'isoCurrencyCode' as currency,
       auth::jsonb->'account'->>'accountCategory' as account_category,
       auth::jsonb->'account'->>'accountName' as account_name,
       auth::jsonb->'account'->>'accountType' as account_type,
       auth::jsonb->'numbers'->'ach'->>'account' as ach_account_number,
       auth::jsonb->'numbers'->'ach'->>'routing' as ach_routing_number,
       auth::jsonb->>'owners' as owners,
       current_as_of_dt,
       created_by
from account_auth
where auth::jsonb->>'partyId' in <USER_ID>
