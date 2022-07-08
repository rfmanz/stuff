-- user_id/borrower_id level metadata (not for use in model)
WITH bid AS
    (SELECT DISTINCT badf.borrower_id
     FROM banking_account_daily_facts badf
     JOIN banking_accounts ba ON badf.banking_account_id = ba.banking_account_id
     WHERE product_type='SoFi Money Cash Account'
     AND ba.active_flag=TRUE)
SELECT bid.borrower_id,
       b.user_id,
       b.date_of_birth,
       b.sofi_employee_ind,
       b.first_name,
       b.last_name,
       b.current_addr_line_1
FROM bid
JOIN borrowers b ON b.borrower_id = bid.borrower_id
