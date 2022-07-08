WITH bid AS -- get borrower_id, banking_account_id for all SoFi money accounts 30+ days old opened after Jan 1st 2019
    (SELECT DISTINCT badf.borrower_id
     FROM banking_account_daily_facts badf
     JOIN banking_accounts ba ON badf.banking_account_id = ba.banking_account_id
     JOIN borrowers b ON b.borrower_id=badf.borrower_id
     WHERE product_type='SoFi Money Cash Account'
         AND ba.active_flag=TRUE)
SELECT bid.borrower_id,
       b.user_id,
       badf.current_account_balance as latest_acc_bal,
       ba.account_open_date,
       ba.account_closed_date,
       ba.account_closed_reason,
       ba.account_restricted_reason,
       bf.funded_student_loan_ind,
       bf.funded_personal_loan_ind,
       bf.funded_mortgage_ind,
       bf.funded_wealth_account_ind,
       b.date_of_birth,
	   b.sofi_employee_ind,
       b.first_name,
       b.last_name,
       b.current_addr_line_1
FROM bid
JOIN banking_account_daily_facts badf ON badf.borrower_id = bid.borrower_id
JOIN
    (SELECT borrower_id,
            max(reporting_date_id) AS reporting_date_id
     FROM banking_account_daily_facts
     GROUP BY 1) bc ON badf.borrower_id = bc.borrower_id
AND badf.reporting_date_id = bc.reporting_date_id
LEFT JOIN banking_accounts ba ON ba.banking_account_id=badf.banking_account_id
LEFT JOIN borrowers_file bf ON bf.borrower_id = bid.borrower_id
LEFT JOIN borrowers b ON b.borrower_id = bid.borrower_id
